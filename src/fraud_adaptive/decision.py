"""Politica economica de tres acciones, umbrales por costo y cola con cupo.

Separacion score / politica / accion
------------------------------------
El modelo produce ``p``. La politica convierte ``p`` y el monto en un costo
esperado. La accion es el minimo de esos costos, sujeta al cupo de revision. Esa
separacion es deliberada: permite cambiar el modelo sin tocar la politica y
reportar por separado el comportamiento del clasificador y el del sistema.

Costos esperados por accion
---------------------------
    E[aprobar]  = p * monto                        (si es fraude, se pierde el monto)
    E[bloquear] = (1 - p) * c_FP                   (si era legitima, friccion)
    E[revisar]  = c_R + p*(1-r_H)*monto + (1-p)*f_H*c_FP

La revision paga un costo fijo y deja pasar el fraude que el analista no detecta
(``1-r_H``) y bloquea legitimas que marca por error (``f_H``). Por eso no siempre
gana: con montos bajos, revisar cuesta mas que asumir la perdida.

Causalidad de la cola (C12)
---------------------------
El cupo se reserva al ADMITIR, de forma irrevocable y por ``event_id``. No se
ordena el dia completo por ``p*monto`` para quedarse con los mejores 150: eso
exigiria conocer transacciones que aun no ocurrieron. La prioridad ordena el
SERVICIO entre los ya admitidos. Cuando el cupo se agota, el caso cae a la mejor
de las dos acciones automaticas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("fraud_adaptive.decision")

APROBAR = "aprobar"
REVISAR = "revisar"
BLOQUEAR = "bloquear"
ACCIONES = (APROBAR, REVISAR, BLOQUEAR)


# --------------------------------------------------------------------------- costos

@dataclass(frozen=True)
class CostModel:
    """Parametros economicos y del analista simulado."""

    c_fp: float = 5.0
    c_review: float = 1.0
    r_h: float = 0.90
    f_h: float = 0.02

    def expected_costs(self, probability: np.ndarray, amount: np.ndarray) -> dict[str, np.ndarray]:
        probability = np.asarray(probability, dtype=float)
        amount = np.asarray(amount, dtype=float)
        return {
            APROBAR: probability * amount,
            BLOQUEAR: (1.0 - probability) * self.c_fp,
            REVISAR: self.c_review
            + probability * (1.0 - self.r_h) * amount
            + (1.0 - probability) * self.f_h * self.c_fp,
        }

    def to_dict(self) -> dict[str, float]:
        return {"c_fp": self.c_fp, "c_review": self.c_review, "r_h": self.r_h, "f_h": self.f_h}


@dataclass(frozen=True)
class Policy:
    """Dos umbrales globales y el cupo diario. Se congela antes del test."""

    tau_low: float
    tau_high: float
    daily_capacity: int = 150
    delta: float = 0.0
    version: str = "politica_v1"

    def __post_init__(self) -> None:
        if self.tau_low > self.tau_high:
            raise ValueError("tau_low (%.6f) no puede superar a tau_high (%.6f)" % (self.tau_low, self.tau_high))

    def zone(self, probability: np.ndarray) -> np.ndarray:
        """Zona propuesta por los umbrales, ANTES de aplicar el cupo."""
        probability = np.asarray(probability, dtype=float)
        low = self.tau_low - self.delta
        high = self.tau_high + self.delta
        out = np.full(probability.shape, REVISAR, dtype=object)
        out[probability < low] = APROBAR
        out[probability >= high] = BLOQUEAR
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau_low": self.tau_low,
            "tau_high": self.tau_high,
            "daily_capacity": self.daily_capacity,
            "delta": self.delta,
            "version": self.version,
        }


def fallback_action(costs: dict[str, np.ndarray], index: int) -> str:
    """Accion automatica cuando no queda cupo: la mas barata entre aprobar y bloquear."""
    return APROBAR if costs[APROBAR][index] <= costs[BLOQUEAR][index] else BLOQUEAR


# --------------------------------------------------------------------------- cola con cupo

@dataclass
class CapacityLedger:
    """Reserva de cupo diaria, atomica e idempotente por ``event_id``.

    La idempotencia es lo que hace seguro reintentar un evento tras un fallo: un
    ``event_id`` ya resuelto devuelve su decision anterior y NO consume un segundo
    cupo. Sin esto, una reanudacion de la corrida nocturna inflaria las revisiones.
    """

    daily_capacity: int = 150
    used_by_day: dict[int, int] = field(default_factory=dict)
    decided: dict[Any, dict[str, Any]] = field(default_factory=dict)

    def remaining(self, day: int) -> int:
        return max(0, self.daily_capacity - self.used_by_day.get(int(day), 0))

    def already_decided(self, event_id: Any) -> dict[str, Any] | None:
        return self.decided.get(event_id)

    def reserve(self, day: int, event_id: Any) -> bool:
        """Intenta reservar un cupo. Devuelve False si el dia esta lleno."""
        day = int(day)
        if self.remaining(day) <= 0:
            return False
        self.used_by_day[day] = self.used_by_day.get(day, 0) + 1
        return True

    def record(self, event_id: Any, record: dict[str, Any]) -> dict[str, Any]:
        self.decided[event_id] = record
        return record

    def stats(self) -> dict[str, Any]:
        used = np.array(list(self.used_by_day.values()), dtype=float) if self.used_by_day else np.array([0.0])
        return {
            "capacidad_diaria": self.daily_capacity,
            "dias_con_revisiones": int(len(self.used_by_day)),
            "revisiones_totales": int(used.sum()),
            "revisiones_dia_medio": float(used.mean()),
            "revisiones_dia_max": int(used.max()),
            "dias_en_capacidad_plena": int((used >= self.daily_capacity).sum()),
            "excedio_capacidad": bool((used > self.daily_capacity).any()),
        }


def decide_batch(
    probability: np.ndarray,
    amount: np.ndarray,
    day: np.ndarray,
    event_ids: Sequence[Any],
    policy: Policy,
    costs_model: CostModel,
    *,
    ledger: CapacityLedger | None = None,
) -> pd.DataFrame:
    """Aplica la politica a una secuencia de eventos EN ORDEN TEMPORAL.

    Recorre los eventos uno a uno porque el cupo es un recurso compartido cuyo
    estado depende de lo ya decidido. Vectorizarlo obligaria a conocer el dia
    completo de antemano, que es exactamente la ventaja irreal que C12 prohibe.
    """
    ledger = ledger if ledger is not None else CapacityLedger(policy.daily_capacity)
    costs = costs_model.expected_costs(probability, amount)
    zones = policy.zone(probability)

    rows: list[dict[str, Any]] = []
    for i, event_id in enumerate(event_ids):
        previous = ledger.already_decided(event_id)
        if previous is not None:
            # Reintento: se devuelve la decision confirmada, sin consumir cupo.
            rows.append({**previous, "idempotente": True})
            continue

        proposed = zones[i]
        action = proposed
        reason = "zona_" + str(proposed)
        admitted = False

        if proposed == REVISAR:
            if ledger.reserve(day[i], event_id):
                admitted = True
                reason = "revision_admitida"
            else:
                action = fallback_action(costs, i)
                reason = "overflow_cupo_menor_costo_esperado"

        record = {
            "event_id": event_id,
            "dia": int(day[i]),
            "p": float(probability[i]),
            "monto": float(amount[i]),
            "accion_propuesta": proposed,
            "accion": action,
            "motivo": reason,
            "revision_admitida": admitted,
            "prioridad": float(probability[i] * amount[i]),
            "costo_esp_aprobar": float(costs[APROBAR][i]),
            "costo_esp_revisar": float(costs[REVISAR][i]),
            "costo_esp_bloquear": float(costs[BLOQUEAR][i]),
            "cupo_restante": ledger.remaining(day[i]),
            "idempotente": False,
        }
        rows.append(ledger.record(event_id, record))

    return pd.DataFrame(rows)


def service_order(decisions: pd.DataFrame) -> pd.DataFrame:
    """Orden de atencion de las revisiones admitidas, por dia.

    Ordena por ``p*monto`` descendente entre los ADMITIDOS de cada dia. No cambia
    ninguna decision: solo dice a quien mira primero el analista.
    """
    admitted = decisions[decisions["revision_admitida"]].copy()
    if admitted.empty:
        return admitted
    admitted = admitted.sort_values(["dia", "prioridad", "event_id"], ascending=[True, False, True])
    admitted["orden_en_dia"] = admitted.groupby("dia").cumcount() + 1
    return admitted


# --------------------------------------------------------------------------- resultado simulado

def simulate_outcomes(
    decisions: pd.DataFrame,
    labels: np.ndarray,
    costs_model: CostModel,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Desenlace y costo observado, calculado SOLO cuando la etiqueta ha madurado.

    C28: el analista simulado se resuelve aqui, en el evaluador, nunca en el
    momento de decidir. La accion emitida no cambia al conocer el desenlace; lo
    unico que se agrega es cuanto costo produjo.

    El sorteo del analista es determinista por ``event_id``: dos corridas del mismo
    experimento dan el mismo veredicto, y reanudar no vuelve a sortear.
    """
    out = decisions.copy()
    labels = np.asarray(labels)
    out["y"] = labels

    ids = out["event_id"].to_numpy()
    # Hash estable por evento -> uniforme en [0,1). Independiente del orden de
    # llegada y reproducible entre corridas.
    draw = np.array([
        (abs(hash((seed, int(i) if isinstance(i, (int, np.integer)) else str(i)))) % 10_000_000) / 10_000_000.0
        for i in ids
    ])

    detected = np.zeros(len(out), dtype=bool)
    flagged = np.zeros(len(out), dtype=bool)
    reviewed = out["revision_admitida"].to_numpy(dtype=bool)

    # Fraude revisado: el analista lo detecta con probabilidad r_H.
    detected[reviewed & (labels == 1)] = draw[reviewed & (labels == 1)] < costs_model.r_h
    # Legitima revisada: el analista la bloquea por error con probabilidad f_H.
    flagged[reviewed & (labels == 0)] = draw[reviewed & (labels == 0)] < costs_model.f_h

    out["analista_detecto_fraude"] = detected
    out["analista_bloqueo_legitima"] = flagged

    action = out["accion"].to_numpy()
    amount = out["monto"].to_numpy()

    # Desenlace final: que ocurrio realmente con la transaccion.
    fraude_aprobado = ((action == APROBAR) & (labels == 1)) | (reviewed & (labels == 1) & ~detected)
    legitima_bloqueada = ((action == BLOQUEAR) & (labels == 0)) | (reviewed & (labels == 0) & flagged)
    fraude_detenido = ((action == BLOQUEAR) & (labels == 1)) | (reviewed & (labels == 1) & detected)

    out["fraude_aprobado"] = fraude_aprobado
    out["legitima_bloqueada"] = legitima_bloqueada
    out["fraude_detenido"] = fraude_detenido
    out["monto_fraude_evitado"] = np.where(fraude_detenido, amount, 0.0)

    out["costo_observado"] = (
        np.where(fraude_aprobado, amount, 0.0)
        + np.where(legitima_bloqueada, costs_model.c_fp, 0.0)
        + np.where(reviewed, costs_model.c_review, 0.0)
    )
    return out


def total_cost(outcomes: pd.DataFrame) -> dict[str, float]:
    """Costo observado agregado y sus componentes."""
    n = len(outcomes)
    if n == 0:
        return {"n": 0, "costo_total": float("nan"), "costo_por_tx": float("nan")}
    return {
        "n": n,
        "costo_total": float(outcomes["costo_observado"].sum()),
        "costo_por_tx": float(outcomes["costo_observado"].mean()),
        "monto_fraude_aprobado": float(outcomes.loc[outcomes["fraude_aprobado"], "monto"].sum()),
        "monto_fraude_evitado": float(outcomes["monto_fraude_evitado"].sum()),
        "n_legitimas_bloqueadas": int(outcomes["legitima_bloqueada"].sum()),
        "n_revisiones": int(outcomes["revision_admitida"].sum()),
        "n_aprobadas": int((outcomes["accion"] == APROBAR).sum()),
        "n_bloqueadas": int((outcomes["accion"] == BLOQUEAR).sum()),
    }


# --------------------------------------------------------------------------- seleccion de umbrales

def select_thresholds(
    probability: np.ndarray,
    amount: np.ndarray,
    day: np.ndarray,
    labels: np.ndarray,
    event_ids: Sequence[Any],
    costs_model: CostModel,
    *,
    quantiles: Sequence[float],
    daily_capacity: int = 150,
    delta: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Elige ``(tau_low, tau_high)`` por costo observado sobre la reserva de politica.

    Recorre la grilla de pares de cuantiles del score, incluyendo pares iguales
    (sin zona gris) y los bordes de aprobar todo / bloquear todo. Minimiza costo,
    nunca F1, y nunca asume 0.5.

    Desempates del plan, en orden: menor bloqueo de legitimas, menor demanda de
    revision y menor complejidad (zona gris mas estrecha).
    """
    probability = np.asarray(probability, dtype=float)
    candidates = sorted({float(np.quantile(probability, q)) for q in quantiles})
    # Bordes explicitos: permiten que la busqueda elija degenerar a una sola accion.
    candidates = sorted(set(candidates) | {0.0, 1.0 + 1e-9})

    results: list[dict[str, Any]] = []
    for i, low in enumerate(candidates):
        for high in candidates[i:]:
            policy = Policy(tau_low=low, tau_high=high, daily_capacity=daily_capacity, delta=delta)
            decisions = decide_batch(
                probability, amount, day, event_ids, policy, costs_model,
                ledger=CapacityLedger(daily_capacity),
            )
            outcomes = simulate_outcomes(decisions, labels, costs_model, seed=seed)
            summary = total_cost(outcomes)
            results.append({
                "tau_low": low,
                "tau_high": high,
                "costo_por_tx": summary["costo_por_tx"],
                "n_legitimas_bloqueadas": summary["n_legitimas_bloqueadas"],
                "n_revisiones": summary["n_revisiones"],
                "demanda_revision": int((decisions["accion_propuesta"] == REVISAR).sum()),
                "amplitud_zona_gris": high - low,
                "n_aprobadas": summary["n_aprobadas"],
                "n_bloqueadas": summary["n_bloqueadas"],
                "monto_fraude_evitado": summary["monto_fraude_evitado"],
            })

    table = pd.DataFrame(results)
    table = table.sort_values(
        ["costo_por_tx", "n_legitimas_bloqueadas", "demanda_revision", "amplitud_zona_gris"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    best = table.iloc[0]
    policy = Policy(
        tau_low=float(best["tau_low"]),
        tau_high=float(best["tau_high"]),
        daily_capacity=daily_capacity,
        delta=delta,
    )
    LOGGER.info(
        "Umbrales elegidos por costo: tau=(%.6f, %.6f) -> %.4f UM/tx",
        policy.tau_low, policy.tau_high, float(best["costo_por_tx"]),
    )
    return {"policy": policy, "grid": table, "best": best.to_dict()}


def baseline_policies(daily_capacity: int = 150) -> dict[str, Policy]:
    """Referencias simuladas, no politicas comerciales observadas.

    ``aprobar_todo`` es el limite inferior de friccion y superior de perdida por
    fraude; sirve como cota que el sistema debe batir para tener sentido economico.
    """
    return {
        "aprobar_todo": Policy(tau_low=1.0 + 1e-9, tau_high=1.0 + 1e-9, daily_capacity=daily_capacity,
                               version="baseline_aprobar_todo"),
        "bloquear_todo": Policy(tau_low=0.0, tau_high=0.0, daily_capacity=daily_capacity,
                                version="baseline_bloquear_todo"),
    }
