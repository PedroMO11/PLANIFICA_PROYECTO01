"""Politica economica de tres acciones, con cupo de revision y costos calibrados.

Separacion score / politica / accion
------------------------------------
El modelo produce ``p``. La politica convierte ``p`` y el monto en un costo
esperado. La accion es el minimo de esos costos, sujeta al cupo de revision. Esa
separacion es deliberada: permite cambiar el modelo sin tocar la politica y
reportar por separado el comportamiento del clasificador y el del sistema.

Dos decisiones de diseno que el modelo de costos impone
-------------------------------------------------------
La accion se elige por ``argmin`` de los tres costos y no por umbrales sobre ``p``,
porque el punto de indiferencia depende del monto (ver ``Policy``). Y ``c_FP`` no se
fija por intuicion sino que se deriva del objetivo de bloqueo de legitimas
(ver ``calibrate_fp_cost``), que es la cantidad que la operacion observa y restringe.

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
    """Regla de decision y cupo diario. Se congela antes del test.

    ``rule`` selecciona como se propone la accion:

    * ``argmin``: el minimo de los tres costos esperados, caso por caso. Es la
      regla de operacion.
    * ``umbral``: dos cortes globales sobre ``p``. Se conserva como referencia
      para poder cuantificar cuanto cuesta ignorar el monto.

    Un umbral global sobre ``p`` no puede ser optimo bajo este modelo de costos,
    porque el punto de indiferencia entre aprobar y bloquear es
    ``p* = c_FP / (monto + c_FP)`` y por lo tanto depende del monto. Un corte fijo
    bloquea de mas en los montos bajos y de menos en los altos. Medido sobre los
    cuatro bloques de test, la regla de umbrales cuesta 1,6262 UM/tx frente a
    1,5054 de la regla economica, con el mismo numero de revisiones y mas bloqueo
    de legitimas. Por eso la regla de operacion es ``argmin``.
    """

    tau_low: float
    tau_high: float
    daily_capacity: int = 150
    delta: float = 0.0
    rule: str = "argmin"
    version: str = "politica_v2"

    def __post_init__(self) -> None:
        if self.rule not in ("argmin", "umbral"):
            raise ValueError("Regla de decision desconocida: %s" % self.rule)
        if self.tau_low > self.tau_high:
            raise ValueError("tau_low (%.6f) no puede superar a tau_high (%.6f)" % (self.tau_low, self.tau_high))

    def propose(self, probability: np.ndarray, costs: dict[str, np.ndarray]) -> np.ndarray:
        """Accion propuesta, ANTES de aplicar el cupo."""
        if self.rule == "umbral":
            return self.zone(probability)
        stacked = np.vstack([costs[APROBAR], costs[REVISAR], costs[BLOQUEAR]])
        return np.asarray(ACCIONES, dtype=object)[stacked.argmin(axis=0)]

    def zone(self, probability: np.ndarray) -> np.ndarray:
        """Zona propuesta por los dos umbrales globales."""
        probability = np.asarray(probability, dtype=float)
        low = self.tau_low - self.delta
        high = self.tau_high + self.delta
        out = np.full(probability.shape, REVISAR, dtype=object)
        out[probability < low] = APROBAR
        out[probability >= high] = BLOQUEAR
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "tau_low": self.tau_low,
            "tau_high": self.tau_high,
            "daily_capacity": self.daily_capacity,
            "delta": self.delta,
            "version": self.version,
        }


def policy_from_dict(data: dict[str, Any], *, daily_capacity: int | None = None) -> Policy:
    """Reconstruye la politica congelada a partir de su serializacion.

    Existe para que la regla de decision y el cupo viajen juntos desde el
    prerregistro hasta el servicio. Reconstruir el objeto campo por campo en cada
    consumidor permitia que uno de ellos olvidara ``rule`` y decidiera distinto que
    el backtest sin que nada fallara.

    Acepta tambien una ``Policy`` ya construida. El mismo llamador recibe la
    seleccion en memoria o releida de JSON segun se haya reanudado la corrida, y
    distinguir ambos casos en cada sitio era una fuente de errores.
    """
    if isinstance(data, Policy):
        return data
    return Policy(
        tau_low=float(data["tau_low"]),
        tau_high=float(data["tau_high"]),
        daily_capacity=int(data.get("daily_capacity", daily_capacity or 150)),
        delta=float(data.get("delta", 0.0)),
        rule=str(data.get("rule", "argmin")),
        version=str(data.get("version", "politica_v2")),
    )


def cost_model_from_dict(data: dict[str, Any]) -> CostModel:
    """Reconstruye el modelo de costos congelado.

    ``c_fp`` se calibra en desarrollo y queda sellado, de modo que el unico origen
    valido es el prerregistro o el manifiesto del paquete, nunca el config.
    """
    if isinstance(data, CostModel):
        return data
    if "c_fp" not in data:
        raise KeyError(
            "El modelo de costos congelado no trae c_fp. Se calibra en desarrollo "
            "y se sella en el prerregistro; no debe leerse del config."
        )
    return CostModel(
        c_fp=float(data["c_fp"]),
        c_review=float(data["c_review"]),
        r_h=float(data["r_h"]),
        f_h=float(data["f_h"]),
    )


def implied_thresholds(costs_model: CostModel, amount: float) -> dict[str, float]:
    """Umbrales de indiferencia que la regla economica aplica a un monto dado.

    La regla ``argmin`` no fija cortes, pero induce uno por monto. Calcularlos
    permite describir la politica en terminos de ``p`` sin cambiar la decision.
    """
    grid = np.linspace(0.0, 1.0, 100_001)
    amounts = np.full_like(grid, float(amount))
    costs = costs_model.expected_costs(grid, amounts)
    stacked = np.vstack([costs[APROBAR], costs[REVISAR], costs[BLOQUEAR]])
    action = stacked.argmin(axis=0)
    no_aprobar = np.flatnonzero(action != 0)
    bloquear = np.flatnonzero(action == 2)
    return {
        "monto": float(amount),
        "tau_low": float(grid[no_aprobar[0]]) if no_aprobar.size else 1.0,
        "tau_high": float(grid[bloquear[0]]) if bloquear.size else 1.0,
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
    zones = policy.propose(probability, costs)

    rows: list[dict[str, Any]] = []
    for i, event_id in enumerate(event_ids):
        previous = ledger.already_decided(event_id)
        if previous is not None:
            # Reintento: se devuelve la decision confirmada, sin consumir cupo.
            rows.append({**previous, "idempotente": True})
            continue

        proposed = zones[i]
        action = proposed
        reason = "%s_%s" % (policy.rule, proposed)
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

def calibrate_fp_cost(
    probability: np.ndarray,
    amount: np.ndarray,
    day: np.ndarray,
    labels: np.ndarray,
    event_ids: Sequence[Any],
    *,
    target_block_rate: float,
    grid: Sequence[float],
    c_review: float = 1.0,
    r_h: float = 0.90,
    f_h: float = 0.02,
    daily_capacity: int = 150,
    seed: int = 42,
) -> dict[str, Any]:
    """Deriva ``c_FP`` del objetivo operativo de bloqueo de legitimas.

    ``c_FP`` no es observable. Fijarlo por intuicion deja sin controlar la cantidad
    que el negocio si observa y si restringe, que es la fraccion de transacciones
    legitimas rechazadas. Con ``c_FP = 5`` la regla economica bloquea el 11,95 % de
    las legitimas sobre IEEE-CIS, un nivel que ninguna operacion de pagos acepta.

    Aqui se invierte la relacion: se declara el objetivo (``diagnostic_targets``,
    ``fpr_target``) y se busca el menor ``c_FP`` de la grilla que lo cumple. El
    valor resultante es el precio sombra de la restriccion. Se calibra una sola vez
    sobre la reserva de politica de desarrollo y queda congelado y sellado en el
    prerregistro, igual que los hiperparametros.

    Se elige el MENOR valor que cumple porque ``c_FP`` mas alto compra menos bloqueo
    a cambio de mas fraude aprobado. El minimo factible es el que respeta la
    restriccion sin sobrepagar.
    """
    legitimate = np.asarray(labels) == 0
    if not legitimate.any():
        raise ValueError("La reserva de calibracion no contiene transacciones legitimas")

    table: list[dict[str, Any]] = []
    for candidate in sorted(float(c) for c in grid):
        costs_model = CostModel(c_fp=candidate, c_review=c_review, r_h=r_h, f_h=f_h)
        policy = Policy(tau_low=0.0, tau_high=1.0, daily_capacity=daily_capacity, rule="argmin")
        decisions = decide_batch(
            probability, amount, day, event_ids, policy, costs_model,
            ledger=CapacityLedger(daily_capacity),
        )
        outcomes = simulate_outcomes(decisions, labels, costs_model, seed=seed)
        blocked = outcomes["legitima_bloqueada"].to_numpy()
        table.append({
            "c_fp": candidate,
            "tasa_bloqueo_legitimo": float(blocked[legitimate].mean()),
            "n_revisiones": int(outcomes["revision_admitida"].sum()),
            "monto_fraude_aprobado": float(outcomes.loc[outcomes["fraude_aprobado"], "monto"].sum()),
            "cumple_objetivo": float(blocked[legitimate].mean()) <= target_block_rate,
        })

    feasible = [row for row in table if row["cumple_objetivo"]]
    if not feasible:
        best = min(table, key=lambda row: row["tasa_bloqueo_legitimo"])
        LOGGER.warning(
            "Ningun c_FP de la grilla alcanza el objetivo de %.3f; se usa el mejor disponible (%.1f -> %.4f)",
            target_block_rate, best["c_fp"], best["tasa_bloqueo_legitimo"],
        )
    else:
        best = feasible[0]
        LOGGER.info(
            "c_FP calibrado en %.1f: bloqueo legitimo %.4f <= objetivo %.3f",
            best["c_fp"], best["tasa_bloqueo_legitimo"], target_block_rate,
        )
    return {
        "c_fp": float(best["c_fp"]),
        "objetivo": float(target_block_rate),
        "alcanzado": bool(best["cumple_objetivo"]),
        "tasa_bloqueo_legitimo": float(best["tasa_bloqueo_legitimo"]),
        "tabla": pd.DataFrame(table),
    }


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
            policy = Policy(tau_low=low, tau_high=high, daily_capacity=daily_capacity,
                            delta=delta, rule="umbral")
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
        rule="umbral",
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
                               rule="umbral", version="baseline_aprobar_todo"),
        "bloquear_todo": Policy(tau_low=0.0, tau_high=0.0, daily_capacity=daily_capacity,
                                rule="umbral", version="baseline_bloquear_todo"),
    }
