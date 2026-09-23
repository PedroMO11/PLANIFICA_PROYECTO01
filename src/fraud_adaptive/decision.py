"""Politica economica de tres acciones, con cupo de revision y costos calibrados.

El modelo produce ``p``; la politica convierte ``p`` y el monto en costos esperados
y la accion es la de menor costo, sujeta al cupo de revision. Separar modelo y
politica permite cambiar uno sin tocar el otro y reportar por separado el
clasificador y el sistema.

* La accion se elige por costo esperado porque el punto de indiferencia entre
  aprobar y bloquear depende del monto (ver ``Policy``).
* Revisar ocupa una plaza escasa, asi que debe ahorrar mas que el precio sombra
  del cupo (ver ``calibrate_review_price``).
* ``c_FP`` se deriva del objetivo de bloqueo de legitimas (ver
  ``calibrate_fp_cost``).

Costos esperados por accion::

    E[aprobar]  = p * monto                        (si es fraude, se pierde el monto)
    E[bloquear] = (1 - p) * c_FP                   (si era legitima, friccion)
    E[revisar]  = c_R + p*(1-r_H)*monto + (1-p)*f_H*c_FP

La revision tiene un costo fijo, deja pasar el fraude que el analista no detecta
(``1-r_H``) y bloquea legitimas por error (``f_H``). Con montos bajos, revisar
cuesta mas que asumir la perdida.

El cupo se reserva al admitir cada caso, de forma irrevocable y por ``event_id``.
Ordenar el dia completo por ``p*monto`` exigiria conocer transacciones futuras; la
prioridad solo ordena la atencion de los casos ya admitidos. Sin cupo, el caso
recibe la accion automatica mas barata.
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
    """Regla de decision y cupo diario, fijados antes del test.

    ``rule`` define como se propone la accion:

    * ``argmin``: por costo esperado, con el cupo racionado por ``review_price``.
      Es la regla de operacion.
    * ``umbral``: dos cortes globales sobre ``p``, que se conservan para medir cuanto
      cuesta ignorar el monto.

    El punto de indiferencia entre aprobar y bloquear es ``p* = c_FP / (monto + c_FP)``,
    asi que un corte fijo bloquea de mas en montos bajos y de menos en montos altos.
    Sobre la reserva de desarrollo, la regla de umbrales cuesta 1,9696 UM/tx y la
    economica 1,5895, aunque los umbrales se eligieron sobre esa misma reserva.

    ``review_price`` es el precio sombra del cupo diario. Un caso se revisa solo si

        min(E[aprobar], E[bloquear]) - E[revisar] > review_price

    Con ``review_price = 0`` la regla propone 704 revisiones diarias para 150 plazas,
    que se llenan por orden de llegada. Sin el precio, la regla economica costaba
    2,3090 UM/tx frente a 2,2339 del mejor umbral fijo, que raciona el cupo de forma
    implicita; con el precio es 19 % mas barata.
    """

    tau_low: float
    tau_high: float
    daily_capacity: int = 150
    delta: float = 0.0
    rule: str = "argmin"
    review_price: float = 0.0
    version: str = "politica_v3"

    def __post_init__(self) -> None:
        if self.rule not in ("argmin", "umbral"):
            raise ValueError("Regla de decision desconocida: %s" % self.rule)
        if self.tau_low > self.tau_high:
            raise ValueError("tau_low (%.6f) no puede superar a tau_high (%.6f)" % (self.tau_low, self.tau_high))

    def propose(self, probability: np.ndarray, costs: dict[str, np.ndarray]) -> np.ndarray:
        """Accion propuesta antes de aplicar el cupo."""
        if self.rule == "umbral":
            return self.zone(probability)
        automatica = np.where(costs[APROBAR] <= costs[BLOQUEAR], APROBAR, BLOQUEAR)
        ahorro = np.minimum(costs[APROBAR], costs[BLOQUEAR]) - costs[REVISAR]
        return np.where(ahorro > self.review_price, REVISAR, automatica).astype(object)

    def review_saving(self, costs: dict[str, np.ndarray]) -> np.ndarray:
        """Cuanto ahorra revisar frente a la mejor accion automatica."""
        return np.minimum(costs[APROBAR], costs[BLOQUEAR]) - costs[REVISAR]

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
            "review_price": self.review_price,
            "version": self.version,
        }


def policy_from_dict(data: dict[str, Any], *, daily_capacity: int | None = None) -> Policy:
    """Reconstruye la politica a partir de su serializacion.

    Es el unico constructor que usan el backtest, el replay y el servicio, de modo
    que todos aplican la misma regla y el mismo cupo. Acepta tambien una ``Policy``
    ya construida, porque la seleccion puede venir en memoria o releida de JSON al
    reanudar.
    """
    if isinstance(data, Policy):
        return data
    return Policy(
        tau_low=float(data["tau_low"]),
        tau_high=float(data["tau_high"]),
        daily_capacity=int(data.get("daily_capacity", daily_capacity or 150)),
        delta=float(data.get("delta", 0.0)),
        rule=str(data.get("rule", "argmin")),
        review_price=float(data.get("review_price", 0.0)),
        version=str(data.get("version", "politica_v3")),
    )


def cost_model_from_dict(data: dict[str, Any]) -> CostModel:
    """Reconstruye el modelo de costos.

    ``c_fp`` se calibra en desarrollo, de modo que se lee del prerregistro o del
    manifiesto del paquete.
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


def implied_thresholds(costs_model: CostModel, amount: float,
                       policy: "Policy | None" = None) -> dict[str, float]:
    """Umbrales de indiferencia que la regla economica aplica a un monto dado.

    La regla no fija cortes, pero induce uno por monto. Calcularlos permite
    describir la politica en terminos de ``p`` sin cambiar la decision.
    """
    grid = np.linspace(0.0, 1.0, 100_001)
    amounts = np.full_like(grid, float(amount))
    costs = costs_model.expected_costs(grid, amounts)
    policy = policy or Policy(tau_low=0.0, tau_high=1.0)
    action = policy.propose(grid, costs)
    no_aprobar = np.flatnonzero(action != APROBAR)
    bloquear = np.flatnonzero(action == BLOQUEAR)
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

    Un ``event_id`` ya resuelto devuelve su decision anterior sin consumir otro cupo,
    de modo que reintentar o reanudar no infla las revisiones.
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
    """Aplica la politica a una secuencia de eventos en orden temporal.

    Recorre los eventos uno a uno porque el estado del cupo depende de lo ya
    decidido; vectorizarlo exigiria conocer el dia completo de antemano.
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
    """Desenlace y costo observado, calculados cuando la etiqueta ya maduro.

    El analista simulado se resuelve aqui, en el evaluador, y no en el momento de
    decidir; la accion emitida no cambia. El sorteo del analista es determinista
    por ``event_id``, asi que dos corridas o una reanudacion dan el mismo veredicto.
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

def calibrate_review_price(
    probability: np.ndarray,
    amount: np.ndarray,
    day: np.ndarray,
    costs_model: CostModel,
    *,
    daily_capacity: int,
    tolerance: float = 1e-4,
    max_iter: int = 60,
) -> dict[str, Any]:
    """Deriva el precio sombra del cupo diario de revision.

    Sin restriccion, la regla de menor costo esperado propone 704 revisiones
    diarias en la reserva de desarrollo para un cupo de 150, y las plazas se llenan
    por orden de llegada. El multiplicador de Lagrange de la restriccion es el
    precio que debe superar el ahorro de revisar:

        min(E[aprobar], E[bloquear]) - E[revisar] > lambda

    ``lambda`` es el menor valor que deja la demanda diaria media dentro del cupo,
    hallado por biseccion sobre una funcion monotona decreciente. No usa etiquetas;
    en IEEE-CIS queda a 0,7 % del valor que minimiza el costo observado.
    """
    day = np.asarray(day)
    n_days = max(1, len(np.unique(day)))
    saving = costs_model.expected_costs(probability, amount)
    saving = np.minimum(saving[APROBAR], saving[BLOQUEAR]) - saving[REVISAR]

    def demand(price: float) -> float:
        return float((saving > price).sum()) / n_days

    if demand(0.0) <= daily_capacity:
        # La regla sin restriccion ya cabe en el cupo.
        LOGGER.info("El cupo no restringe: demanda de %.1f/dia con precio cero", demand(0.0))
        return {"review_price": 0.0, "demanda_diaria": demand(0.0),
                "cupo_diario": daily_capacity, "restringe": False}

    low, high = 0.0, float(max(saving.max(), 1.0))
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        if demand(mid) > daily_capacity:
            low = mid
        else:
            high = mid
        if high - low < tolerance:
            break
    price = high
    LOGGER.info(
        "Precio sombra del cupo: %.4f. Demanda %.1f/dia frente a un cupo de %d (sin precio: %.1f)",
        price, demand(price), daily_capacity, demand(0.0),
    )
    return {
        "review_price": float(price),
        "demanda_diaria": demand(price),
        "demanda_diaria_sin_precio": demand(0.0),
        "cupo_diario": daily_capacity,
        "restringe": True,
    }


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
    review_price: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Deriva ``c_FP`` del objetivo operativo de bloqueo de legitimas.

    ``c_FP`` no es observable, pero la fraccion de legitimas rechazadas si. Con
    ``c_FP = 5`` la regla economica bloquea el 11,17 % de las legitimas en IEEE-CIS.
    Se declara el objetivo (``diagnostic_targets.fpr_target``) y se busca el menor
    ``c_FP`` de la grilla que lo cumple; en IEEE-CIS resulta 25 UM, con 0,89 %.

    Se toma el menor valor factible porque un ``c_FP`` mayor reduce el bloqueo a
    cambio de aprobar mas fraude. Se calibra una vez sobre la reserva de politica de
    desarrollo y queda sellado en el prerregistro.
    """
    legitimate = np.asarray(labels) == 0
    if not legitimate.any():
        raise ValueError("La reserva de calibracion no contiene transacciones legitimas")

    table: list[dict[str, Any]] = []
    for candidate in sorted(float(c) for c in grid):
        costs_model = CostModel(c_fp=candidate, c_review=c_review, r_h=r_h, f_h=f_h)
        policy = Policy(tau_low=0.0, tau_high=1.0, daily_capacity=daily_capacity,
                        rule="argmin", review_price=review_price)
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

    Recorre la grilla de pares de cuantiles del score, incluidos los pares iguales
    (sin zona gris) y los bordes de aprobar todo y bloquear todo, y minimiza el
    costo. Desempates, en orden: menor bloqueo de legitimas, menor demanda de
    revision y zona gris mas estrecha.
    """
    probability = np.asarray(probability, dtype=float)
    candidates = sorted({float(np.quantile(probability, q)) for q in quantiles})
    # Los bordes permiten elegir una sola accion para todos los casos.
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

    ``aprobar_todo`` minimiza la friccion y maximiza la perdida por fraude; es la
    cota que el sistema debe superar para justificarse economicamente.
    """
    return {
        "aprobar_todo": Policy(tau_low=1.0 + 1e-9, tau_high=1.0 + 1e-9, daily_capacity=daily_capacity,
                               rule="umbral", version="baseline_aprobar_todo"),
        "bloquear_todo": Policy(tau_low=0.0, tau_high=0.0, daily_capacity=daily_capacity,
                                rule="umbral", version="baseline_bloquear_todo"),
    }
