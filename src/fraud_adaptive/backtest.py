"""Motor de backtest prequential (test-then-train) con dos relojes.

Bucle por dia del periodo de test, para cada estrategia:

1. Si el dia es de actualizacion, construir la nueva version con datos maduros,
   pasarla por el gate y activarla si corresponde.
2. Puntuar los eventos del dia con la version activa, antes de conocer su etiqueta.
3. Decidir con la politica congelada y el cupo del dia.
4. Guardar la prediccion con su version, sin modificarla despues.
5. Madurar las etiquetas de hace L dias y recien entonces alimentar ADWIN y
   calcular los costos observados.

Las predicciones emitidas no se recalculan cuando llega una version nueva: quedan
atribuidas a la version que las emitio, como ocurriria en operacion.

El reloj de eventos se detiene mientras se ajusta un paquete. El backtest mide el
efecto de actualizar en el corte, sin la latencia real de un reentrenamiento ni el
tiempo de una aprobacion humana.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .adaptation import ModelPackage, build_package, promotion_gate
from .decision import (
    APROBAR, BLOQUEAR, REVISAR, CapacityLedger, CostModel, Policy,
    decide_batch, simulate_outcomes,
)
from .drift import BrierAdwin, CovariateMonitor, DriftLog, domain_classifier, score_drift
from .splits import Interval, TemporalConfig, VersionRoles, build_version_roles

LOGGER = logging.getLogger("fraud_adaptive.backtest")


@dataclass
class StrategyState:
    """Estado vivo de una estrategia durante el replay."""

    name: str
    kind: str
    window_days: int | None
    active: ModelPackage | None = None
    previous: ModelPackage | None = None
    ledger: CapacityLedger = None  # type: ignore[assignment]
    detectors: dict[str, BrierAdwin] = field(default_factory=dict)
    predictions: list[pd.DataFrame] = field(default_factory=list)
    version_changes: list[dict[str, Any]] = field(default_factory=list)
    gate_records: list[dict[str, Any]] = field(default_factory=list)
    unavailable_days: list[int] = field(default_factory=list)

    def detector_for(self, version_id: str, *, delta: float, clock: int) -> BrierAdwin:
        if version_id not in self.detectors:
            # Los detectores de versiones retiradas se conservan hasta que maduren
            # todas sus predicciones.
            self.detectors[version_id] = BrierAdwin(version_id=version_id, delta=delta, clock=clock)
        return self.detectors[version_id]


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    outcomes: pd.DataFrame
    drift_log: pd.DataFrame
    version_changes: pd.DataFrame
    gates: pd.DataFrame
    capacity: dict[str, Any]
    coverage: dict[str, Any]
    adwin_detections: pd.DataFrame


def run_backtest(
    frame: pd.DataFrame,
    *,
    strategies: Sequence[dict[str, Any]],
    temporal: TemporalConfig,
    family: str,
    model_config: dict[str, Any],
    model_base: dict[str, Any],
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    policy: Policy,
    cost_model: CostModel,
    calibration_params: dict[str, Any],
    adaptation_config: dict[str, Any],
    promotion_gates: dict[str, Any],
    monitor_panel: Sequence[str],
    reference_frame: pd.DataFrame,
    seed: int = 42,
    n_threads: int = 4,
    max_rows: int | None = None,
    encoding: dict[str, Any] | None = None,
    target: str = "isFraud",
    id_column: str = "TransactionID",
    on_day: Any = None,
    shared_initial: Sequence[str] = (),
    models_dir: str | None = None,
) -> BacktestResult:
    """Ejecuta el replay completo sobre el periodo de test.

    Con ``models_dir``, cada version activada se guarda como paquete completo, de
    modo que el servicio y el replay usan el mismo artefacto que produjo las
    metricas.
    """
    label_delay = temporal.label_delay_days
    days = frame["dia"].to_numpy()
    labels = frame[target].to_numpy()
    max_day = int(days.max())

    test_start = temporal.test_blocks[0][1].start
    test_end = min(temporal.test_blocks[-1][1].end, max_day + 1)

    states: dict[str, StrategyState] = {}
    for spec in strategies:
        states[spec["name"]] = StrategyState(
            name=spec["name"], kind=spec["kind"], window_days=spec.get("window_days"),
            ledger=CapacityLedger(policy.daily_capacity),
        )

    drift_log = DriftLog()
    monitor = CovariateMonitor(
        reference=reference_frame,
        columns=list(monitor_panel),
        psi_alert=adaptation_config["detectors"]["ks_psi"]["psi_alert"],
        ks_alert=adaptation_config["detectors"]["ks_psi"]["ks_alert"],
        bh_q=adaptation_config["detectors"]["ks_psi"]["bh_q"],
        panel_fraction_alert=adaptation_config["detectors"]["ks_psi"]["panel_fraction_alert"],
        consecutive_closes=adaptation_config["detectors"]["ks_psi"]["consecutive_closes"],
    )
    adwin_config = adaptation_config["detectors"]["adwin"]
    s1_config = adaptation_config["detectors"]["domain_classifier"]
    ks_window = adaptation_config["detectors"]["ks_psi"]["window_days"]

    # Indice por dia, para no filtrar el frame completo en cada dia y estrategia.
    index_by_day: dict[int, np.ndarray] = {
        int(day): np.where(days == day)[0] for day in np.unique(days)
    }

    pending_maturity: dict[int, list[dict[str, Any]]] = {}
    all_outcomes: list[pd.DataFrame] = []
    shared_package: ModelPackage | None = None

    for current_day in range(test_start, test_end):
        # ---------------- actualizacion programada ----------------
        if current_day in temporal.update_times:
            shared_package = None
            for spec in strategies:
                name = spec["name"]
                state = states[name]
                if spec["kind"] == "static" and state.active is not None:
                    continue  # las estrategias estaticas se ajustan una sola vez

                roles = build_version_roles(
                    name, current_day, temporal,
                    window_days=spec.get("window_days"), kind=spec["kind"],
                    lag_days=spec.get("lag_days", 0), fit_days=spec.get("fit_days"),
                )

                # En el primer corte, las estrategias de shared_initial tienen el
                # mismo predictor, datos, configuracion y semilla; se ajusta una vez
                # y se cuenta una vez en el costo de computo.
                reuse = (
                    shared_package is not None
                    and name in shared_initial
                    and current_day == temporal.update_times[0]
                    and shared_package.roles.predictor == roles.predictor
                )
                if reuse:
                    package = ModelPackage(
                        version_id=roles.version_id, strategy=name, update_time=current_day,
                        roles=roles, model=shared_package.model, calibrator=shared_package.calibrator,
                        policy=policy, costs=cost_model, valid=shared_package.valid,
                        invalid_reason=shared_package.invalid_reason,
                        support=shared_package.support,
                        diagnostics={**shared_package.diagnostics, "paquete_compartido_con": shared_package.version_id},
                        ids_by_role=shared_package.ids_by_role,
                    )
                else:
                    package = build_package(
                        name, roles, frame, family=family, model_config=model_config,
                        model_base=model_base, numeric_columns=numeric_columns,
                        categorical_columns=categorical_columns, policy=policy,
                        costs=cost_model,
                        temporal_config=temporal, calibration_params=calibration_params,
                        seed=seed, n_threads=n_threads, max_rows=max_rows, encoding=encoding,
                        target=target, id_column=id_column,
                    )
                    if name in shared_initial and current_day == temporal.update_times[0]:
                        shared_package = package

                # Gate sobre H, con el champion actual como comparacion.
                holdout_mask = (
                    roles.promotion_validation.mask(days)
                    & ((days + label_delay) < roles.job_time)
                )
                gate = promotion_gate(
                    package, state.active, frame.loc[holdout_mask], labels[holdout_mask],
                    cost_model=cost_model, gates=promotion_gates, target=target,
                    id_column=id_column, seed=seed,
                )
                gate["dia"] = current_day
                gate["estrategia"] = name
                # En el backtest, una recomendacion favorable se activa con la
                # autorizacion previa de la corrida y queda rotulada como simulada.
                gate["aprobacion"] = "simulada_preautorizada"
                state.gate_records.append(gate)

                if package.valid and gate.get("recomendacion") in ("promover", "activar"):
                    state.previous = state.active
                    state.active = package
                    if models_dir is not None:
                        # Solo se guardan los paquetes que pasaron el gate.
                        package.save(models_dir)
                    state.version_changes.append({
                        "dia": current_day, "estrategia": name,
                        "version_nueva": package.version_id,
                        "version_anterior": state.previous.version_id if state.previous else None,
                        "motivo": gate.get("motivo"),
                        "tipo": "activacion_inicial" if state.previous is None else "promocion",
                        "aprobacion": "simulada_preautorizada",
                    })
                    LOGGER.info("Dia %d: %s activa %s", current_day, name, package.version_id)
                else:
                    state.version_changes.append({
                        "dia": current_day, "estrategia": name,
                        "version_nueva": None,
                        "version_anterior": state.active.version_id if state.active else None,
                        "motivo": gate.get("motivo"),
                        "tipo": "conservar_champion" if state.active else "sin_version_valida",
                        "aprobacion": "simulada_preautorizada",
                    })

                # S1: un domain classifier por bloque.
                if name == strategies[0]["name"]:
                    recent = frame.loc[
                        Interval(max(0, current_day - s1_config["recent_window_days"]), current_day).mask(days)
                    ]
                    s1 = domain_classifier(
                        frame.loc[Interval(*s1_config["reference_interval"]).mask(days)],
                        recent, list(monitor_panel),
                        cap_per_domain=s1_config["cap_per_domain"],
                        inner_split=s1_config["inner_split"],
                        auc_alert=s1_config["auc_alert"], seed=seed,
                    )
                    drift_log.add("S1", event_day=current_day, available_day=current_day, **s1)

        # ---------------- scoring y decision del dia ----------------
        positions = index_by_day.get(current_day)
        if positions is None or positions.size == 0:
            continue
        day_frame = frame.iloc[positions]
        day_labels = labels[positions]
        day_ids = day_frame[id_column].tolist()
        amounts = day_frame["TransactionAmt"].to_numpy()

        for name, state in states.items():
            if state.active is None or not state.active.valid:
                # Sin version valida la estrategia se pausa y el dia queda registrado.
                state.unavailable_days.append(current_day)
                continue

            raw, calibrated = state.active.score(day_frame)
            decisions = decide_batch(
                calibrated, amounts, day_frame["dia"].to_numpy(), day_ids,
                state.active.policy, cost_model, ledger=state.ledger,
            )
            decisions["estrategia"] = name
            decisions["version_id"] = state.active.version_id
            decisions["p_cruda"] = raw
            decisions["semana"] = day_frame["semana"].to_numpy()
            decisions["available_at"] = current_day + label_delay
            state.predictions.append(decisions)

            pending_maturity.setdefault(current_day + label_delay, []).append({
                "estrategia": name,
                "version_id": state.active.version_id,
                "decisions": decisions,
                "labels": day_labels,
                "event_day": current_day,
            })

            # S2: desplazamiento de los scores de esta version.
            calibration_scores = state.active.diagnostics.get("calibracion", {}).get(
                "curva_calibrada", {}
            ).get("pred", [])
            if calibration_scores:
                s2 = score_drift(np.array(calibration_scores, dtype=float), calibrated)
                drift_log.add("S2", event_day=current_day, available_day=current_day,
                              estrategia=name, version_id=state.active.version_id, **s2)

        # ---------------- monitor diario de covariables ----------------
        recent_mask = Interval(max(0, current_day - ks_window), current_day).mask(days)
        if recent_mask.sum() > 0:
            result = monitor.evaluate(frame.loc[recent_mask], day=current_day)
            drift_log.add("S_KS_PSI", event_day=current_day, available_day=current_day, **result)

        # ---------------- madurez de etiquetas ----------------
        # El desenlace se conoce L dias despues del evento.
        for item in pending_maturity.pop(current_day, []):
            state = states[item["estrategia"]]
            outcomes = simulate_outcomes(item["decisions"], item["labels"], cost_model, seed=seed)
            outcomes["dia_disponibilidad"] = current_day
            outcomes["dia_evento"] = item["event_day"]
            outcomes.attrs["daily_capacity"] = policy.daily_capacity
            all_outcomes.append(outcomes)

            detector = state.detector_for(
                item["version_id"], delta=adwin_config["delta"], clock=adwin_config["clock"]
            )
            for event_id, probability, label in zip(
                outcomes["event_id"], outcomes["p"], outcomes["y"]
            ):
                detector.update(
                    event_id, probability, label,
                    event_day=item["event_day"], available_day=current_day,
                )

        if on_day is not None:
            on_day(current_day, states)

    # ---------------- cierre de madurez ----------------
    # Se avanza solo el reloj de disponibilidad hasta madurar la ultima cohorte,
    # sin nuevas transacciones ni reentrenamientos.
    for available_day in sorted(pending_maturity):
        for item in pending_maturity[available_day]:
            state = states[item["estrategia"]]
            outcomes = simulate_outcomes(item["decisions"], item["labels"], cost_model, seed=seed)
            outcomes["dia_disponibilidad"] = available_day
            outcomes["dia_evento"] = item["event_day"]
            outcomes.attrs["daily_capacity"] = policy.daily_capacity
            all_outcomes.append(outcomes)
            detector = state.detector_for(
                item["version_id"], delta=adwin_config["delta"], clock=adwin_config["clock"]
            )
            for event_id, probability, label in zip(outcomes["event_id"], outcomes["p"], outcomes["y"]):
                detector.update(event_id, probability, label,
                                event_day=item["event_day"], available_day=available_day)

    # ---------------- consolidacion ----------------
    predictions = pd.concat(
        [p for state in states.values() for p in state.predictions], ignore_index=True
    ) if any(state.predictions for state in states.values()) else pd.DataFrame()
    outcomes_frame = pd.concat(all_outcomes, ignore_index=True) if all_outcomes else pd.DataFrame()

    detections = []
    for state in states.values():
        for detector in state.detectors.values():
            for record in detector.detections:
                detections.append({"estrategia": state.name, **record})

    version_changes = pd.DataFrame(
        [c for state in states.values() for c in state.version_changes]
    )
    gates = pd.DataFrame([
        {k: v for k, v in g.items() if not isinstance(v, dict) or k in ("gates",)}
        for state in states.values() for g in state.gate_records
    ])

    coverage = {
        name: {
            "dias_evaluados": int(len(state.predictions)),
            "dias_sin_version": state.unavailable_days,
            "n_dias_sin_version": len(state.unavailable_days),
            "versiones_activadas": int(sum(1 for c in state.version_changes if c["version_nueva"])),
            "n_detectores": len(state.detectors),
            "n_detecciones_adwin": sum(d.n_detections for d in state.detectors.values()),
        }
        for name, state in states.items()
    }
    capacity = {name: state.ledger.stats() for name, state in states.items()}

    return BacktestResult(
        predictions=predictions,
        outcomes=outcomes_frame,
        drift_log=drift_log.to_frame(),
        version_changes=version_changes,
        gates=gates,
        capacity=capacity,
        coverage=coverage,
        adwin_detections=pd.DataFrame(detections),
    )
