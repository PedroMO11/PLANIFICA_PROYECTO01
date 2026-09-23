"""Construccion de paquetes versionados y gates de promocion.

Un paquete es la unidad desplegable: preprocesamiento, predictor, calibrador,
politica, esquema de features y un manifest con los datos usados en cada pieza. El
servicio lo carga sin reconstruir nada.

Orden de construccion:

1. ``predictor``  [c-W, c-14)   ajusta preprocesamiento y modelo
2. ``calibrador`` [c-14, c-7)   Platt sobre los scores del predictor ya ajustado
3. ``validacion`` [c-7, c)      gate de promocion

Cada paso usa scores del anterior sobre datos que ese paso no vio, de modo que el
calibrador corrige el mismo modelo que se despliega. Los umbrales se eligen una vez
en desarrollo, por eso no hay una reserva de politica por actualizacion.

Si un rol no alcanza el soporte minimo, la version se marca no valida; no se amplia
la ventana ni se usan etiquetas inmaduras para completar el soporte.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from .calibration import PlattCalibrator, calibration_report
from .decision import CostModel, Policy
from .metrics import average_precision, block_bootstrap_ci
from .models import FittedModel, fit_model
from .splits import VersionRoles, check_support, mature_training_mask
from .tracking import sha256_obj, utc_now, write_json

LOGGER = logging.getLogger("fraud_adaptive.adaptation")


@dataclass
class ModelPackage:
    """Paquete completo y autocontenido de una version."""

    version_id: str
    strategy: str
    update_time: int
    roles: VersionRoles
    model: FittedModel
    calibrator: PlattCalibrator
    policy: Policy
    # El modelo de costos viaja con el paquete porque c_FP se calibra en desarrollo;
    # asi el servicio decide con la misma economia que se valido.
    costs: CostModel = field(default_factory=CostModel)
    valid: bool = True
    invalid_reason: str | None = None
    support: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    ids_by_role: dict[str, set[Any]] = field(default_factory=dict, repr=False)
    created_at: str = field(default_factory=utc_now)

    # -- inferencia

    def score(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve ``(p_cruda, p_calibrada)``."""
        raw = self.model.predict_proba(frame)
        return raw, self.calibrator.transform(raw)

    # -- persistencia

    def manifest(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "strategy": self.strategy,
            "update_time": self.update_time,
            "created_at": self.created_at,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "roles": self.roles.to_dict(),
            "model": self.model.describe(),
            "calibrator": self.calibrator.to_dict(),
            "policy": self.policy.to_dict(),
            "costos": self.costs.to_dict(),
            "support": self.support,
            "diagnostics": self.diagnostics,
            "n_ids_por_rol": {role: len(ids) for role, ids in self.ids_by_role.items()},
            "contrato": {
                "predictor_no_ve": "TransactionID, tiempo absoluto, isFraud",
                "calibrador_no_ve": "filas del predictor",
                "H_no_ve": "filas de predictor, calibrador ni politica",
            },
        }

    def save(self, directory: str | Path) -> Path:
        """Escribe el paquete completo. El manifest incluye el hash de las piezas."""
        directory = Path(directory) / self.version_id
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model.estimator, directory / "predictor.joblib")
        joblib.dump(self.model.pipeline, directory / "preprocessor.joblib")
        joblib.dump(self.calibrator, directory / "calibrator.joblib")
        write_json(directory / "feature_schema.json", self.model.pipeline.schema())
        write_json(directory / "policy.json",
                   {**self.policy.to_dict(), "costos": self.costs.to_dict()})
        manifest = self.manifest()
        manifest["package_hash"] = sha256_obj({
            "schema": self.model.pipeline.schema(),
            "calibrator": self.calibrator.to_dict(),
            "policy": self.policy.to_dict(),
            "costos": self.costs.to_dict(),
            "roles": self.roles.to_dict(),
        })
        write_json(directory / "manifest.json", manifest)
        return directory

    @staticmethod
    def load_for_serving(directory: str | Path) -> dict[str, Any]:
        """Carga minima para el servicio: sin datos de entrenamiento."""
        directory = Path(directory)
        import json

        with open(directory / "manifest.json", "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        return {
            "estimator": joblib.load(directory / "predictor.joblib"),
            "pipeline": joblib.load(directory / "preprocessor.joblib"),
            "calibrator": joblib.load(directory / "calibrator.joblib"),
            "policy": manifest["policy"],
            "costos": manifest["costos"],
            "manifest": manifest,
        }


# --------------------------------------------------------------------------- construccion

def build_package(
    strategy: str,
    roles: VersionRoles,
    frame: pd.DataFrame,
    *,
    family: str,
    model_config: dict[str, Any],
    model_base: dict[str, Any],
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    policy: Policy,
    costs: CostModel,
    temporal_config: Any,
    calibration_params: dict[str, Any],
    seed: int = 42,
    n_threads: int = 4,
    max_rows: int | None = None,
    encoding: dict[str, Any] | None = None,
    target: str = "isFraud",
    id_column: str = "TransactionID",
) -> ModelPackage:
    """Construye una version completa respetando el orden y la madurez de etiquetas."""
    job_time = roles.job_time
    label_delay = temporal_config.label_delay_days
    days = frame["dia"].to_numpy()
    labels = frame[target].to_numpy()

    def role_slice(interval: Any) -> tuple[pd.DataFrame, np.ndarray, set[Any]]:
        # Filas dentro del intervalo y con etiqueta madura en job_time (ver
        # VersionRoles.job_time). Como los roles terminan en el corte, la condicion
        # de madurez protege ante cambios futuros de la grilla.
        mask = mature_training_mask(days, interval, job_time, label_delay)
        subset = frame.loc[mask]
        return subset, labels[mask], set(subset[id_column].tolist())

    predictor_frame, predictor_labels, predictor_ids = role_slice(roles.predictor)
    calibration_frame, calibration_labels, calibration_ids = role_slice(roles.calibration)
    holdout_frame, holdout_labels, holdout_ids = role_slice(roles.promotion_validation)

    support: dict[str, Any] = {}
    ok_fit, counts_fit = check_support(predictor_labels, temporal_config.min_support["fit"])
    ok_cal, counts_cal = check_support(calibration_labels, temporal_config.min_support["calibration"])
    ok_h, counts_h = check_support(holdout_labels, temporal_config.min_support["promotion_validation"])
    support.update({
        "predictor": {**counts_fit, "suficiente": ok_fit, "dias": roles.predictor.days},
        "calibracion": {**counts_cal, "suficiente": ok_cal},
        "validacion_promocion": {**counts_h, "suficiente": ok_h},
    })

    ids_by_role = {
        "predictor": predictor_ids,
        "calibracion": calibration_ids,
        "validacion_promocion": holdout_ids,
    }

    if not (ok_fit and ok_cal):
        reason = "soporte_insuficiente: fit=%s calibracion=%s" % (counts_fit, counts_cal)
        LOGGER.warning("Version %s no valida: %s", roles.version_id, reason)
        # Un paquete invalido permite al backtest conservar la version anterior.
        return ModelPackage(
            version_id=roles.version_id, strategy=strategy, update_time=roles.update_time,
            roles=roles, model=None, calibrator=PlattCalibrator(), policy=policy,
            costs=costs,
            valid=False, invalid_reason=reason, support=support, ids_by_role=ids_by_role,
        )

    model = fit_model(
        family, model_config, model_base, predictor_frame, predictor_labels,
        numeric_columns, categorical_columns,
        seed=seed, n_threads=n_threads, max_rows=max_rows, id_column=id_column, encoding=encoding,
    )

    calibrator = PlattCalibrator(clip=tuple(calibration_params.get("clip", (1e-6, 1 - 1e-6))))
    if len(calibration_frame):
        raw_calibration = model.predict_proba(calibration_frame)
        calibrator.fit(
            raw_calibration, calibration_labels, ids=calibration_ids,
            C=calibration_params.get("C", 1.0),
            solver=calibration_params.get("solver", "lbfgs"),
            max_iter=calibration_params.get("max_iter", 1000),
        )
        diagnostics = {
            "calibracion": calibration_report(
                raw_calibration, calibrator.transform(raw_calibration), calibration_labels
            )
        }
    else:
        diagnostics = {"calibracion": {"motivo_na": "cola_de_calibracion_vacia"}}

    # Diagnostico sobre H, que es validacion de promocion y no test final.
    if len(holdout_frame) and ok_h:
        raw_h, calibrated_h = ModelPackage(
            version_id=roles.version_id, strategy=strategy, update_time=roles.update_time,
            roles=roles, model=model, calibrator=calibrator, policy=policy,
            costs=costs,
        ).score(holdout_frame)
        diagnostics["validacion_promocion"] = {
            "n": int(len(holdout_frame)),
            "ap": average_precision(holdout_labels, calibrated_h),
            "prevalencia": float(holdout_labels.mean()),
            "brier": float(np.mean((calibrated_h - holdout_labels) ** 2)),
            "nota": "Validacion de seleccion/promocion. No es metrica final independiente.",
        }

    package = ModelPackage(
        version_id=roles.version_id, strategy=strategy, update_time=roles.update_time,
        roles=roles, model=model, calibrator=calibrator, policy=policy,
        costs=costs,
        valid=True, support=support, diagnostics=diagnostics, ids_by_role=ids_by_role,
    )
    assert_package_roles_disjoint(package)
    return package


def assert_package_roles_disjoint(package: ModelPackage) -> None:
    """Verifica sobre los TransactionID usados que ningun rol comparte filas.

    Complementa la disjuncion de intervalos ante posibles errores de filtrado.
    """
    roles = list(package.ids_by_role)
    for i, role_a in enumerate(roles):
        for role_b in roles[i + 1:]:
            shared = package.ids_by_role[role_a] & package.ids_by_role[role_b]
            if shared:
                raise ValueError(
                    "Fuga en %s: %d IDs compartidos entre %s y %s"
                    % (package.version_id, len(shared), role_a, role_b)
                )


# --------------------------------------------------------------------------- promocion

def promotion_gate(
    challenger: ModelPackage,
    champion: ModelPackage | None,
    holdout_frame: pd.DataFrame,
    holdout_labels: np.ndarray,
    *,
    cost_model: CostModel,
    gates: dict[str, Any],
    target: str = "isFraud",
    id_column: str = "TransactionID",
    seed: int = 42,
) -> dict[str, Any]:
    """Evalua la version candidata sobre H y recomienda si promoverla.

    La regla es de no inferioridad con tolerancia al ruido (costo <= 1.01 veces el
    de la version vigente). La funcion devuelve una recomendacion; la promocion
    requiere aprobacion humana registrada.
    """
    from .decision import CapacityLedger, decide_batch, simulate_outcomes, total_cost

    result: dict[str, Any] = {
        "version_id": challenger.version_id,
        "champion_id": champion.version_id if champion else None,
        "evaluado_en": "H = [c-7, c)",
        "aprobacion_humana_requerida": True,
        "promovido": False,
    }

    if not challenger.valid:
        result.update({"recomendacion": "rechazar", "motivo": "challenger_no_valido"})
        return result

    if len(holdout_frame) == 0:
        result.update({"recomendacion": "rechazar", "motivo": "holdout_vacio"})
        return result

    # H no puede haber participado en ningun ajuste de las dos versiones.
    holdout_ids = set(holdout_frame[id_column].tolist())
    for name, package in (("challenger", challenger), ("champion", champion)):
        if package is None:
            continue
        for role in ("predictor", "calibracion"):
            shared = holdout_ids & package.ids_by_role.get(role, set())
            if shared:
                result.update({
                    "recomendacion": "rechazar",
                    "motivo": "H_contaminado_con_%s_de_%s (%d IDs)" % (role, name, len(shared)),
                })
                return result

    def evaluate(package: ModelPackage) -> dict[str, Any]:
        raw, calibrated = package.score(holdout_frame)
        decisions = decide_batch(
            calibrated, holdout_frame["TransactionAmt"].to_numpy(),
            holdout_frame["dia"].to_numpy(), holdout_frame[id_column].tolist(),
            package.policy, cost_model, ledger=CapacityLedger(package.policy.daily_capacity),
        )
        outcomes = simulate_outcomes(decisions, holdout_labels, cost_model, seed=seed)
        summary = total_cost(outcomes)
        return {
            "costo_por_tx": summary["costo_por_tx"],
            "ap": average_precision(holdout_labels, calibrated),
            "brier": float(np.mean((calibrated - holdout_labels) ** 2)),
            "n_revisiones": summary["n_revisiones"],
            "tasa_bloqueo_legitimas": float(
                outcomes.loc[outcomes["y"] == 0, "legitima_bloqueada"].mean()
            ) if (holdout_labels == 0).any() else float("nan"),
            "_outcomes": outcomes,
        }

    challenger_stats = evaluate(challenger)
    result["challenger"] = {k: v for k, v in challenger_stats.items() if not k.startswith("_")}

    # Gate de arranque: costo inferior al de aprobar todo.
    approve_all_cost = float(holdout_frame.loc[holdout_labels == 1, "TransactionAmt"].sum() / len(holdout_frame))
    result["costo_aprobar_todo"] = approve_all_cost
    if not (challenger_stats["costo_por_tx"] <= approve_all_cost):
        result.update({
            "recomendacion": "rechazar",
            "motivo": "costo_no_mejora_aprobar_todo (%.4f > %.4f)"
                      % (challenger_stats["costo_por_tx"], approve_all_cost),
        })
        return result

    if champion is None or not champion.valid:
        result.update({
            "recomendacion": "activar",
            "motivo": "sin_champion_previo_valido; supera aprobar_todo",
            "gate_bootstrap": True,
        })
        return result

    champion_stats = evaluate(champion)
    result["champion"] = {k: v for k, v in champion_stats.items() if not k.startswith("_")}

    # Intervalo pareado de la diferencia de costo sobre los mismos eventos.
    diff = block_bootstrap_ci(
        challenger_stats["_outcomes"]["costo_observado"].to_numpy()
        - champion_stats["_outcomes"]["costo_observado"].to_numpy(),
        holdout_frame["dia"].to_numpy(), block_days=7, n_resamples=200, seed=seed,
    )
    result["delta_costo_ic"] = diff

    checks: dict[str, bool] = {}
    ratio_max = gates.get("cost_ratio_max", 1.01)
    champion_cost = champion_stats["costo_por_tx"]
    checks["costo_no_inferioridad"] = bool(
        challenger_stats["costo_por_tx"] <= ratio_max * champion_cost
    )
    checks["ap_no_cae"] = bool(
        (challenger_stats["ap"] - champion_stats["ap"]) >= -gates.get("ap_drop_max", 0.01)
    )
    checks["brier_no_sube"] = bool(
        (challenger_stats["brier"] - champion_stats["brier"]) <= gates.get("brier_increase_max", 0.005)
    )
    gap_pp = (challenger_stats["tasa_bloqueo_legitimas"] - champion_stats["tasa_bloqueo_legitimas"]) * 100.0
    checks["bloqueo_legitimas_no_crece"] = bool(
        not np.isfinite(gap_pp) or gap_pp <= gates.get("segment_block_increase_pp_max", 2.0)
    )

    result["gates"] = checks
    result["brecha_bloqueo_pp"] = float(gap_pp) if np.isfinite(gap_pp) else None
    passed = all(checks.values())
    result["recomendacion"] = "promover" if passed else "conservar_champion"
    result["motivo"] = "todos los gates tecnicos pasan" if passed else \
        "fallan: " + ", ".join(k for k, v in checks.items() if not v)
    return result


def authorization_manifest(
    run_id: str,
    tasks: Sequence[dict[str, Any]],
    *,
    authorized_by: str,
    simulated: bool = True,
) -> dict[str, Any]:
    """Manifest de autorizacion humana previa de la corrida offline.

    Enumera las tareas permitidas antes de lanzarlas. Las promociones del backtest
    quedan rotuladas como simuladas y ningun script despliega recursos.
    """
    return {
        "run_id": run_id,
        "autorizado_por": authorized_by,
        "autorizado_en": utc_now(),
        "alcance": "corrida offline local, reanudable",
        "promociones": "simuladas" if simulated else "reales",
        "tareas_autorizadas": list(tasks),
        "no_autoriza": [
            "despliegue remoto o creacion de recursos cloud",
            "promocion real de un modelo a produccion",
            "pagos, bloqueos o decisiones sobre clientes reales",
        ],
        "nota": "La aprobacion humana por acto sigue siendo obligatoria en el diseno operativo.",
    }
