"""Orquestacion de las fases F2-F4: datos, tuning, modelos estaticos y adaptacion.

Cada fase deja artefactos verificados por hash y un checkpoint reanudable. Una
fase posterior no recalcula lo que ya existe salvo que se pida ``--rebuild``: una
cache invalida provoca error o reconstruccion explicita, nunca reutilizacion
silenciosa.

El orden de las fases codifica el prerregistro del plan. La familia, los
hiperparametros y los umbrales se congelan en ``fit_static`` (que solo ve
desarrollo) y el hash de esa seleccion se sella ANTES de abrir el periodo de test
en ``run_adaptation``. Si alguien cambiara la configuracion entre ambas, el hash
no coincidiria y la corrida falla.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from . import data as data_module
from . import features as features_module
from .adaptation import authorization_manifest
from .backtest import run_backtest
from .calibration import PlattCalibrator, calibration_report
from .decision import (
    CapacityLedger, CostModel, Policy, baseline_policies, calibrate_fp_cost,
    calibrate_review_price, cost_model_from_dict, decide_batch, implied_thresholds,
    policy_from_dict, select_thresholds, simulate_outcomes, total_cost,
)
from .metrics import (
    average_precision, threshold_for_fpr, threshold_for_precision,
)
from .models import fit_model, iter_family_configs
from .splits import Interval, TemporalConfig, build_splits_manifest, mature_training_mask
from .tracking import RunContext, sha256_obj, write_json

LOGGER = logging.getLogger("fraud_adaptive.pipeline")


def load_configs(config_dir: str | Path = "configs") -> dict[str, Any]:
    config_dir = Path(config_dir)
    out: dict[str, Any] = {}
    for name in ("base", "temporal", "decision", "models", "adaptation", "serving", "gcp"):
        path = config_dir / (name + ".yaml")
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                out[name] = yaml.safe_load(handle)
    return out


# --------------------------------------------------------------------------- F2

def prepare_data(
    configs: dict[str, Any],
    run: RunContext,
    *,
    rebuild: bool = False,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Carga, valida, integra y genera features causales. Materializa Parquet."""
    base = configs["base"]
    paths = base["paths"]
    data_config = base["data"]
    processed = Path(paths["processed"])
    processed.mkdir(parents=True, exist_ok=True)
    parquet_path = processed / "eventos.parquet"

    if parquet_path.exists() and not rebuild:
        LOGGER.info("Reutilizando Parquet verificado: %s", parquet_path)
        return pd.read_parquet(parquet_path)

    sources = data_module.resolve_sources(
        paths["data_root"], data_config["transactions_file"], data_config["identity_file"]
    )
    data_module.assert_no_forbidden_files(paths["data_root"], data_config["forbidden_files"])

    with run.task("cargar_fuentes", kind="datos"):
        transactions, identity = data_module.load_raw(sources, nrows=nrows)

    with run.task("validar_esquema", kind="datos"):
        schema_report = data_module.validate_schema(
            transactions, identity,
            join_key=data_config["join_key"], target=data_config["target"],
            time_col=data_config["time_col"], amount_col=data_config["amount_col"],
        )

    with run.task("join_fuentes", kind="datos"):
        merged, join_audit = data_module.join_sources(
            transactions, identity, join_key=data_config["join_key"]
        )
        del transactions, identity

    with run.task("derivar_tiempo", kind="datos"):
        merged = data_module.derive_time_columns(merged, time_col=data_config["time_col"])
        merged = data_module.order_events(
            merged, time_col=data_config["time_col"], join_key=data_config["join_key"]
        )

    with run.task("features_causales", kind="datos"):
        merged, feature_meta = features_module.prepare_features(
            merged, time_col=data_config["time_col"], amount_col=data_config["amount_col"]
        )

    # Los objetos de proxy son texto largo; se conservan como categoricas para
    # que el Parquet no crezca de forma innecesaria.
    for column in ("card_proxy", "cliente_proxy", "device_proxy"):
        if column in merged.columns:
            merged[column] = merged[column].astype("category")

    merged.to_parquet(parquet_path, index=False)
    run.register_artifact("eventos_parquet", parquet_path)

    manifest_dir = Path(paths["manifests"])
    sources_manifest = data_module.build_sources_manifest(
        sources,
        extra={
            "esquema": schema_report,
            "join": join_audit,
            "features": feature_meta,
            "data_source": _detect_data_source(paths["data_root"]),
        },
    )
    write_json(manifest_dir / "sources.json", sources_manifest)
    run.register_artifact("sources_manifest", manifest_dir / "sources.json")

    temporal = TemporalConfig.from_yaml("configs/temporal.yaml")
    splits_manifest = build_splits_manifest(
        temporal, configs["adaptation"]["strategies"], max_day=int(merged["dia"].max())
    )
    write_json(manifest_dir / "splits.json", splits_manifest)
    run.register_artifact("splits_manifest", manifest_dir / "splits.json")

    run.log("datos_preparados", filas=len(merged), columnas=merged.shape[1],
            dias=int(merged["dia"].max()) + 1)
    return merged


def _detect_data_source(data_root: str | Path) -> str:
    """Distingue datos reales de sustitutos, para rotular cada artefacto."""
    if (Path(data_root) / "LEEME_DATOS_SUSTITUTOS.txt").exists():
        return "sintetico_sustituto"
    return "ieee_cis_real"


# --------------------------------------------------------------------------- F3

def run_tuning(
    frame: pd.DataFrame,
    configs: dict[str, Any],
    run: RunContext,
    temporal: TemporalConfig,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    """18 fits: 3 familias x 3 configuraciones x 2 folds forward.

    La seleccion es por AP media de validacion. Los folds son internos: sus
    metricas nunca se reportan como resultado final.
    """
    base = configs["base"]
    models_config = configs["models"]
    label_delay = temporal.label_delay_days
    days = frame["dia"].to_numpy()
    labels = frame[configs["base"]["data"]["target"]].to_numpy()

    rows: list[dict[str, Any]] = []
    for fold in temporal.tuning_folds:
        fit_interval = Interval(*fold["fit"])
        val_interval = Interval(*fold["val"])
        # Los folds se reconstruyen como desarrollo en T=120, cuando sus etiquetas
        # ya maduraron. No simulan una autorizacion instantanea en el dia 30.
        # La madurez se mide contra el tiempo del job (T), no contra T - L.
        job_time = temporal.update_times[0]
        fit_mask = mature_training_mask(days, fit_interval, job_time, label_delay)
        val_mask = mature_training_mask(days, val_interval, job_time, label_delay)

        fit_frame, fit_labels = frame.loc[fit_mask], labels[fit_mask]
        val_frame, val_labels = frame.loc[val_mask], labels[val_mask]

        for family, config, model_base in iter_family_configs(models_config):
            task_name = "tuning_%s_%s_%s" % (family, config["name"], fold["name"])
            if run.has_checkpoint(task_name):
                rows.append(run.load_checkpoint(task_name))
                continue

            max_rows = models_config["resource_caps"]["rf_max_rows"] if family == "random_forest" else None
            with run.task(task_name, kind="fit_tuning",
                          limit_seconds=base["budget"]["per_fit_limits_seconds"]["tuning"]) as record:
                model = fit_model(
                    family, config, model_base, fit_frame, fit_labels,
                    numeric_columns, categorical_columns,
                    seed=base["seed"], n_threads=base["n_threads"], max_rows=max_rows,
                    encoding=models_config["encoding"],
                )
                scores = model.predict_proba(val_frame)
                ap = average_precision(val_labels, scores)

            row = {
                "fold": fold["name"], "familia": family, "config": config["name"],
                "ap_validacion": ap,
                "n_fit": model.fit_rows, "n_val": int(val_mask.sum()),
                "prevalencia_fit": model.fit_positives / model.fit_rows if model.fit_rows else np.nan,
                "segundos": record.get("seconds"),
                "submuestreado": model.subsampled,
            }
            rows.append(row)
            run.save_checkpoint(task_name, row)
            LOGGER.info("%s -> AP %.4f (%.1fs)", task_name, ap, record.get("seconds", 0))

    table = pd.DataFrame(rows)
    summary = (
        table.groupby(["familia", "config"], as_index=False)
        .agg(ap_media=("ap_validacion", "mean"), segundos=("segundos", "sum"))
        .sort_values(["familia", "ap_media"], ascending=[True, False])
    )
    run.save_checkpoint("tuning_resumen", {"tabla": summary.to_dict("records")})
    return summary


def select_best_configs(summary: pd.DataFrame) -> dict[str, str]:
    """Mejor configuracion por familia: AP media, desempate por menor tiempo."""
    best: dict[str, str] = {}
    for family, group in summary.groupby("familia"):
        ordered = group.sort_values(["ap_media", "segundos"], ascending=[False, True])
        best[family] = str(ordered.iloc[0]["config"])
    return best


def fit_static_models(
    frame: pd.DataFrame,
    configs: dict[str, Any],
    run: RunContext,
    temporal: TemporalConfig,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    best_configs: dict[str, str],
) -> dict[str, Any]:
    """Ajusta los tres modelos finales, calibra, elige umbrales y congela la familia.

    Todo ocurre dentro de desarrollo. Al terminar se sella el hash de la seleccion:
    es el prerregistro que impide retunear despues de ver el test.
    """
    base = configs["base"]
    models_config = configs["models"]
    decision_config = configs["decision"]
    target = base["data"]["target"]

    days = frame["dia"].to_numpy()
    labels = frame[target].to_numpy()
    # Madurez contra el tiempo del job T, no contra cutoff = T - L: los tramos de
    # desarrollo terminan en T - L y por tanto ya maduraron cuando el job corre.
    job_time = temporal.update_times[0]

    fit_mask = mature_training_mask(days, temporal.base_fit, job_time, temporal.label_delay_days)
    cal_mask = mature_training_mask(days, temporal.calibration, job_time, temporal.label_delay_days)
    pol_mask = mature_training_mask(days, temporal.policy, job_time, temporal.label_delay_days)

    capacity = decision_config["capacity"]["daily_reviews"]
    analyst = decision_config["analyst"]
    c_review = decision_config["costs"]["c_review"]

    results: dict[str, Any] = {"familias": {}}
    # Primera pasada: ajustar y calibrar. Ningun costo se evalua todavia, porque
    # c_FP aun no esta calibrado y evaluarlo con un valor provisional produciria
    # una eleccion de familia que luego habria que rehacer.
    for family, spec in models_config["families"].items():
        config = next(c for c in spec["configs"] if c["name"] == best_configs[family])
        task_name = "estatico_%s" % family

        max_rows = models_config["resource_caps"]["rf_max_rows"] if family == "random_forest" else None
        with run.task(task_name, kind="fit_final",
                      limit_seconds=base["budget"]["per_fit_limits_seconds"]["final"]) as record:
            model = fit_model(
                family, config, spec.get("base", {}), frame.loc[fit_mask], labels[fit_mask],
                numeric_columns, categorical_columns,
                seed=base["seed"], n_threads=base["n_threads"], max_rows=max_rows,
                encoding=models_config["encoding"],
            )

            # Platt sobre la cola de calibracion, que el predictor no vio.
            raw_cal = model.predict_proba(frame.loc[cal_mask])
            calibrator = PlattCalibrator().fit(
                raw_cal, labels[cal_mask], **{
                    k: v for k, v in models_config["calibration"].items()
                    if k in ("C", "solver", "max_iter")
                }
            )
            report = calibration_report(raw_cal, calibrator.transform(raw_cal), labels[cal_mask])

            policy_frame = frame.loc[pol_mask]
            policy_labels = labels[pol_mask]
            calibrated_policy = calibrator.transform(model.predict_proba(policy_frame))

            # Umbrales diagnosticos, congelados junto con la politica.
            fpr_threshold = threshold_for_fpr(
                policy_labels, calibrated_policy, decision_config["diagnostic_targets"]["fpr_target"]
            )
            precision_threshold = threshold_for_precision(
                policy_labels, calibrated_policy,
                decision_config["diagnostic_targets"]["precision_target"],
            )

        results["familias"][family] = {
            "config": config["name"],
            "modelo": model,
            "calibrador": calibrator,
            "p_politica": calibrated_policy,
            "ap_politica": average_precision(policy_labels, calibrated_policy),
            "calibracion": report,
            "umbral_fpr": fpr_threshold,
            "umbral_precision": precision_threshold,
            "segundos": record.get("seconds"),
            "n_fit": model.fit_rows,
        }
        LOGGER.info("Estatico %s (%s): AP %.4f en la reserva de politica",
                    family, config["name"], results["familias"][family]["ap_politica"])

    policy_frame = frame.loc[pol_mask]
    policy_labels = labels[pol_mask]
    policy_amount = policy_frame["TransactionAmt"].to_numpy()
    policy_day = policy_frame["dia"].to_numpy()
    policy_ids = policy_frame["TransactionID"].tolist()

    # Calibracion de c_FP.
    #
    # Se usa la familia de mayor AP como referencia, no la de menor costo: el costo
    # depende de c_FP y elegir por costo antes de calibrarlo seria circular. El AP
    # no depende del modelo economico, de modo que la referencia queda fijada por un
    # criterio ajeno al parametro que se esta calibrando.
    reference = max(results["familias"].items(), key=lambda kv: kv[1]["ap_politica"])[0]
    reference_scores = results["familias"][reference]["p_politica"]
    calibration_spec = decision_config["costs"]["calibracion_c_fp"]

    # Los dos parametros se condicionan mutuamente. c_FP fija la escala de los
    # costos, que determina cuanto ahorra revisar y por tanto el precio sombra del
    # cupo. Y racionar el cupo cambia cuantos casos terminan bloqueados, que es lo
    # que c_FP controla. Se alternan hasta que ambos se estabilizan, lo que ocurre
    # en dos o tres pasadas.
    review_price = 0.0
    fp_calibration = None
    price_calibration = None
    with run.task("calibrar_politica", kind="politica"):
        for pasada in range(3):
            fp_calibration = calibrate_fp_cost(
                reference_scores, policy_amount, policy_day, policy_labels, policy_ids,
                target_block_rate=calibration_spec["objetivo_bloqueo_legitimo"],
                grid=calibration_spec["grid"], c_review=c_review,
                r_h=analyst["r_h"], f_h=analyst["f_h"],
                daily_capacity=capacity, review_price=review_price, seed=base["seed"],
            )
            candidate = CostModel(c_fp=fp_calibration["c_fp"], c_review=c_review,
                                  r_h=analyst["r_h"], f_h=analyst["f_h"])
            price_calibration = calibrate_review_price(
                reference_scores, policy_amount, policy_day, candidate,
                daily_capacity=capacity,
            )
            nuevo_precio = price_calibration["review_price"]
            LOGGER.info("Pasada %d: c_FP=%.1f, precio sombra=%.4f",
                        pasada + 1, fp_calibration["c_fp"], nuevo_precio)
            if abs(nuevo_precio - review_price) < 1e-3:
                review_price = nuevo_precio
                break
            review_price = nuevo_precio

    cost_model = CostModel(c_fp=fp_calibration["c_fp"], c_review=c_review,
                           r_h=analyst["r_h"], f_h=analyst["f_h"])
    results["calibracion_c_fp"] = {
        "familia_referencia": reference,
        "c_fp": fp_calibration["c_fp"],
        "objetivo": fp_calibration["objetivo"],
        "alcanzado": fp_calibration["alcanzado"],
        "tasa_bloqueo_legitimo": fp_calibration["tasa_bloqueo_legitimo"],
        "tabla": fp_calibration["tabla"],
    }
    results["calibracion_cupo"] = price_calibration
    results["costos"] = cost_model.to_dict()

    # Segunda pasada: costo de cada familia bajo la politica ya congelada, y la
    # regla de umbrales como referencia para medir lo que cuesta ignorar el monto.
    operating = Policy(tau_low=0.0, tau_high=1.0, daily_capacity=capacity,
                       rule=decision_config["rule"], review_price=review_price)
    results["umbrales_implicados"] = [
        implied_thresholds(cost_model, monto, operating)
        for monto in (25.0, float(np.median(policy_amount)), 250.0)
    ]
    for family, info in results["familias"].items():
        decisions = decide_batch(
            info["p_politica"], policy_amount, policy_day, policy_ids,
            operating, cost_model, ledger=CapacityLedger(capacity),
        )
        outcomes = simulate_outcomes(decisions, policy_labels, cost_model, seed=base["seed"])
        info["costo_politica"] = float(total_cost(outcomes)["costo_por_tx"])

        selection = select_thresholds(
            info["p_politica"], policy_amount, policy_day, policy_labels, policy_ids,
            cost_model, quantiles=decision_config["thresholds"]["grid_quantiles"],
            daily_capacity=capacity, delta=decision_config["thresholds"]["delta"],
            seed=base["seed"],
        )
        info["policy"] = operating
        info["policy_umbral"] = selection["policy"]
        info["costo_umbral"] = float(selection["best"]["costo_por_tx"])
        info["grid_umbrales"] = selection["grid"]
        LOGGER.info(
            "Estatico %s: costo %.4f UM/tx con argmin, %.4f con el mejor umbral fijo",
            family, info["costo_politica"], info["costo_umbral"],
        )

    # Familia adaptativa: menor costo en la reserva de politica.
    # Desempate a <=1% de costo por menor tiempo de fit.
    ordered = sorted(
        results["familias"].items(),
        key=lambda kv: (kv[1]["costo_politica"], kv[1]["segundos"]),
    )
    best_cost = ordered[0][1]["costo_politica"]
    viable = [(name, info) for name, info in ordered if info["costo_politica"] <= best_cost * 1.01]
    chosen = min(viable, key=lambda kv: kv[1]["segundos"])[0]
    results["familia_elegida"] = chosen

    # Baselines simulados, para la comparacion economica del informe.
    policy_frame = frame.loc[pol_mask]
    policy_labels = labels[pol_mask]
    baselines: dict[str, Any] = {}
    for name, baseline_policy in baseline_policies(capacity).items():
        neutral = np.full(len(policy_frame), 0.5)
        decisions = decide_batch(
            neutral, policy_frame["TransactionAmt"].to_numpy(), policy_frame["dia"].to_numpy(),
            policy_frame["TransactionID"].tolist(), baseline_policy, cost_model,
            ledger=CapacityLedger(capacity),
        )
        outcomes = simulate_outcomes(decisions, policy_labels, cost_model, seed=base["seed"])
        baselines[name] = total_cost(outcomes)
    results["baselines"] = baselines

    # Prerregistro: hash de todo lo que queda congelado antes del test.
    frozen = {
        "familia": chosen,
        "config": results["familias"][chosen]["config"],
        "policy": results["familias"][chosen]["policy"].to_dict(),
        # c_FP entra al sello porque ahora es un parametro derivado de los datos de
        # desarrollo, no una constante del config. Sin sellarlo se podria recalibrar
        # despues de ver el test, que es exactamente lo que el prerregistro impide.
        "costos": cost_model.to_dict(),
        "objetivo_bloqueo_legitimo": results["calibracion_c_fp"]["objetivo"],
        "umbral_fpr": results["familias"][chosen]["umbral_fpr"].get("umbral"),
        "umbral_precision": results["familias"][chosen]["umbral_precision"].get("umbral"),
        "seed": base["seed"],
        "label_delay": temporal.label_delay_days,
        "cadence": temporal.cadence_days,
    }
    results["prerregistro"] = {"contenido": frozen, "hash": sha256_obj(frozen)}
    run.save_checkpoint("prerregistro", results["prerregistro"])
    LOGGER.info("Familia congelada: %s | hash prerregistro %s",
                chosen, results["prerregistro"]["hash"][:16])
    return results


def select_deployable_window(
    frame: pd.DataFrame,
    configs: dict[str, Any],
    run: RunContext,
    temporal: TemporalConfig,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    static_results: dict[str, Any],
) -> dict[str, Any]:
    """Elige la ventana desplegable por costo, usando SOLO desarrollo.

    Es el paso que convierte el experimento en una recomendacion operativa. Sin
    el, la unica forma de recomendar una ventana seria mirar el test y quedarse
    con la que gano, que es precisamente el retuning retrospectivo que el
    protocolo prohibe (C17, C22).

    Protocolo:

    * se construyen los paquetes iniciales de W30, W60 y W90 en el primer corte;
    * cada uno se evalua sobre ``[c-7, c)``, posterior a su predictor y a su
      calibrador. En el primer corte no hay champion, de modo que esa cola no
      cumple todavia funcion de gate y puede usarse para elegir;
    * se aplica la politica YA CONGELADA, sin reoptimizar umbrales;
    * gana el menor costo. Desempate a <=1 %: la W mas pequena tecnicamente
      viable, porque a igualdad de costo la ventana corta reentrena mas barato.

    S0 y E15 no participan: son referencias de comparacion y el plan prohibe
    entregarlas como sistema.
    """
    from .adaptation import build_package
    from .decision import CapacityLedger, decide_batch, simulate_outcomes, total_cost
    from .splits import build_version_roles

    adaptation_config = configs["adaptation"]
    decision_config = configs["decision"]
    family = static_results["familia_elegida"]
    chosen = static_results["familias"][family]
    family_spec = configs["models"]["families"][family]
    config = next(c for c in family_spec["configs"] if c["name"] == chosen["config"])

    policy_data = chosen["policy"]
    policy = policy_from_dict(policy_data)
    cost_model = cost_model_from_dict(static_results["costos"])

    update_time = temporal.update_times[0]
    days = frame["dia"].to_numpy()
    labels = frame[configs["base"]["data"]["target"]].to_numpy()
    candidatos = [
        s for s in adaptation_config["strategies"]
        if s.get("role") == "desplegable" and s.get("kind") == "sliding"
    ]

    filas: list[dict[str, Any]] = []
    for spec in sorted(candidatos, key=lambda s: s["window_days"]):
        roles = build_version_roles(
            spec["name"], update_time, temporal,
            window_days=spec.get("window_days"), kind=spec["kind"],
            lag_days=spec.get("lag_days", 0), fit_days=spec.get("fit_days"),
        )
        with run.task("seleccion_W_%s" % spec["name"], kind="fit_seleccion_W"):
            package = build_package(
                spec["name"], roles, frame, family=family, model_config=config,
                model_base=family_spec.get("base", {}), numeric_columns=numeric_columns,
                categorical_columns=categorical_columns, policy=policy, costs=cost_model,
                temporal_config=temporal, calibration_params=configs["models"]["calibration"],
                seed=configs["base"]["seed"], n_threads=configs["base"]["n_threads"],
                max_rows=configs["models"]["resource_caps"]["rf_max_rows"] if family == "random_forest" else None,
                encoding=configs["models"]["encoding"], target=configs["base"]["data"]["target"],
            )

        if not package.valid:
            filas.append({
                "estrategia": spec["name"], "window_days": spec["window_days"],
                "valida": False, "motivo": package.invalid_reason,
                "costo_por_tx": float("inf"),
            })
            continue

        # Cola posterior al predictor y al calibrador de ESTE paquete.
        mask = (roles.promotion_validation.mask(days)
                & ((days + temporal.label_delay_days) < roles.job_time))
        subset = frame.loc[mask]
        subset_labels = labels[mask]
        _, calibrated = package.score(subset)
        decisiones = decide_batch(
            calibrated, subset["TransactionAmt"].to_numpy(), subset["dia"].to_numpy(),
            subset[configs["base"]["data"]["join_key"]].tolist() if "join_key" in configs["base"]["data"]
            else subset["TransactionID"].tolist(),
            policy, cost_model, ledger=CapacityLedger(policy.daily_capacity),
        )
        desenlaces = simulate_outcomes(decisiones, subset_labels, cost_model, seed=configs["base"]["seed"])
        resumen = total_cost(desenlaces)
        filas.append({
            "estrategia": spec["name"],
            "window_days": spec["window_days"],
            "dias_de_fit": roles.predictor.days,
            "valida": True,
            "n_eventos_evaluacion": int(len(subset)),
            "costo_por_tx": resumen["costo_por_tx"],
            "n_revisiones": resumen["n_revisiones"],
            "fraudes_en_fit": package.support["predictor"]["fraud"],
        })

    tabla = pd.DataFrame(filas).sort_values("costo_por_tx").reset_index(drop=True)
    validas = tabla[tabla["valida"]]
    if validas.empty:
        # C21: sin ventana valida no se activa nada. No se degrada a S0 ni a E15.
        LOGGER.error("Ninguna ventana desplegable es valida: el sistema quedaria en pausa")
        return {"elegida": None, "motivo": "ninguna_ventana_valida", "tabla": tabla}

    mejor = float(validas.iloc[0]["costo_por_tx"])
    empatadas = validas[validas["costo_por_tx"] <= mejor * 1.01]
    elegida = empatadas.sort_values("window_days").iloc[0]

    resultado = {
        "elegida": str(elegida["estrategia"]),
        "window_days": int(elegida["window_days"]),
        "costo_en_desarrollo": float(elegida["costo_por_tx"]),
        "tabla": tabla,
        "criterio": "menor costo en la reserva de politica; desempate a <=1 % por menor W",
        "evaluado_en": "cola [c-7, c) del primer corte, solo desarrollo",
        "nota": "Elegida SIN mirar el test. Que otra gane en el test es diagnostico retrospectivo.",
    }
    run.save_checkpoint("seleccion_W", {k: v for k, v in resultado.items() if k != "tabla"})
    LOGGER.info(
        "Ventana desplegable elegida en desarrollo: %s (%.4f UM/tx)",
        resultado["elegida"], resultado["costo_en_desarrollo"],
    )
    return resultado


# --------------------------------------------------------------------------- F4

def run_adaptation(
    frame: pd.DataFrame,
    configs: dict[str, Any],
    run: RunContext,
    temporal: TemporalConfig,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    static_results: dict[str, Any],
) -> Any:
    """Experimento central: cinco estrategias sobre los mismos eventos."""
    base = configs["base"]
    models_config = configs["models"]
    decision_config = configs["decision"]
    adaptation_config = configs["adaptation"]

    family = static_results["familia_elegida"]
    chosen = static_results["familias"][family]
    spec = models_config["families"][family]
    config = next(c for c in spec["configs"] if c["name"] == chosen["config"])

    # Verificacion de prerregistro: la config congelada debe seguir siendo la misma.
    expected = run.load_checkpoint("prerregistro")
    if expected and expected["hash"] != static_results["prerregistro"]["hash"]:
        raise RuntimeError(
            "El hash de prerregistro cambio entre la seleccion y el test. "
            "Eso indica un retuning despues de congelar; la corrida se detiene."
        )

    # El test usa la economia sellada en el prerregistro, no la del config: si
    # alguien recalibrara c_FP despues de congelar, esta lectura lo ignoraria.
    cost_model = cost_model_from_dict(static_results["prerregistro"]["contenido"]["costos"])

    # Manifest de autorizacion humana previa (C23).
    tasks = [
        {"tarea": "fit_%s_T%d" % (s["name"], t), "tipo": "ajuste_offline"}
        for s in adaptation_config["strategies"] for t in temporal.update_times
    ]
    authorization = authorization_manifest(run.run_id, tasks, authorized_by="equipo (revision previa)")
    write_json(run.dir / "authorization.json", authorization)
    run.register_artifact("authorization", run.dir / "authorization.json")

    reference = frame.loc[Interval(*adaptation_config["detectors"]["domain_classifier"]
                                   ["reference_interval"]).mask(frame["dia"].to_numpy())]
    monitor_panel = data_module.select_monitor_panel(
        reference, numeric_columns,
        max_columns=adaptation_config["detectors"]["ks_psi"]["panel_max_numeric"],
    )
    # Variables de negocio siempre presentes en el panel, aunque no ganen por varianza.
    for column in ("TransactionAmt", "monto_log", "has_identity"):
        if column in frame.columns and column not in monitor_panel:
            monitor_panel.append(column)
    run.save_checkpoint("monitor_panel", {"columnas": monitor_panel})

    with run.task("backtest_adaptacion", kind="backtest"):
        result = run_backtest(
            frame,
            strategies=adaptation_config["strategies"],
            temporal=temporal,
            family=family,
            model_config=config,
            model_base=spec.get("base", {}),
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            policy=chosen["policy"],
            cost_model=cost_model,
            calibration_params=models_config["calibration"],
            adaptation_config=adaptation_config,
            promotion_gates=decision_config["promotion_gates"],
            monitor_panel=monitor_panel,
            reference_frame=reference,
            seed=base["seed"],
            n_threads=base["n_threads"],
            max_rows=models_config["resource_caps"]["rf_max_rows"] if family == "random_forest" else None,
            encoding=models_config["encoding"],
            target=base["data"]["target"],
            shared_initial=adaptation_config.get("share_initial_package", []),
            models_dir=base["paths"]["models"],
        )

    run.log("backtest_terminado",
            n_predicciones=len(result.predictions), n_desenlaces=len(result.outcomes))
    return result


# --------------------------------------------------------------------------- F5

def export_package(
    frame: pd.DataFrame,
    configs: dict[str, Any],
    run: RunContext,
    temporal: TemporalConfig,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    static_results: dict[str, Any],
    *,
    strategy: str = "W30",
    update_time: int | None = None,
) -> dict[str, Any]:
    """Reconstruye y persiste un paquete desplegable concreto.

    Sirve para dos casos: entregar al equipo el paquete a desplegar sin repetir el
    backtest completo, y regenerar un artefacto perdido. Usa exactamente los mismos
    roles, familia, hiperparametros y politica congelados, asi que el paquete es el
    mismo que produjo las metricas.

    Solo exporta estrategias DESPLEGABLES: S0 y E15 son referencias de comparacion
    y el plan prohibe entregarlas como sistema.
    """
    from .adaptation import build_package
    from .splits import build_version_roles

    adaptation_config = configs["adaptation"]
    spec = next((s for s in adaptation_config["strategies"] if s["name"] == strategy), None)
    if spec is None:
        raise ValueError("Estrategia desconocida: %s" % strategy)
    if spec["role"] != "desplegable":
        raise ValueError(
            "%s es una referencia no desplegable (%s). Exporta W30, W60 o W90."
            % (strategy, spec["role"])
        )

    update_time = update_time if update_time is not None else temporal.update_times[-1]
    family = static_results["familia_elegida"]
    chosen = static_results["familias"][family]
    family_spec = configs["models"]["families"][family]
    config = next(c for c in family_spec["configs"] if c["name"] == chosen["config"])

    policy_data = chosen["policy"]
    policy = policy_from_dict(policy_data)
    costs = cost_model_from_dict(static_results["costos"])

    roles = build_version_roles(
        strategy, update_time, temporal, window_days=spec.get("window_days"), kind=spec["kind"]
    )

    with run.task("exportar_paquete_%s" % roles.version_id, kind="fit_adaptivo"):
        package = build_package(
            strategy, roles, frame, family=family, model_config=config,
            model_base=family_spec.get("base", {}), numeric_columns=numeric_columns,
            categorical_columns=categorical_columns, policy=policy, costs=costs,
            temporal_config=temporal,
            calibration_params=configs["models"]["calibration"],
            seed=configs["base"]["seed"], n_threads=configs["base"]["n_threads"],
            max_rows=configs["models"]["resource_caps"]["rf_max_rows"] if family == "random_forest" else None,
            encoding=configs["models"]["encoding"], target=configs["base"]["data"]["target"],
        )

    if not package.valid:
        raise RuntimeError("El paquete %s no es valido: %s" % (roles.version_id, package.invalid_reason))

    directory = package.save(configs["base"]["paths"]["models"])
    run.register_artifact("paquete_%s" % roles.version_id, directory / "manifest.json")
    LOGGER.info("Paquete exportado a %s", directory)
    return {
        "version_id": package.version_id,
        "directorio": directory.as_posix(),
        "estrategia": strategy,
        "familia": family,
        "roles": roles.to_dict(),
        "soporte": package.support,
        "politica": policy.to_dict(),
        "costos": costs.to_dict(),
    }
