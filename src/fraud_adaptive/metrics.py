"""Metricas tecnicas, de decision y sociales.

* PR-AUC se calcula como Average Precision, ``sum (R_n - R_{n-1}) * P_n``, sin
  interpolacion trapezoidal, igual que en el benchmark de seleccion del dataset.
* Las metricas con restriccion (recall a FPR fijo o a precision fija) devuelven el
  nivel obtenido junto al recall, y N/A con motivo si ningun umbral la cumple.
* Las metricas por segmento marcan ``evidencia_insuficiente`` cuando el grupo no
  alcanza el soporte minimo.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .calibration import brier_score, expected_calibration_error
from .data import wilson_interval
from .decision import APROBAR, BLOQUEAR, REVISAR

LOGGER = logging.getLogger("fraud_adaptive.metrics")

NA = float("nan")


# --------------------------------------------------------------------------- tecnicas

def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return NA
    return float(average_precision_score(labels, scores))


def skill_score(ap: float, prevalence: float) -> float:
    """AP normalizado contra el clasificador aleatorio, cuyo AP es la prevalencia.

    Permite comparar bloques con prevalencias distintas.
    """
    if not np.isfinite(ap) or not np.isfinite(prevalence) or prevalence >= 1.0:
        return NA
    return float((ap - prevalence) / (1.0 - prevalence))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return NA
    return float(roc_auc_score(labels, np.asarray(scores, dtype=float)))


def confusion_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    labels = np.asarray(labels)
    predicted = np.asarray(scores, dtype=float) >= threshold
    return {
        "tp": int(np.sum(predicted & (labels == 1))),
        "fp": int(np.sum(predicted & (labels == 0))),
        "tn": int(np.sum(~predicted & (labels == 0))),
        "fn": int(np.sum(~predicted & (labels == 1))),
    }


def f1_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """F1, precision, recall y balanced accuracy en un umbral declarado."""
    matrix = confusion_at_threshold(labels, scores, threshold)
    tp, fp, tn, fn = matrix["tp"], matrix["fp"], matrix["tn"], matrix["fn"]
    precision = tp / (tp + fp) if (tp + fp) else NA
    recall = tp / (tp + fn) if (tp + fn) else NA
    specificity = tn / (tn + fp) if (tn + fp) else NA
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = NA
    balanced = (recall + specificity) / 2 if np.isfinite(recall) and np.isfinite(specificity) else NA
    return {
        "umbral": float(threshold), "f1": f1, "precision": precision, "recall": recall,
        "specificity": specificity, "balanced_accuracy": balanced, **matrix,
    }


def threshold_for_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> dict[str, Any]:
    """Umbral mas bajo cuyo FPR no supera el objetivo, lo que maximiza el recall.

    Se elige en validacion y se aplica sin cambios al test, donde se reporta el FPR
    que efectivamente se obtiene.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    negatives = scores[labels == 0]
    if negatives.size == 0:
        return {"umbral": NA, "motivo": "sin_negativos"}
    # El cuantil (1 - fpr) de los negativos deja como maximo esa fraccion por encima.
    threshold = float(np.quantile(negatives, 1.0 - target_fpr))
    achieved = float(np.mean(negatives >= threshold))
    if achieved > target_fpr:
        # Con muchos empates en el score el cuantil puede no alcanzar el objetivo.
        candidates = np.unique(negatives[negatives >= threshold])
        for candidate in candidates:
            if float(np.mean(negatives >= candidate)) <= target_fpr:
                threshold = float(candidate)
                achieved = float(np.mean(negatives >= threshold))
                break
    return {"umbral": threshold, "fpr_en_validacion": achieved, "objetivo_fpr": target_fpr}


def threshold_for_precision(labels: np.ndarray, scores: np.ndarray, target_precision: float) -> dict[str, Any]:
    """Umbral mas bajo que alcanza la precision objetivo, para maximizar el recall.

    Si ningun umbral la alcanza devuelve N/A con el motivo.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or labels.sum() == 0:
        return {"umbral": NA, "motivo": "sin_positivos"}

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_labels == 1)
    positives_predicted = np.arange(1, labels.size + 1)
    precision = tp / positives_predicted

    feasible = np.where(precision >= target_precision)[0]
    if feasible.size == 0:
        return {
            "umbral": NA,
            "motivo": "precision_objetivo_inalcanzable",
            "precision_maxima_observada": float(np.max(precision)),
            "objetivo_precision": target_precision,
        }
    cut = int(feasible[-1])
    return {
        "umbral": float(sorted_scores[cut]),
        "precision_en_validacion": float(precision[cut]),
        "objetivo_precision": target_precision,
    }


def recall_at_constraint(
    labels: np.ndarray, scores: np.ndarray, threshold: float, *, kind: str
) -> dict[str, Any]:
    """Recall en un umbral fijo, junto al FPR y la precision obtenidos."""
    if not np.isfinite(threshold):
        return {"recall": NA, "motivo": "umbral_no_disponible", "restriccion": kind}
    stats = f1_at_threshold(labels, scores, threshold)
    fpr = stats["fp"] / (stats["fp"] + stats["tn"]) if (stats["fp"] + stats["tn"]) else NA
    return {
        "restriccion": kind,
        "umbral": float(threshold),
        "recall": stats["recall"],
        "fpr_obtenido": fpr,
        "precision_obtenida": stats["precision"],
        "tp": stats["tp"], "fp": stats["fp"], "fn": stats["fn"], "tn": stats["tn"],
    }


def technical_metrics(
    labels: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    *,
    threshold_binary: float,
    threshold_fpr: float = NA,
    threshold_precision: float = NA,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    n = int(labels.size)
    prevalence = float(labels.mean()) if n else NA
    ap = average_precision(labels, calibrated_scores)

    out: dict[str, Any] = {
        "n": n,
        "n_fraudes": int(labels.sum()) if n else 0,
        "prevalencia": prevalence,
        "ap": ap,
        "skill": skill_score(ap, prevalence),
        "roc_auc": roc_auc(labels, calibrated_scores),
        "brier": brier_score(calibrated_scores, labels) if n else NA,
        "brier_crudo": brier_score(raw_scores, labels) if n else NA,
        "ece": expected_calibration_error(calibrated_scores, labels) if n else NA,
    }
    if n == 0 or len(np.unique(labels)) < 2:
        out["motivo_na"] = "sin_ambas_clases"
        return out

    binary = f1_at_threshold(labels, calibrated_scores, threshold_binary)
    out.update({
        "f1": binary["f1"],
        "precision_umbral_politica": binary["precision"],
        "recall_umbral_politica": binary["recall"],
        "balanced_accuracy": binary["balanced_accuracy"],
        "umbral_binario": threshold_binary,
    })
    out["recall_at_fpr"] = recall_at_constraint(labels, calibrated_scores, threshold_fpr, kind="fpr")
    out["recall_at_precision"] = recall_at_constraint(
        labels, calibrated_scores, threshold_precision, kind="precision"
    )
    return out


# --------------------------------------------------------------------------- decision

def decision_metrics(outcomes: pd.DataFrame, *, analyst_hours_rate: float = 10.0) -> dict[str, Any]:
    """Metricas economicas y de capacidad sobre desenlaces ya madurados."""
    n = len(outcomes)
    if n == 0:
        return {"n": 0, "motivo_na": "sin_eventos"}

    reviews = int(outcomes["revision_admitida"].sum())
    overflow = int((outcomes["motivo"] == "overflow_cupo_menor_costo_esperado").sum())
    demand = int((outcomes["accion_propuesta"] == REVISAR).sum())
    hours = reviews / analyst_hours_rate if analyst_hours_rate else NA

    approve_all_cost = float(outcomes.loc[outcomes["y"] == 1, "monto"].sum())
    observed_cost = float(outcomes["costo_observado"].sum())

    by_day = outcomes[outcomes["revision_admitida"]].groupby("dia").size()

    return {
        "n": n,
        "costo_por_tx": float(outcomes["costo_observado"].mean()),
        "costo_total": observed_cost,
        "monto_fraude_evitado": float(outcomes["monto_fraude_evitado"].sum()),
        "monto_fraude_aprobado": float(outcomes.loc[outcomes["fraude_aprobado"], "monto"].sum()),
        "costo_falsos_positivos": float(outcomes["legitima_bloqueada"].sum()),
        "n_revisiones": reviews,
        "demanda_revision": demand,
        "demanda_excedente": max(0, demand - reviews),
        "n_overflow": overflow,
        "utilizacion_cupo": float(by_day.mean()) if not by_day.empty else 0.0,
        "dias_en_capacidad_plena": int((by_day >= outcomes.attrs.get("daily_capacity", 150)).sum()) if not by_day.empty else 0,
        "horas_analista": hours,
        # Cuanto costo evita el sistema frente a aprobar todo, por hora de analista.
        "retorno_por_analista_hora": float((approve_all_cost - observed_cost) / hours) if hours else NA,
        "costo_aprobar_todo": approve_all_cost,
        "costo_por_tx_aprobar_todo": approve_all_cost / n,
        "cobertura_automatica": float(((outcomes["accion"] == APROBAR) | (outcomes["accion"] == BLOQUEAR)).mean()),
        "n_aprobadas": int((outcomes["accion"] == APROBAR).sum()),
        "n_bloqueadas": int((outcomes["accion"] == BLOQUEAR).sum()),
    }


# --------------------------------------------------------------------------- sociales

def legitimate_block_rate(outcomes: pd.DataFrame) -> dict[str, Any]:
    """Tasa de bloqueo de clientes legitimos.

    A diferencia del FPR del clasificador, mide el desenlace del sistema e incluye
    las legitimas que el analista bloqueo por error.
    """
    legit = outcomes[outcomes["y"] == 0]
    if legit.empty:
        return {"tasa": NA, "n_legitimas": 0, "motivo_na": "sin_legitimas"}
    blocked = int(legit["legitima_bloqueada"].sum())
    low, high = wilson_interval(blocked, len(legit))
    return {
        "tasa": blocked / len(legit),
        "n_legitimas": int(len(legit)),
        "n_bloqueadas": blocked,
        "ic_low": low,
        "ic_high": high,
        "tasa_revision_legitimas": float(legit["revision_admitida"].mean()),
    }


def segment_disparity(
    outcomes: pd.DataFrame,
    segment_values: pd.Series,
    *,
    segment_name: str,
    min_support: int = 1000,
    gap_alert_pp: float = 2.0,
) -> pd.DataFrame:
    """Bloqueo de legitimas por grupo, con soporte, intervalos y alerta de brecha.

    Los grupos con soporte insuficiente se conservan marcados como
    ``evidencia_insuficiente``. Los segmentos son variables de negocio, porque
    IEEE-CIS no contiene atributos protegidos.
    """
    frame = outcomes.copy()
    frame["_segmento"] = segment_values.fillna("desconocido").astype(str).to_numpy()
    legit = frame[frame["y"] == 0]
    if legit.empty:
        return pd.DataFrame()

    global_rate = float(legit["legitima_bloqueada"].mean())
    rows = []
    for group, subset in legit.groupby("_segmento", sort=True):
        n = int(len(subset))
        blocked = int(subset["legitima_bloqueada"].sum())
        rate = blocked / n if n else NA
        low, high = wilson_interval(blocked, n)
        sufficient = n >= min_support
        gap_pp = (rate - global_rate) * 100.0 if np.isfinite(rate) else NA
        rows.append({
            "variable": segment_name,
            "grupo": group,
            "n_legitimas": n,
            "n_bloqueadas": blocked,
            "tasa_bloqueo": rate,
            "ic_low": low,
            "ic_high": high,
            "tasa_global": global_rate,
            "brecha_pp": gap_pp,
            "soporte_suficiente": sufficient,
            "alerta_brecha": bool(sufficient and np.isfinite(gap_pp) and abs(gap_pp) > gap_alert_pp),
            "evidencia_insuficiente": not sufficient,
        })
    return pd.DataFrame(rows).sort_values("n_legitimas", ascending=False)


def amount_quartile_bins(amounts: np.ndarray) -> np.ndarray:
    """Bordes de cuartil de monto. Se fijan en desarrollo y se congelan."""
    return np.unique(np.quantile(np.asarray(amounts, dtype=float), [0.0, 0.25, 0.5, 0.75, 1.0]))


def apply_amount_bins(amounts: np.ndarray, edges: np.ndarray) -> pd.Series:
    labels = ["Q1", "Q2", "Q3", "Q4"][: max(1, len(edges) - 1)]
    indices = np.clip(np.digitize(np.asarray(amounts, dtype=float), edges[1:-1], right=False), 0, len(labels) - 1)
    return pd.Series([labels[i] for i in indices])


# --------------------------------------------------------------------------- incertidumbre

def block_bootstrap_ci(
    values: np.ndarray,
    days: np.ndarray,
    *,
    block_days: int = 7,
    n_resamples: int = 200,
    confidence: float = 0.95,
    seed: int = 42,
    statistic: str = "mean",
) -> dict[str, float]:
    """Intervalo por remuestreo de bloques contiguos de dias.

    Los eventos de una misma semana comparten regimen, asi que un bootstrap por
    filas daria intervalos demasiado estrechos. El intervalo describe una corrida y
    no incluye la variabilidad entre semillas de entrenamiento.
    """
    values = np.asarray(values, dtype=float)
    days = np.asarray(days)
    if values.size == 0:
        return {"estimacion": NA, "ic_low": NA, "ic_high": NA, "n_bloques": 0}

    block_id = days // block_days
    unique_blocks = np.unique(block_id)
    if unique_blocks.size < 2:
        point = float(np.mean(values)) if statistic == "mean" else float(np.sum(values))
        return {"estimacion": point, "ic_low": NA, "ic_high": NA, "n_bloques": int(unique_blocks.size),
                "motivo_na": "bloques_insuficientes"}

    index_by_block = {b: np.where(block_id == b)[0] for b in unique_blocks}
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        chosen = rng.choice(unique_blocks, size=unique_blocks.size, replace=True)
        sample = np.concatenate([index_by_block[b] for b in chosen])
        estimates[i] = np.mean(values[sample]) if statistic == "mean" else np.sum(values[sample])

    alpha = (1.0 - confidence) / 2.0
    point = float(np.mean(values)) if statistic == "mean" else float(np.sum(values))
    return {
        "estimacion": point,
        "ic_low": float(np.quantile(estimates, alpha)),
        "ic_high": float(np.quantile(estimates, 1.0 - alpha)),
        "n_bloques": int(unique_blocks.size),
        "n_resamples": n_resamples,
    }


def paired_block_bootstrap(
    values_a: np.ndarray,
    values_b: np.ndarray,
    days: np.ndarray,
    *,
    block_days: int = 7,
    n_resamples: int = 200,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Intervalo de la diferencia a-b remuestreando los mismos bloques para ambas.

    Dos estrategias evaluadas sobre los mismos eventos comparten el ruido del
    periodo; remuestrearlas por separado ensancharia el intervalo.
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if values_a.shape != values_b.shape:
        raise ValueError("Las series pareadas deben tener el mismo tamano")
    return block_bootstrap_ci(
        values_a - values_b, days, block_days=block_days, n_resamples=n_resamples,
        confidence=confidence, seed=seed, statistic="mean",
    )


# --------------------------------------------------------------------------- agregacion

def metrics_by_period(
    outcomes: pd.DataFrame,
    *,
    period_column: str = "semana",
    threshold_binary: float,
    threshold_fpr: float = NA,
    threshold_precision: float = NA,
) -> pd.DataFrame:
    """Serie temporal de metricas por periodo."""
    rows = []
    for period, group in outcomes.groupby(period_column, sort=True):
        labels = group["y"].to_numpy()
        technical = technical_metrics(
            labels, group["p_cruda"].to_numpy(), group["p"].to_numpy(),
            threshold_binary=threshold_binary,
            threshold_fpr=threshold_fpr,
            threshold_precision=threshold_precision,
        )
        economic = decision_metrics(group)
        row = {
            period_column: period,
            "n": technical["n"],
            "prevalencia": technical["prevalencia"],
            "ap": technical["ap"],
            "skill": technical["skill"],
            "brier": technical["brier"],
            "f1": technical.get("f1", NA),
            "costo_por_tx": economic.get("costo_por_tx", NA),
            "n_revisiones": economic.get("n_revisiones", 0),
            "monto_fraude_evitado": economic.get("monto_fraude_evitado", NA),
        }
        recall_fpr = technical.get("recall_at_fpr", {})
        recall_precision = technical.get("recall_at_precision", {})
        row["recall_at_fpr1"] = recall_fpr.get("recall", NA)
        row["fpr_obtenido"] = recall_fpr.get("fpr_obtenido", NA)
        row["recall_at_precision80"] = recall_precision.get("recall", NA)
        row["precision_obtenida"] = recall_precision.get("precision_obtenida", NA)
        rows.append(row)
    return pd.DataFrame(rows)
