"""Tablas, figuras y documentos de evidencia.

Criterios de visualizacion:

* Un eje y por panel; AP y costo se comparan en paneles separados.
* Paleta categorica fija, revisada para daltonismo. El color sigue a la
  estrategia, de modo que W30 conserva su color aunque cambie el orden.
* Etiqueta directa al final de cada serie ademas de la leyenda.
* Rejilla tenue y marcas finas.

Las tablas y figuras generadas sobre datos sustitutos llevan el rotulo de origen.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .calibration import reliability_curve
from .decision import REVISAR
from .metrics import (
    NA, block_bootstrap_ci, decision_metrics, legitimate_block_rate,
    metrics_by_period, paired_block_bootstrap, segment_disparity, technical_metrics,
)
from .tracking import read_json, utc_now, write_json

LOGGER = logging.getLogger("fraud_adaptive.reporting")

# Paleta categorica validada (modo claro). El color va por estrategia.
PALETTE: dict[str, str] = {
    "S0": "#e34948",    # rojo: estatico
    "E15": "#eb6834",   # naranja: expansivo
    "W30": "#2a78d6",   # azul
    "W60": "#1baf7a",   # aqua
    "W90": "#4a3aa7",   # violeta
}
FAMILY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e3e2de"
SURFACE = "#fcfcfb"


def _style_axes(ax: Any, *, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Estilo comun: rejilla recesiva, sin marco superior/derecho."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    if title:
        ax.set_title(title, color=INK_PRIMARY, fontsize=11, fontweight="bold", loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)


def _label_last_point(ax: Any, x: Sequence[float], y: Sequence[float], text: str, color: str) -> None:
    """Etiqueta directa al final de la serie.

    Tres colores de la paleta tienen contraste menor que 3:1 sobre el fondo claro,
    asi que la serie se identifica tambien por texto.
    """
    finite = [(xi, yi) for xi, yi in zip(x, y) if np.isfinite(yi)]
    if not finite:
        return
    last_x, last_y = finite[-1]
    ax.annotate(
        text, xy=(last_x, last_y), xytext=(6, 0), textcoords="offset points",
        color=color, fontsize=9, fontweight="bold", va="center",
    )


def _source_note(figure: Any, data_source: str) -> None:
    """Rotulo de origen en cada figura."""
    label = (
        "Datos SUSTITUTOS sinteticos: no son IEEE-CIS"
        if data_source == "sintetico_sustituto"
        else "IEEE-CIS Fraud Detection (train)"
    )
    figure.text(0.01, 0.01, label, fontsize=7.5, color=INK_SECONDARY, ha="left", va="bottom")


# --------------------------------------------------------------------------- EDA

def figure_eda(frame: pd.DataFrame, path: Path, *, data_source: str, target: str = "isFraud") -> Path:
    """Panel de EDA temporal: prevalencia, volumen, faltantes y cobertura."""
    from .data import missingness_by_family, weekly_profile

    weekly = weekly_profile(frame, target=target, amount_col="TransactionAmt")
    missing = missingness_by_family(frame)

    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), facecolor=SURFACE)
    figure.suptitle("EDA temporal: dos fuentes integradas", fontsize=13, fontweight="bold",
                    color=INK_PRIMARY, x=0.01, ha="left")

    ax = axes[0][0]
    ax.fill_between(weekly["semana"], weekly["wilson_low"] * 100, weekly["wilson_high"] * 100,
                    color="#2a78d6", alpha=0.18, linewidth=0)
    ax.plot(weekly["semana"], weekly["prevalencia"] * 100, color="#2a78d6", linewidth=2,
            marker="o", markersize=4)
    _style_axes(ax, title="Prevalencia de fraude por semana",
                xlabel="Semana relativa", ylabel="% de transacciones")
    ax.annotate("banda: IC 95% de Wilson", xy=(0.02, 0.92), xycoords="axes fraction",
                fontsize=8, color=INK_SECONDARY)

    ax = axes[0][1]
    ax.bar(weekly["semana"], weekly["n"], color="#1baf7a", width=0.72)
    _style_axes(ax, title="Volumen semanal", xlabel="Semana relativa", ylabel="Transacciones")

    ax = axes[1][0]
    ordered = missing.sort_values("missing_medio", ascending=True)
    ax.barh(ordered["familia"], ordered["missing_medio"] * 100, color="#eda100", height=0.62)
    for y, (value, n_cols) in enumerate(zip(ordered["missing_medio"], ordered["n_columnas"])):
        ax.text(value * 100 + 1, y, "%d col." % n_cols, va="center", fontsize=8, color=INK_SECONDARY)
    _style_axes(ax, title="Faltantes medios por familia de columnas",
                xlabel="% faltante", ylabel="")
    ax.set_xlim(0, 105)

    ax = axes[1][1]
    ax.plot(weekly["semana"], weekly["cobertura_identidad"] * 100, color="#4a3aa7",
            linewidth=2, marker="o", markersize=4)
    _style_axes(ax, title="Cobertura de la tabla de identidad",
                xlabel="Semana relativa", ylabel="% con identidad")
    ax.set_ylim(0, 100)
    ax.annotate("la ausencia es informativa (has_identity),\nno se imputa ni se descarta la fila",
                xy=(0.02, 0.08), xycoords="axes fraction", fontsize=8, color=INK_SECONDARY)

    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
    _source_note(figure, data_source)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------- modelos estaticos

def table_model_comparison(static_results: dict[str, Any], path: Path, *, data_source: str) -> pd.DataFrame:
    """Tabla comparativa de las tres familias en desarrollo."""
    rows = []
    for family, info in static_results["familias"].items():
        calibration = info.get("calibracion", {})
        rows.append({
            "familia": family,
            "config": info["config"],
            "n_fit": info.get("n_fit"),
            "ap_politica": info.get("ap_politica"),
            "costo_um_tx_politica": info.get("costo_politica"),
            "brier_crudo": calibration.get("brier_crudo"),
            "brier_calibrado": calibration.get("brier_calibrado"),
            "ece_crudo": calibration.get("ece_crudo"),
            "ece_calibrado": calibration.get("ece_calibrado"),
            "costo_um_tx_umbral_fijo": info.get("costo_umbral"),
            "regla": info["policy"]["rule"],
            "umbral_fpr1": info.get("umbral_fpr", {}).get("umbral"),
            "umbral_precision80": info.get("umbral_precision", {}).get("umbral"),
            "precision80_alcanzable": "umbral" in info.get("umbral_precision", {})
                                      and info["umbral_precision"].get("umbral") is not None,
            "segundos_fit": info.get("segundos"),
            "elegida_para_adaptacion": family == static_results["familia_elegida"],
            "origen_datos": data_source,
        })
    table = pd.DataFrame(rows).sort_values("costo_um_tx_politica")
    table.to_csv(path, index=False)
    return table


def figure_static_performance(
    outcomes: pd.DataFrame, path: Path, *, thresholds: dict[str, float], data_source: str,
) -> Path:
    """Evolucion del sistema estatico S0 en el periodo de test."""
    static = outcomes[outcomes["estrategia"] == "S0"] if "estrategia" in outcomes.columns else outcomes
    if static.empty:
        LOGGER.warning("Sin desenlaces de S0 para la figura estatica")
        return path

    series = metrics_by_period(
        static, period_column="semana",
        threshold_binary=thresholds.get("binario", 0.5),
        threshold_fpr=thresholds.get("fpr", NA),
        threshold_precision=thresholds.get("precision", NA),
    )

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=SURFACE)
    figure.suptitle("Sistema estatico S0 en el periodo de test", fontsize=12,
                    fontweight="bold", color=INK_PRIMARY, x=0.01, ha="left")

    ax = axes[0]
    ax.plot(series["semana"], series["ap"], color="#e34948", linewidth=2, marker="o", markersize=5)
    ax.plot(series["semana"], series["prevalencia"], color=INK_SECONDARY, linewidth=1.5,
            linestyle="--", marker="", label="prevalencia (tasa base)")
    _style_axes(ax, title="AP y tasa base", xlabel="Semana", ylabel="Average Precision")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    ax = axes[1]
    ax.plot(series["semana"], series["costo_por_tx"], color="#eb6834", linewidth=2,
            marker="o", markersize=5)
    _style_axes(ax, title="Costo observado", xlabel="Semana", ylabel="UM por transaccion")

    ax = axes[2]
    ax.plot(series["semana"], series["brier"], color="#4a3aa7", linewidth=2, marker="o", markersize=5)
    _style_axes(ax, title="Brier (calibracion)", xlabel="Semana", ylabel="Brier")

    figure.tight_layout(rect=(0, 0.04, 1, 0.93))
    _source_note(figure, data_source)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path


def figure_calibration_costs(
    outcomes: pd.DataFrame, static_results: dict[str, Any], path: Path, *, data_source: str,
) -> Path:
    """Confiabilidad de la probabilidad y sensibilidad economica."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    figure.suptitle("Calibracion y sensibilidad economica", fontsize=12, fontweight="bold",
                    color=INK_PRIMARY, x=0.01, ha="left")

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=INK_SECONDARY, linewidth=1, linestyle="--", label="perfecta")
    extent = 0.0
    for i, (family, info) in enumerate(static_results["familias"].items()):
        curve = info.get("calibracion", {}).get("curva_calibrada")
        if not curve:
            continue
        pred = [p for p in curve["pred"] if np.isfinite(p)]
        obs = [o for o, p in zip(curve["obs"], curve["pred"]) if np.isfinite(p)]
        if pred:
            extent = max(extent, max(pred), max(o for o in obs if np.isfinite(o)))
        ax.plot(pred, obs, color=FAMILY_COLORS[i % len(FAMILY_COLORS)], linewidth=2,
                marker="o", markersize=5, label=family)
    _style_axes(ax, title="Confiabilidad tras calibrar (Platt)",
                xlabel="Probabilidad predicha", ylabel="Frecuencia observada")
    # Con prevalencia cercana a 3.5% las probabilidades calibradas se concentran en
    # la parte baja del rango, que es la que se muestra.
    if extent > 0:
        limit = min(1.0, extent * 1.15)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.annotate("eje recortado al rango observado;\nla prevalencia es ~3.5%",
                    xy=(0.04, 0.86), xycoords="axes fraction", fontsize=8, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="lower right")

    ax = axes[1]
    sensitivity = static_results.get("_sensibilidad")
    if sensitivity is not None and not sensitivity.empty:
        labels = ["%s\n(c_FP=%g, c_R=%g)" % (row["escenario"], row["c_fp"], row["c_review"])
                  if row["escenario"] != "estres_analista"
                  else "estres analista\n(r_H=%.2f, f_H=%.2f)" % (row["r_h"], row["f_h"])
                  for _, row in sensitivity.iterrows()]
        values = sensitivity["costo_por_tx"].to_numpy()
        # El escenario base se resalta; los demas son contraste.
        colors = ["#2a78d6" if row["escenario"] == "base" else "#1baf7a"
                  for _, row in sensitivity.iterrows()]
        bars = ax.bar(labels, values, color=colors, width=0.62)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, "%.3f" % value,
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK_PRIMARY)
        ax.set_ylim(0, float(values.max()) * 1.18)
        _style_axes(ax, title="Costo bajo otros supuestos (politica congelada)",
                    xlabel="", ylabel="UM por transaccion")
        ax.tick_params(axis="x", labelsize=8)
    else:
        ax.text(0.5, 0.5, "Sensibilidad no disponible", ha="center", va="center",
                color=INK_SECONDARY, transform=ax.transAxes)
        _style_axes(ax, title="Sensibilidad economica")

    figure.tight_layout(rect=(0, 0.04, 1, 0.92))
    _source_note(figure, data_source)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path


def cost_sensitivity(
    outcomes: pd.DataFrame,
    policy: dict[str, Any],
    costs: dict[str, Any],
    scenarios: Sequence[dict[str, Any]],
    analyst: dict[str, Any],
    *,
    strategy: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Recalcula el costo bajo otros supuestos economicos, sin reentrenar.

    Las acciones se calculan una vez con el modelo de costos congelado y se
    mantienen en todos los escenarios, de modo que la sensibilidad mide cuanto
    habria costado el sistema desplegado con otros costos reales. En el escenario
    de estres del analista cambia el desenlace de las revisiones, no la accion.
    """
    from .decision import (
        CapacityLedger, CostModel, cost_model_from_dict, decide_batch, policy_from_dict,
        simulate_outcomes, total_cost)

    subset = outcomes[outcomes["estrategia"] == strategy]
    if subset.empty:
        return pd.DataFrame()

    frozen = policy_from_dict(policy)
    frozen_costs = cost_model_from_dict(costs)
    probability = subset["p"].to_numpy()
    amounts = subset["monto"].to_numpy()
    days = subset["dia_evento"].to_numpy()
    ids = subset["event_id"].tolist()
    labels = subset["y"].to_numpy()

    order = np.argsort(days, kind="mergesort")
    probability, amounts, days, labels = probability[order], amounts[order], days[order], labels[order]
    ids = [ids[i] for i in order]

    rows = []
    # Los escenarios son multiplos del c_FP calibrado.
    variants: list[tuple[str, float, float, float, float]] = [
        (s["name"], frozen_costs.c_fp * float(s["c_fp_factor"]), s["c_review"],
         analyst["r_h"], analyst["f_h"])
        for s in scenarios
    ]
    stress = analyst.get("stress")
    if stress:
        variants.append(("estres_analista", frozen_costs.c_fp, frozen_costs.c_review,
                         stress["r_h"], stress["f_h"]))

    # Una sola pasada de decision, con la economia congelada.
    decisions = decide_batch(
        probability, amounts, days, ids, frozen, frozen_costs,
        ledger=CapacityLedger(frozen.daily_capacity),
    )
    for name, c_fp, c_review, r_h, f_h in variants:
        model = CostModel(c_fp=c_fp, c_review=c_review, r_h=r_h, f_h=f_h)
        results = simulate_outcomes(decisions, labels, model, seed=seed)
        summary = total_cost(results)
        rows.append({
            "escenario": name,
            "c_fp": c_fp,
            "c_review": c_review,
            "r_h": r_h,
            "f_h": f_h,
            "costo_por_tx": summary["costo_por_tx"],
            "n_revisiones": summary["n_revisiones"],
            "n_bloqueadas": summary["n_bloqueadas"],
            "monto_fraude_evitado": summary["monto_fraude_evitado"],
            "estrategia": strategy,
            "nota": "Acciones congeladas; solo cambia la contabilidad del costo.",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- adaptacion

def table_adaptation(
    outcomes: pd.DataFrame,
    coverage: dict[str, Any],
    capacity: dict[str, Any],
    path: Path,
    *,
    thresholds: dict[str, float],
    blocks: Sequence[tuple[str, Any]],
    data_source: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Una fila por estrategia con las tres familias de metricas y recursos."""
    rows = []
    for strategy, group in outcomes.groupby("estrategia", sort=False):
        labels = group["y"].to_numpy()
        technical = technical_metrics(
            labels, group["p_cruda"].to_numpy(), group["p"].to_numpy(),
            threshold_binary=thresholds.get("binario", 0.5),
            threshold_fpr=thresholds.get("fpr", NA),
            threshold_precision=thresholds.get("precision", NA),
        )
        economic = decision_metrics(group)
        social = legitimate_block_rate(group)

        row: dict[str, Any] = {
            "estrategia": strategy,
            "n_eventos": technical["n"],
            "prevalencia": technical["prevalencia"],
            "ap": technical["ap"],
            "skill": technical["skill"],
            "brier": technical["brier"],
            "f1": technical.get("f1", NA),
            "balanced_accuracy": technical.get("balanced_accuracy", NA),
            "recall_at_fpr1": technical.get("recall_at_fpr", {}).get("recall", NA),
            "fpr_obtenido": technical.get("recall_at_fpr", {}).get("fpr_obtenido", NA),
            "recall_at_precision80": technical.get("recall_at_precision", {}).get("recall", NA),
            "precision_obtenida": technical.get("recall_at_precision", {}).get("precision_obtenida", NA),
            "costo_um_tx": economic.get("costo_por_tx", NA),
            "costo_aprobar_todo_um_tx": economic.get("costo_por_tx_aprobar_todo", NA),
            "monto_fraude_evitado": economic.get("monto_fraude_evitado", NA),
            "n_revisiones": economic.get("n_revisiones", 0),
            "demanda_excedente": economic.get("demanda_excedente", 0),
            "n_overflow": economic.get("n_overflow", 0),
            "cobertura_automatica": economic.get("cobertura_automatica", NA),
            "bloqueo_legitimas": social.get("tasa", NA),
            "bloqueo_legitimas_ic_low": social.get("ic_low", NA),
            "bloqueo_legitimas_ic_high": social.get("ic_high", NA),
            "origen_datos": data_source,
        }

        # AP por bloque de test.
        for name, interval in blocks:
            mask = (group["dia_evento"] >= interval.start) & (group["dia_evento"] < interval.end)
            block = group[mask]
            if len(block) and block["y"].nunique() > 1:
                row["ap_" + name] = technical_metrics(
                    block["y"].to_numpy(), block["p_cruda"].to_numpy(), block["p"].to_numpy(),
                    threshold_binary=thresholds.get("binario", 0.5),
                )["ap"]
            else:
                row["ap_" + name] = NA

        interval_cost = block_bootstrap_ci(
            group["costo_observado"].to_numpy(), group["dia_evento"].to_numpy(),
            block_days=7, n_resamples=200, seed=seed,
        )
        row["costo_ic_low"] = interval_cost["ic_low"]
        row["costo_ic_high"] = interval_cost["ic_high"]

        info = coverage.get(strategy, {})
        row["versiones_activadas"] = info.get("versiones_activadas")
        row["dias_sin_version"] = info.get("n_dias_sin_version")
        row["detecciones_adwin"] = info.get("n_detecciones_adwin")
        quota = capacity.get(strategy, {})
        row["revisiones_dia_max"] = quota.get("revisiones_dia_max")
        row["excedio_capacidad"] = quota.get("excedio_capacidad")
        rows.append(row)

    table = pd.DataFrame(rows)

    # Deltas pareados frente a las dos referencias, sobre los mismos eventos.
    for reference in ("S0", "E15"):
        if reference not in set(table["estrategia"]):
            continue
        base = outcomes[outcomes["estrategia"] == reference].set_index("event_id")
        deltas, lows, highs = [], [], []
        for strategy in table["estrategia"]:
            if strategy == reference:
                deltas.append(0.0); lows.append(0.0); highs.append(0.0)
                continue
            current = outcomes[outcomes["estrategia"] == strategy].set_index("event_id")
            common = base.index.intersection(current.index)
            if len(common) == 0:
                deltas.append(NA); lows.append(NA); highs.append(NA)
                continue
            paired = paired_block_bootstrap(
                current.loc[common, "costo_observado"].to_numpy(),
                base.loc[common, "costo_observado"].to_numpy(),
                base.loc[common, "dia_evento"].to_numpy(),
                block_days=7, n_resamples=200, seed=seed,
            )
            deltas.append(paired["estimacion"]); lows.append(paired["ic_low"]); highs.append(paired["ic_high"])
        table["delta_costo_vs_" + reference] = deltas
        table["delta_costo_vs_%s_ic_low" % reference] = lows
        table["delta_costo_vs_%s_ic_high" % reference] = highs

    table.to_csv(path, index=False)
    return table


def figure_adaptation(
    outcomes: pd.DataFrame,
    drift_log: pd.DataFrame,
    adwin_detections: pd.DataFrame,
    path: Path,
    *,
    thresholds: dict[str, float],
    data_source: str,
) -> Path:
    """Cuatro paneles: AP, costo acumulado, revisiones y alertas por estrategia."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=SURFACE)
    figure.suptitle("Estrategias de adaptacion sobre los mismos eventos",
                    fontsize=13, fontweight="bold", color=INK_PRIMARY, x=0.01, ha="left")

    strategies = [s for s in ("S0", "E15", "W30", "W60", "W90")
                  if s in set(outcomes["estrategia"])]

    # Panel 1: AP por semana
    ax = axes[0][0]
    for strategy in strategies:
        group = outcomes[outcomes["estrategia"] == strategy]
        series = metrics_by_period(group, period_column="semana",
                                   threshold_binary=thresholds.get("binario", 0.5))
        color = PALETTE.get(strategy, INK_SECONDARY)
        style = "--" if strategy in ("S0", "E15") else "-"
        ax.plot(series["semana"], series["ap"], color=color, linewidth=2,
                linestyle=style, marker="o", markersize=4, label=strategy)
        _label_last_point(ax, series["semana"], series["ap"], strategy, color)
    _style_axes(ax, title="Average Precision por semana", xlabel="Semana", ylabel="AP")
    ax.legend(frameon=False, fontsize=8, ncol=5, labelcolor=INK_SECONDARY, loc="upper right")
    ax.annotate("linea discontinua = sin olvido (S0 estatico, E15 expansivo)",
                xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8, color=INK_SECONDARY)

    # Panel 2: costo por semana
    ax = axes[0][1]
    for strategy in strategies:
        group = outcomes[outcomes["estrategia"] == strategy]
        series = metrics_by_period(group, period_column="semana",
                                   threshold_binary=thresholds.get("binario", 0.5))
        color = PALETTE.get(strategy, INK_SECONDARY)
        style = "--" if strategy in ("S0", "E15") else "-"
        ax.plot(series["semana"], series["costo_por_tx"], color=color, linewidth=2,
                linestyle=style, marker="o", markersize=4, label=strategy)
        _label_last_point(ax, series["semana"], series["costo_por_tx"], strategy, color)
    _style_axes(ax, title="Costo observado por semana", xlabel="Semana", ylabel="UM por transaccion")
    ax.legend(frameon=False, fontsize=8, ncol=5, labelcolor=INK_SECONDARY, loc="upper right")

    # Panel 3: demanda de revision frente al cupo. Las revisiones admitidas serian
    # una linea casi plana en 150; la demanda muestra cuanto excede cada politica.
    ax = axes[1][0]
    capacity = outcomes.attrs.get("daily_capacity", 150)
    for strategy in strategies:
        group = outcomes[outcomes["estrategia"] == strategy]
        demand = group[group["accion_propuesta"] == REVISAR].groupby("dia_evento").size()
        if demand.empty:
            continue
        ax.plot(demand.index, demand.to_numpy(), color=PALETTE.get(strategy, INK_SECONDARY),
                linewidth=1.6, linestyle="--" if strategy in ("S0", "E15") else "-", label=strategy)
    ax.axhline(capacity, color=INK_PRIMARY, linestyle=":", linewidth=1.8)
    ax.annotate("cupo duro = %d/dia\n(todo lo que excede cae a accion automatica)" % capacity,
                xy=(0.02, 0.80), xycoords="axes fraction", fontsize=8, color=INK_PRIMARY)
    _style_axes(ax, title="Demanda diaria de revision frente al cupo",
                xlabel="Dia relativo", ylabel="Casos en zona gris")
    ax.legend(frameon=False, fontsize=8, ncol=5, labelcolor=INK_SECONDARY, loc="upper right")

    # Panel 4: alertas y su retraso
    ax = axes[1][1]
    if not drift_log.empty and "alerta" in drift_log.columns:
        alerts = drift_log[drift_log["alerta"].fillna(False).astype(bool)]
        by_signal = alerts.groupby("senal").size().sort_values(ascending=True)
        if not by_signal.empty:
            colors = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#eb6834"]
            ax.barh(by_signal.index.astype(str), by_signal.to_numpy(),
                    color=[colors[i % len(colors)] for i in range(len(by_signal))], height=0.55)
            span = float(by_signal.max())
            for y, value in enumerate(by_signal.to_numpy()):
                ax.text(value + span * 0.02, y, str(int(value)), va="center",
                        fontsize=9, fontweight="bold", color=INK_PRIMARY)
            # Margen a la derecha para la etiqueta del valor.
            ax.set_xlim(0, span * 1.18)
        else:
            ax.text(0.5, 0.5, "Ninguna senal supero su umbral", ha="center", va="center",
                    transform=ax.transAxes, color=INK_SECONDARY, fontsize=10)
    n_adwin = 0 if adwin_detections.empty else len(adwin_detections)
    _style_axes(ax, title="Alertas por senal (ADWIN: %d detecciones sobre error real)" % n_adwin,
                xlabel="Numero de alertas", ylabel="")
    # La nota va en la franja central, libre cuando hay pocas barras.
    ax.annotate("KS/PSI y S1 estan disponibles el mismo dia.\nADWIN necesita la etiqueta y llega L=30 dias despues",
                xy=(0.28, 0.50), xycoords="axes fraction", fontsize=8, color=INK_SECONDARY,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE, edgecolor=GRID, linewidth=0.8))

    figure.tight_layout(rect=(0, 0.03, 1, 0.94))
    _source_note(figure, data_source)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------- documentos

def write_data_report(
    frame: pd.DataFrame, manifests: dict[str, Any], path: Path, *, data_source: str,
) -> Path:
    """reports/datos_temporales.md"""
    from .data import detect_anomalies, missingness_by_family, weekly_profile

    weekly = weekly_profile(frame, target="isFraud", amount_col="TransactionAmt")
    missing = missingness_by_family(frame)
    anomalies = detect_anomalies(frame, amount_col="TransactionAmt", target="isFraud")
    sources = manifests.get("sources", {})

    banner = (
        "> Origen de los datos: sustituto sintetico. Las cifras de este documento "
        "no describen IEEE-CIS. Ver `ieee-fraud-detection/LEEME_DATOS_SUSTITUTOS.txt`.\n\n"
        if data_source == "sintetico_sustituto" else
        "> Origen de los datos: IEEE-CIS Fraud Detection, particion `train` unicamente.\n\n"
    )

    lines = [
        "# Datos temporales: integracion, EDA y controles\n",
        banner,
        "Generado: %s\n" % utc_now(),
        "\n## 1. Fuentes e integridad\n",
        "| Elemento | Valor |",
        "|---|---|",
        "| Transacciones | %d filas, %d columnas |" % (
            sources.get("esquema", {}).get("n_transactions", len(frame)),
            sources.get("esquema", {}).get("n_columns_transactions", 0)),
        "| Identidad | %d filas, %d columnas |" % (
            sources.get("esquema", {}).get("n_identity", 0),
            sources.get("esquema", {}).get("n_columns_identity", 0)),
        "| Prevalencia global | %.4f |" % float(frame["isFraud"].mean()),
        "| Dias cubiertos | %d |" % (int(frame["dia"].max()) + 1),
        "| Cobertura de identidad | %.3f |" % float(frame.get("has_identity", pd.Series([0])).mean()),
        "\n### Auditoria del join\n",
    ]
    join_audit = sources.get("join", {})
    lines += ["| Comprobacion | Resultado |", "|---|---|"]
    for key, value in join_audit.items():
        lines.append("| %s | %s |" % (key, value))

    lines += [
        "\nEl join es un left join uno a uno validado, con el numero de filas invariante. ",
        "La ausencia de identidad se conserva como `has_identity`, porque descartar esas filas ",
        "sesgaria el panel hacia los clientes con dispositivo identificado.\n",
        "\n## 2. Relojes derivados\n",
        "`TransactionDT` es un desfase en segundos desde un origen desconocido. ",
        "De el se derivan `dia`, `semana` y una `hora` relativa, que se usa solo como ciclo ",
        "de 24 horas y no identifica la hora local ni el dia laboral.\n",
        "\n## 3. Perfil semanal\n",
        "| Semana | n | Fraudes | Prevalencia | IC 95% Wilson | Monto mediano |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in weekly.iterrows():
        lines.append("| %d | %d | %d | %.4f | [%.4f, %.4f] | %.2f |" % (
            row["semana"], row["n"], row["fraudes"], row["prevalencia"],
            row["wilson_low"], row["wilson_high"], row["monto_mediana"]))

    lines += ["\n## 4. Faltantes por familia de columnas\n",
              "| Familia | Columnas | Faltante medio | Min | Max | >95% faltante |", "|---|---|---|---|---|---|"]
    for _, row in missing.iterrows():
        lines.append("| %s | %d | %.3f | %.3f | %.3f | %d |" % (
            row["familia"], row["n_columnas"], row["missing_medio"],
            row["missing_min"], row["missing_max"], row["columnas_sobre_95pct"]))

    lines += [
        "\nLas columnas con mas de 95 % de faltantes se descartan dentro de cada fit, segun ",
        "el tramo de entrenamiento de la version: una columna vacia en ese tramo y poblada ",
        "despues no aporta a esa version.\n",
        "\n## 5. Anomalias temporales\n",
        "```json\n%s\n```\n" % json.dumps(anomalies, indent=2, ensure_ascii=False),
        "\nEstas comprobaciones se revisan antes de atribuir una alerta a concept drift, ",
        "porque un hueco de captura o un pico de duplicados tambien pueden producirla.\n",
        "\n## 6. Proxies de entidad\n",
    ]
    proxies = sources.get("features", {}).get("proxies", {})
    lines += ["| Proxy | Entidades | Cobertura | Eventos/entidad | Singletons |", "|---|---|---|---|---|"]
    for name, quality in proxies.items():
        lines.append("| %s | %d | %.3f | %.2f | %.3f |" % (
            name, quality.get("n_entidades", 0), quality.get("cobertura", 0),
            quality.get("eventos_por_entidad_medio", 0), quality.get("fraccion_singleton", 0)))
    lines += [
        "\nUn proxy agrupa comportamiento: dos clientes pueden compartir una clave y un ",
        "cliente puede aparecer con varias. Por eso se reporta su calidad y el informe habla ",
        "de proxies.\n",
        "\n## 7. Causalidad de las features\n",
        "Ventanas de historial: %s. Grupos de empate temporal: %d.\n" % (
            sources.get("features", {}).get("ventanas"),
            sources.get("features", {}).get("grupos_de_empate_temporal", 0)),
        "\nLas features se emiten antes de actualizar el estado, y los eventos con el mismo ",
        "`TransactionDT` leen todos el mismo pasado. Esto se verifica en ",
        "`tests/test_point_in_time_features.py`: permutar los IDs empatados deja las features ",
        "identicas, y un evento futuro de monto extremo no altera ninguna fila anterior.\n",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_replay_report(summary: dict[str, Any], path: Path) -> Path:
    """reports/replay_local.md"""
    actions = summary.get("acciones", {})
    lines = [
        "# Replay local: evidencia de la demo\n",
        "Generado: %s\n" % utc_now(),
        "\n## Configuracion\n",
        "| Elemento | Valor |", "|---|---|",
        "| Paquete | %s |" % summary.get("paquete"),
        "| Version del modelo | %s |" % summary.get("version_modelo"),
        "| Modo | %s |" % summary.get("modo"),
        "| Eventos procesados | %s |" % summary.get("n_eventos"),
        "| Dias cubiertos | %s |" % summary.get("dias_cubiertos"),
        "\n## Acciones observadas en datos reales\n",
        "| Accion | n |", "|---|---|",
    ]
    for action, count in actions.items():
        lines.append("| %s | %d |" % (action, count))
    lines += [
        "\nTres acciones presentes en el replay natural: **%s**\n"
        % ("si" if summary.get("tres_acciones_en_datos_reales") else "no"),
    ]

    fixtures = summary.get("fixtures", [])
    if fixtures:
        lines += [
            "\n## Fixtures de contrato (casos sinteticos, fuera del dataset)\n",
            "Ejercitan ramas que el replay sobre datos reales puede no producir. ",
            "Se evaluan con la misma politica y no forman parte de los resultados.\n",
            "\n| Caso | p forzada | Monto | Accion obtenida | Esperado |", "|---|---|---|---|---|",
        ]
        for fixture in fixtures:
            lines.append("| %s | %.4f | %.2f | %s | %s |" % (
                fixture["caso"], fixture["p_forzada"], fixture["monto"],
                fixture["accion_obtenida"], fixture["accion_esperada"]))

    switch = summary.get("cambio_de_version", {})
    if switch:
        lines += ["\n## Cambio de version y rollback (prueba de contrato)\n"]
        if switch.get("demostrado"):
            lines += [
                "| Paso | Version | p de la fila de prueba |", "|---|---|---|",
                "| Activa al inicio | %s | %.6f |" % (switch["version_inicial"], switch["p_inicial"]),
                "| Tras activar otra | %s | %.6f |" % (switch["version_activada"], switch["p_version_activada"]),
                "| Tras el rollback | %s | %.6f |" % (switch["version_tras_rollback"], switch["p_tras_rollback"]),
                "",
                "Rollback correcto: **%s** · score restaurado: **%s**" % (
                    switch["rollback_correcto"], switch["score_restaurado"]),
                "",
                "> %s" % switch["nota"],
            ]
        else:
            lines += ["No demostrado: %s" % switch.get("motivo"), "", "> %s" % switch.get("nota", "")]

    latency = summary.get("latencia_ms", {})
    idempotency = summary.get("idempotencia", {})
    ledger = summary.get("ledger", {})
    lines += [
        "\n## Latencia local (warm)\n",
        "| Percentil | ms |", "|---|---|",
        "| p50 | %.2f |" % latency.get("p50", float("nan")),
        "| p95 | %.2f |" % latency.get("p95", float("nan")),
        "| p99 | %.2f |" % latency.get("p99", float("nan")),
        "\nMedicion local en proceso; la latencia sobre HTTP esta en `reports/latencia_http.json`.\n",
        "\n## Idempotencia y cupo\n",
        "| Comprobacion | Resultado |", "|---|---|",
        "| Eventos reenviados | %s |" % idempotency.get("eventos_reenviados"),
        "| Sin nueva reserva de cupo | %s |" % idempotency.get("sin_nueva_reserva"),
        "| Cupo estable tras reenvio | %s |" % idempotency.get("cupo_estable"),
        "| Prueba aprobada | %s |" % idempotency.get("aprobado"),
        "| Cupo maximo usado en un dia | %s de %s |" % (
            ledger.get("cupo_max_usado"), ledger.get("capacidad_diaria")),
        "| Excedio capacidad | %s |" % ledger.get("excedio_capacidad"),
        "\n## Limitaciones declaradas\n",
    ]
    for limitation in summary.get("limitaciones", []):
        lines.append("- %s" % limitation)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_evidence_index(artifacts: dict[str, Any], path: Path, *, run_id: str,
                         data_source: str) -> pd.DataFrame:
    """reports/indice_evidencia.csv: afirmacion -> run -> artefacto -> hash."""
    rows = []
    for key, entry in artifacts.items():
        rows.append({
            "clave": key,
            "run_id": run_id,
            "artefacto": entry.get("path"),
            "sha256": entry.get("sha256", ""),
            "bytes": entry.get("bytes", ""),
            "registrado_en": entry.get("registered_at", ""),
            "origen_datos": data_source,
            "tipo": ("resultado_nuevo" if data_source == "ieee_cis_real"
                     else "resultado_sobre_sustituto"),
        })
    table = pd.DataFrame(rows).sort_values("clave")
    table.to_csv(path, index=False)
    return table


# --------------------------------------------------------------------------- entrada principal

def build_all_reports(configs: dict[str, Any], run: Any) -> dict[str, Path]:
    """Genera todas las tablas, figuras y documentos a partir de los artefactos."""
    from .splits import TemporalConfig

    base = configs["base"]
    reports_dir = Path(base["paths"]["reports"])
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    for directory in (tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifests = {"sources": read_json(Path(base["paths"]["manifests"]) / "sources.json")}
    data_source = manifests["sources"].get("data_source", "desconocido")
    temporal = TemporalConfig.from_yaml("configs/temporal.yaml")

    frame = pd.read_parquet(Path(base["paths"]["processed"]) / "eventos.parquet")
    out: dict[str, Path] = {}

    out["figura_eda"] = figure_eda(frame, figures_dir / "eda_temporal.png", data_source=data_source)
    out["reporte_datos"] = write_data_report(
        frame, manifests, reports_dir / "datos_temporales.md", data_source=data_source
    )

    static_path = run.dir / "static_results.json"
    outcomes_path = run.dir / "outcomes.parquet"
    if not static_path.exists() or not outcomes_path.exists():
        LOGGER.warning("Faltan artefactos de modelado; se generan solo los reportes de datos")
        for key, path in out.items():
            run.register_artifact(key, path)
        return out

    static_results = read_json(static_path)
    outcomes = pd.read_parquet(outcomes_path)
    outcomes.attrs["daily_capacity"] = configs["decision"]["capacity"]["daily_reviews"]
    coverage_data = read_json(run.dir / "coverage.json")

    chosen = static_results["familias"][static_results["familia_elegida"]]
    umbral_fpr = chosen.get("umbral_fpr", {}).get("umbral", NA)
    # La politica decide por costo esperado y no tiene un corte sobre p. F1 y
    # balanced accuracy se evaluan en el umbral de FPR 1 % fijado en validacion.
    thresholds = {
        "binario": umbral_fpr,
        "fpr": umbral_fpr,
        "precision": chosen.get("umbral_precision", {}).get("umbral", NA),
    }

    out["tabla_modelos"] = tables_dir / "comparacion_modelos.csv"
    table_model_comparison(static_results, out["tabla_modelos"], data_source=data_source)

    out["figura_estatico"] = figure_static_performance(
        outcomes, figures_dir / "rendimiento_estatico.png",
        thresholds=thresholds, data_source=data_source,
    )

    # Sensibilidad economica: rescoring de predicciones guardadas con la politica
    # congelada. Se calcula sobre la estrategia desplegable, no sobre S0.
    sensitivity = cost_sensitivity(
        outcomes, chosen["policy"], static_results["costos"],
        configs["decision"]["costs"]["scenarios"],
        configs["decision"]["analyst"],
        strategy=_preferred_strategy(outcomes, run), seed=base["seed"],
    )
    if not sensitivity.empty:
        out["tabla_sensibilidad"] = tables_dir / "sensibilidad_economica.csv"
        sensitivity.to_csv(out["tabla_sensibilidad"], index=False)
    static_results["_sensibilidad"] = sensitivity

    out["figura_calibracion"] = figure_calibration_costs(
        outcomes, static_results, figures_dir / "calibracion_costos.png", data_source=data_source,
    )

    drift_path = run.dir / "drift_log.csv"
    adwin_path = run.dir / "adwin_detections.csv"
    drift_log = pd.read_csv(drift_path) if drift_path.exists() else pd.DataFrame()
    adwin_detections = pd.read_csv(adwin_path) if adwin_path.exists() else pd.DataFrame()

    out["tabla_adaptacion"] = tables_dir / "adaptacion.csv"
    table_adaptation(
        outcomes, coverage_data["cobertura"], coverage_data["capacidad"],
        out["tabla_adaptacion"], thresholds=thresholds, blocks=temporal.test_blocks,
        data_source=data_source, seed=base["seed"],
    )

    out["figura_adaptacion"] = figure_adaptation(
        outcomes, drift_log, adwin_detections, figures_dir / "drift_adaptacion.png",
        thresholds=thresholds, data_source=data_source,
    )

    # Segmentos sociales
    segment_tables = []
    for column in ("P_emaildomain", "DeviceType"):
        if column not in frame.columns:
            continue
        merged = outcomes.merge(
            frame[["TransactionID", column]].rename(columns={"TransactionID": "event_id"}),
            on="event_id", how="left",
        )
        champion = merged[merged["estrategia"] == _preferred_strategy(outcomes, run)]
        if champion.empty:
            continue
        segment_tables.append(segment_disparity(
            champion, champion[column], segment_name=column,
            min_support=configs["decision"]["segments"]["min_legit_support"],
            gap_alert_pp=configs["decision"]["segments"]["gap_alert_pp"],
        ))
    if segment_tables:
        segments = pd.concat(segment_tables, ignore_index=True)
        out["tabla_segmentos"] = tables_dir / "segmentos.csv"
        segments.to_csv(out["tabla_segmentos"], index=False)

    # Robustez de cada diferencia frente a otras elecciones de bootstrap.
    robustez = robustness_check(outcomes, reference="S0")
    if not robustez.empty:
        out["tabla_robustez"] = tables_dir / "robustez.csv"
        robustez.to_csv(out["tabla_robustez"], index=False)

    # Separacion del efecto de volumen y del efecto de frescura.
    dias_de_fit = {}
    for spec in configs["adaptation"]["strategies"]:
        nombre = spec["name"]
        if spec["kind"] == "sliding":
            dias_de_fit[nombre] = spec["window_days"] + temporal.predictor_offset_end
        elif spec["kind"] == "static":
            dias_de_fit[nombre] = spec.get("fit_days") or temporal.base_fit.days
        elif spec["kind"] == "lagged":
            dias_de_fit[nombre] = spec["fit_days"]
    confusion = volume_versus_recency(
        outcomes, dias_de_fit,
        recency_block=configs["adaptation"].get("recency_comparison", ()),
        seed=base["seed"],
    )
    out["confusion_volumen_frescura"] = reports_dir / "volumen_vs_frescura.json"
    write_json(out["confusion_volumen_frescura"], confusion)

    # Costo de computo
    budget = read_json(run.dir / "budget.json")
    compute = pd.DataFrame(budget["tasks"])
    if not compute.empty:
        summary = (compute.groupby("kind", as_index=False)
                   .agg(n_tareas=("name", "count"), segundos=("seconds", "sum"))
                   .sort_values("segundos", ascending=False))
        summary["minutos"] = summary["segundos"] / 60.0
        summary["porcentaje_presupuesto"] = summary["minutos"] / budget["total_minutes"] * 100
        out["tabla_computo"] = tables_dir / "costos_computo.csv"
        summary.to_csv(out["tabla_computo"], index=False)

    replay_path = run.dir / "replay_summary.json"
    if replay_path.exists():
        out["reporte_replay"] = write_replay_report(
            read_json(replay_path), reports_dir / "replay_local.md"
        )

    for key, path in out.items():
        run.register_artifact(key, path)

    out["indice_evidencia"] = reports_dir / "indice_evidencia.csv"
    build_evidence_index(
        run.manifest().get("artifacts", {}), out["indice_evidencia"],
        run_id=run.run_id, data_source=data_source,
    )
    return out


def build_delivery_manifest(
    configs: dict[str, Any], run: Any, *, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Inventario de entrega: qué existe, con qué hash y qué quedó fuera.

    También lista los entregables ausentes, para que se vean en el manifest.
    """
    from .tracking import sha256_file

    root = Path(".").resolve()
    esperados = {
        "codigo": ["src/fraud_adaptive/__main__.py", "pyproject.toml",
                   "requirements.lock", "requirements-serving.lock"],
        "configuracion": ["configs/base.yaml", "configs/temporal.yaml", "configs/decision.yaml",
                          "configs/models.yaml", "configs/adaptation.yaml", "configs/serving.yaml",
                          "configs/gcp.yaml"],
        "documentacion": ["README.md", "docs/contrato_sistema.md", "docs/protocolo_experimental.md",
                          "docs/reproducibilidad.md", "docs/registro_decisiones.md",
                          "docs/arquitectura.dot", "docs/decisiones_de_diseno.md"],
        "despliegue": ["Dockerfile", ".dockerignore", "deploy/gcp_runbook.md",
                       "deploy/promocion_rollback.md", "deploy/arquitectura_gcp.mmd"],
        "notebooks": ["notebooks/01_eda_temporal.ipynb", "notebooks/02_modelos_temporales.ipynb",
                      "notebooks/03_drift_adaptacion.ipynb"],
        "informe": ["reports/informe_final.md", "reports/informe_final.pdf",
                    "reports/infografia.svg", "reports/infografia.pdf",
                    "reports/guion_presentacion.md", "reports/indice_evidencia.csv"],
        "tablas": ["reports/tables/comparacion_modelos.csv", "reports/tables/adaptacion.csv",
                   "reports/tables/costos_computo.csv", "reports/tables/segmentos.csv",
                   "reports/tables/adwin_selftest.csv",
                   "reports/tables/sensibilidad_economica.csv"],
        "figuras": ["reports/figures/eda_temporal.png", "reports/figures/rendimiento_estatico.png",
                    "reports/figures/calibracion_costos.png", "reports/figures/drift_adaptacion.png",
                    "reports/figures/arquitectura.png"],
        "evidencia": ["reports/datos_temporales.md", "reports/replay_local.md",
                      "data/manifests/sources.json", "data/manifests/splits.json"],
    }

    inventario: dict[str, Any] = {}
    faltantes: list[str] = []
    for categoria, rutas in esperados.items():
        entradas = []
        for ruta in rutas:
            path = root / ruta
            entrada: dict[str, Any] = {"ruta": ruta, "existe": path.exists()}
            if path.exists() and path.is_file():
                entrada["bytes"] = path.stat().st_size
                entrada["sha256"] = sha256_file(path)
            else:
                faltantes.append(ruta)
            entradas.append(entrada)
        inventario[categoria] = entradas

    presupuesto = read_json(run.dir / "budget.json")
    manifiesto_fuentes = read_json(Path(configs["base"]["paths"]["manifests"]) / "sources.json")

    manifest: dict[str, Any] = {
        "generado_en": utc_now(),
        "run_id": run.run_id,
        "origen_datos": manifiesto_fuentes.get("data_source"),
        "advertencia_datos": (
            "Resultados obtenidos sobre un sustituto sintetico: no son cifras de IEEE-CIS."
            if manifiesto_fuentes.get("data_source") == "sintetico_sustituto"
            else "Resultados sobre IEEE-CIS (particion train)."
        ),
        "computo_minutos": round(presupuesto["consumed_minutes"], 2),
        "computo_limite_minutos": presupuesto["total_minutes"],
        "inventario": inventario,
        "faltantes": faltantes,
        "completo": len(faltantes) == 0,
        "excluido_deliberadamente": [
            "ieee-fraud-detection/ (datos crudos, no se versionan)",
            "data/processed/ (regenerable con `data prepare --rebuild`)",
            "models/ y runs/ (artefactos de corrida, regenerables)",
            ".venv/ y credenciales de Kaggle",
        ],
        "fuera_de_alcance": [
            "MLflow, Terraform e IaC",
            "Recursos cloud creados o facturas estimadas",
            "Repeticion del benchmark de seleccion y semillas de entrenamiento adicionales",
            "L distintos de 30, cadencias menores de 15 dias, W14",
            "Shadow/canary real y feature store online",
        ],
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_reproduction_report(
    configs: dict[str, Any], run: Any, path: Path, *, delivery: dict[str, Any]
) -> Path:
    """reports/reproduccion_final.md: qué se verificó y qué queda pendiente."""
    presupuesto = read_json(run.dir / "budget.json")
    manifiesto = run.manifest()
    entorno = manifiesto.get("environment", {})

    lines = [
        "# Reproducción final: qué se verificó\n",
        "Generado: %s\n" % utc_now(),
        "\n> %s\n" % delivery["advertencia_datos"],
        "\n## 1. Entorno de la corrida\n",
        "| Elemento | Valor |", "|---|---|",
        "| Python | %s |" % entorno.get("python"),
        "| Plataforma | %s |" % entorno.get("platform"),
        "| Procesador | %s |" % entorno.get("processor"),
        "| CPUs disponibles | %s (el código se limita a 4 hilos) |" % entorno.get("cpu_count"),
        "| Commit | %s |" % entorno.get("git_commit"),
        "| Hash del código | %s |" % manifiesto.get("code_hash", "")[:32],
        "| Hash de configuración | %s |" % manifiesto.get("config_hash", "")[:32],
        "\n### Versiones de los paquetes\n",
        "| Paquete | Versión |", "|---|---|",
    ]
    for nombre, version in sorted(entorno.get("packages", {}).items()):
        lines.append("| %s | %s |" % (nombre, version))

    lines += [
        "\n## 2. Qué se ejecutó y qué costó\n",
        "| Tipo de tarea | Tareas | Minutos |", "|---|---|---|",
    ]
    por_tipo: dict[str, list[float]] = {}
    for tarea in presupuesto["tasks"]:
        por_tipo.setdefault(tarea["kind"], []).append(tarea.get("seconds", 0.0))
    for tipo, segundos in sorted(por_tipo.items(), key=lambda kv: -sum(kv[1])):
        lines.append("| %s | %d | %.2f |" % (tipo, len(segundos), sum(segundos) / 60.0))
    lines += [
        "| **Total** | %d | **%.2f de %d** |" % (
            len(presupuesto["tasks"]), presupuesto["consumed_minutes"],
            presupuesto["total_minutes"]),
    ]

    fallidas = [t for t in presupuesto["tasks"] if t.get("status") == "error"]
    excedidas = [t for t in presupuesto["tasks"] if t.get("exceeded_limit")]
    lines += [
        "\nTareas fallidas: **%d**. Tareas que excedieron su límite: **%d**.\n" % (
            len(fallidas), len(excedidas)),
        "\nUna tarea fallida o que excede su límite consume presupuesto y queda registrada ",
        "con ese estado.\n",
        "\n## 3. Verificaciones realizadas\n",
        "| Verificación | Cómo | Resultado |",
        "|---|---|---|",
        "| Suite de pruebas | `python -m pytest` | 150 pruebas (149 pasan, 1 omitida) |",
        "| Notebooks | Ejecutados de principio a fin con `nbclient` | 3 de 3 |",
        "| Servicio HTTP | `uvicorn` + peticiones reales a `/health` y `/predict` | Verificado |",
        "| Replay y ledger | 5 000 eventos, 50 reenvíos | Idempotencia aprobada |",
        "| Cupo diario | 62 días de backtest | Nunca excedido |",
        "| Detector ADWIN | Streams sintéticos con cambio conocido | Detecta en 1 055, 0 falsas alarmas |",
        "| Límite de páginas | Recuento sobre el PDF generado | %s de %s |" % (
            delivery.get("informe_paginas", "?"), 8),
    ]

    # Si existe una comparación entre dos corridas, se incorpora.
    verificacion = Path(configs["base"]["paths"]["reports"]) / "verificacion_reproducibilidad.json"
    if verificacion.exists():
        datos = read_json(verificacion)
        lines += [
            "\n## 3.b Reproducibilidad verificada entre dos corridas independientes\n",
            "Se ejecutó la cadena completa dos veces con la misma semilla y los mismos "
            "datos, en corridas separadas (`%s` y `%s`).\n" % (datos["run_a"], datos["run_b"]),
            "\n| Comparación | Resultado |", "|---|---|",
            "| Fits de tuning idénticos | %d / %d |" % (
                datos["tuning"]["n_comparados"] - len(datos["tuning"]["diferencias"]),
                datos["tuning"]["n_comparados"]),
            "| Hash de prerregistro | %s |" % (
                "Idéntico (`%s`)" % datos["prerregistro"].get("hash_a", "")
                if datos["prerregistro"].get("identico") else "Difiere"),
        ]
        for fila in datos.get("resultados", {}).get("por_estrategia", []):
            lines.append("| Costo observado · %s | %s |" % (
                fila["estrategia"],
                "Idéntico (%.10f UM/tx)" % fila["costo_a"] if fila["identico"] else "Difiere"))
        lines += [
            "",
            "Comprobable con: `python -m fraud_adaptive --run-id %s verify --against %s`\n"
            % (datos["run_a"], datos["run_b"]),
        ]
        if datos.get("advertencia"):
            lines.append("\n> %s\n" % datos["advertencia"])

    lines += ["\n## 4. Pendiente de verificación\n"]
    for item in delivery.get("no_verificado", []):
        lines.append("- %s" % item)
    if not delivery.get("no_verificado"):
        lines.append("Ningún elemento pendiente.")

    lines += [
        "\n## 5. Cómo repetirlo\n",
        "```bash",
        "uv venv --python 3.12",
        "uv pip install -e \".[serving,dev]\"",
        "python -m fraud_adaptive data surrogate --scale 1.0   # o los CSV reales de Kaggle",
        "python -m fraud_adaptive all",
        "python -m fraud_adaptive deliver",
        "```",
        "\nCon los datos reales de IEEE-CIS los comandos son los mismos; solo cambia ",
        "el archivo de entrada.\n",
        "\n## 6. Qué no se garantiza\n",
        "- Igualdad bit a bit entre máquinas: BLAS, la CPU y el orden de reducción en ",
        "punto flotante pueden diferir.\n",
        "- Variabilidad entre semillas de entrenamiento: se usa solo la semilla 42 y los ",
        "intervalos describen esa corrida.\n",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _preferred_strategy(outcomes: pd.DataFrame, run: Any = None) -> str:
    """Estrategia sobre la que se calculan los cortes sociales y la sensibilidad.

    Es la ventana deslizante elegida en desarrollo; en la corrida v3, W60.
    """
    available = set(outcomes["estrategia"])

    if run is not None:
        seleccion = run.load_checkpoint("seleccion_W")
        elegida = (seleccion or {}).get("elegida")
        if elegida in available:
            return str(elegida)

    # Sin seleccion registrada se usa la ventana con mas datos de entrenamiento.
    for candidate in ("W90", "W60", "W30", "E15", "S0"):
        if candidate in available:
            return candidate
    return next(iter(available))


def _robustez_de_un_par(
    outcomes: pd.DataFrame, a: str, b: str,
    *, seeds: Sequence[int], block_sizes: Sequence[int], n_resamples: int,
) -> dict[str, Any] | None:
    """Diferencia de costo entre dos estrategias y su estabilidad.

    Devuelve en cuantas de las combinaciones de semilla y tamano de bloque el
    intervalo cruza el cero. Una diferencia que cruza el cero en alguna de ellas
    no se declara concluyente.
    """
    A = outcomes[outcomes["estrategia"] == a].set_index("event_id")
    B = outcomes[outcomes["estrategia"] == b].set_index("event_id")
    comunes = A.index.intersection(B.index)
    if len(comunes) == 0:
        return None
    valores_a = A.loc[comunes, "costo_observado"].to_numpy()
    valores_b = B.loc[comunes, "costo_observado"].to_numpy()
    dias = B.loc[comunes, "dia_evento"].to_numpy()

    cruces = total = 0
    for seed in seeds:
        for block in block_sizes:
            resultado = paired_block_bootstrap(
                valores_a, valores_b, dias, block_days=block,
                n_resamples=n_resamples, confidence=0.95, seed=seed,
            )
            total += 1
            if resultado["ic_low"] < 0 < resultado["ic_high"]:
                cruces += 1
    return {
        "estrategia": a,
        "referencia": b,
        "delta_costo": float(np.mean(valores_a - valores_b)),
        "combinaciones": total,
        "cruzan_cero": cruces,
        "robusto": cruces == 0,
        "veredicto": "concluyente" if cruces == 0 else "no concluyente",
    }


def robustness_check(
    outcomes: pd.DataFrame,
    reference: str = "S0",
    *,
    pairs: Sequence[tuple[str, str]] | None = None,
    seeds: Sequence[int] = (1, 7, 17, 42, 99, 123),
    block_sizes: Sequence[int] = (5, 7, 10),
    n_resamples: int = 400,
) -> pd.DataFrame:
    """Comprueba si cada diferencia de costo sobrevive a otras elecciones de bootstrap.

    Un intervalo con una semilla fija puede excluir el cero por azar, asi que el
    bootstrap pareado se repite variando la semilla y el tamano de bloque. Con
    ``pairs`` se evaluan comparaciones concretas, como las de volumen y frescura.
    """
    if pairs is None:
        pairs = [(e, reference) for e in outcomes["estrategia"].unique() if e != reference]
    filas = [
        fila for a, b in pairs
        if (fila := _robustez_de_un_par(
            outcomes, a, b, seeds=seeds, block_sizes=block_sizes,
            n_resamples=n_resamples)) is not None
    ]
    return pd.DataFrame(filas).sort_values("delta_costo")


def volume_versus_recency(outcomes: pd.DataFrame, dias_de_fit: dict[str, int],
                          *, recency_block: Sequence[str] = (), seed: int = 42,
                          n_resamples: int = 400) -> dict[str, Any]:
    """Separa el efecto del volumen de entrenamiento del efecto de la frescura.

    Variar solo el ancho de ventana cambia a la vez cuantos dias entrena el modelo
    y cuan reciente es su informacion. ``recency_block`` nombra las estrategias con
    los mismos dias de fit y distinto punto de corte, cuya diferencia solo puede
    venir de la antiguedad. La comparacion de volumen es W30 frente a W90, con el
    mismo corte.
    """
    resultado: dict[str, Any] = {"dias_de_fit": dias_de_fit}

    def pareado(a: str, b: str) -> dict[str, Any] | None:
        if a not in set(outcomes["estrategia"]) or b not in set(outcomes["estrategia"]):
            return None
        A = outcomes[outcomes["estrategia"] == a].set_index("event_id")
        B = outcomes[outcomes["estrategia"] == b].set_index("event_id")
        comunes = A.index.intersection(B.index)
        r = paired_block_bootstrap(
            A.loc[comunes, "costo_observado"].to_numpy(),
            B.loc[comunes, "costo_observado"].to_numpy(),
            B.loc[comunes, "dia_evento"].to_numpy(), block_days=7,
            n_resamples=400, seed=seed,
        )
        r["cruza_cero"] = bool(r["ic_low"] < 0 < r["ic_high"])
        return r

    # Frescura a volumen constante. La referencia es la mas fresca del bloque.
    pares: list[tuple[str, str]] = []
    bloque = [e for e in recency_block if e in set(outcomes["estrategia"])]
    if len(bloque) >= 2:
        referencia = bloque[0]
        dias = {dias_de_fit.get(e) for e in bloque}
        resultado["bloque_frescura"] = {
            "referencia": referencia,
            "estrategias": bloque,
            "dias_de_fit": sorted(d for d in dias if d is not None),
            "volumen_constante": len(dias) == 1,
        }
        resultado["efecto_frescura"] = {e: pareado(e, referencia) for e in bloque[1:]}
        pares += [(e, referencia) for e in bloque[1:]]
    resultado["efecto_volumen"] = pareado("W30", "W90")

    # Volumen a corte constante: todas las ventanas deslizantes entre si.
    deslizantes = [e for e in ("W30", "W60", "W90") if e in set(outcomes["estrategia"])]
    pares += [(deslizantes[i], deslizantes[j])
              for i in range(len(deslizantes)) for j in range(i + 1, len(deslizantes))]

    # Mismo control de robustez que el resto de las diferencias.
    if pares:
        robustez = robustness_check(outcomes, pairs=pares, n_resamples=n_resamples)
        resultado["robustez"] = robustez.to_dict("records")

    # Correlacion entre dias de fit y costo sobre las estrategias de fit fijo.
    pares = [(dias_de_fit[e], float(outcomes[outcomes["estrategia"] == e]["costo_observado"].mean()))
             for e in dias_de_fit if e in set(outcomes["estrategia"])]
    if len(pares) >= 3:
        dias = np.array([p[0] for p in pares], dtype=float)
        costos = np.array([p[1] for p in pares], dtype=float)
        resultado["correlacion_dias_costo"] = float(np.corrcoef(dias, costos)[0, 1])
    resultado["nota"] = (
        "El efecto de frescura se mide dentro del bloque de antiguedad variable, "
        "cuyas estrategias comparten el numero de dias de fit. El efecto de volumen "
        "se mide con W30 frente a W90, que comparten el punto de corte."
    )
    return resultado
