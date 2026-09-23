"""El backtest completo se ejecuta y propaga la economia congelada.

Las demas pruebas cubren piezas sueltas; esta recorre la cadena completa con datos
reducidos. Comprueba que el backtest termina, que produce las siete estrategias y
que el ``c_FP`` calibrado llega al manifiesto de cada paquete guardado.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fraud_adaptive.backtest import run_backtest
from fraud_adaptive.decision import CostModel, Policy

STRATEGIES = [
    {"name": "S0", "kind": "static", "window_days": None},
    {"name": "E15", "kind": "expanding", "window_days": None},
    {"name": "W30", "kind": "sliding", "window_days": 30},
    {"name": "W90", "kind": "sliding", "window_days": 90},
    {"name": "R46_medio", "kind": "lagged", "fit_days": 46, "lag_days": 30},
    {"name": "R46_antiguo", "kind": "static", "fit_days": 46},
]

GATES = {
    "cost_ratio_max": 1.01, "ap_drop_max": 0.01, "brier_increase_max": 0.005,
    "segment_block_increase_pp_max": 2.0, "cooldown_days": 15,
    "human_approval_required": True,
}


@pytest.fixture(scope="module")
def resultado(prepared_frame, temporal_config, tmp_path_factory):
    import dataclasses

    # El frame de prueba tiene ~64 tx/dia, de modo que ningun rol alcanza el
    # soporte minimo de produccion. Se baja el umbral porque lo que esta prueba
    # verifica es el recorrido completo, no el criterio de soporte, que ya cubre
    # `test_temporal_integrity.py`.
    temporal = dataclasses.replace(temporal_config, min_support={
        "fit": {"fraud": 5, "legit": 50},
        "calibration": {"fraud": 2, "legit": 20},
        "promotion_validation": {"fraud": 2, "legit": 20},
    })
    numeric = [c for c in ("TransactionAmt", "monto_log", "card_cnt_24h", "cliente_cnt_24h")
               if c in prepared_frame.columns]
    models_dir = tmp_path_factory.mktemp("paquetes")
    costs = CostModel(c_fp=75.0, c_review=1.0, r_h=0.90, f_h=0.02)
    salida = run_backtest(
        prepared_frame,
        strategies=STRATEGIES,
        temporal=temporal,
        family="lightgbm",
        model_config={"name": "lgbm_15", "num_leaves": 15, "n_estimators": 40},
        model_base={},
        numeric_columns=numeric,
        categorical_columns=["ProductCD"],
        policy=Policy(tau_low=0.0, tau_high=1.0, daily_capacity=150),
        cost_model=costs,
        calibration_params={"C": 1.0, "solver": "lbfgs", "max_iter": 200},
        adaptation_config=yaml.safe_load(
            Path("configs/adaptation.yaml").read_text(encoding="utf-8")),
        promotion_gates=GATES,
        monitor_panel=numeric,
        reference_frame=prepared_frame[prepared_frame["dia"] < 60],
        seed=42,
        shared_initial=["E15", "W90"],
        models_dir=str(models_dir),
    )
    return salida, costs, Path(models_dir)


def test_el_backtest_recorre_las_siete_estrategias(resultado):
    salida, _, _ = resultado
    assert not salida.outcomes.empty
    assert set(salida.outcomes["estrategia"]) == {s["name"] for s in STRATEGIES}


def test_la_economia_congelada_llega_al_paquete(resultado):
    """El manifiesto es lo que consume el servicio. Si ahi no viaja el c_FP
    calibrado, produccion decidiria con una economia distinta de la validada."""
    _, costs, models_dir = resultado
    manifiestos = list(models_dir.rglob("manifest.json"))
    assert manifiestos, "el backtest debe persistir al menos un paquete"
    for ruta in manifiestos:
        manifest = json.loads(ruta.read_text(encoding="utf-8"))
        assert manifest["costos"]["c_fp"] == costs.c_fp, ruta.name
        assert manifest["policy"]["rule"] == "argmin", ruta.name


def test_ninguna_decision_excede_el_cupo(resultado):
    salida, _, _ = resultado
    por_dia = (salida.outcomes[salida.outcomes["revision_admitida"]]
               .groupby(["estrategia", "dia_evento"]).size())
    assert por_dia.empty or int(por_dia.max()) <= 150
