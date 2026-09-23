"""Etiquetas tardias: nada puede depender de una etiqueta aun inmadura.

Se prueba mutando: si cambia el valor de una etiqueta que todavia no madura,
ninguna salida del sistema (transformaciones, predictor, calibrador, umbrales o la
p emitida) debe moverse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_adaptive.calibration import PlattCalibrator
from fraud_adaptive.decision import (
    CapacityLedger, Policy, decide_batch, simulate_outcomes,
)
from fraud_adaptive.drift import BrierAdwin, analyst_signal
from fraud_adaptive.models import FeaturePipeline, fit_model
from fraud_adaptive.splits import available_at_day, label_eligible, mature_training_mask


NUMERIC = ["TransactionAmt", "monto_log", "card_cnt_24h"]
CATEGORICAL = ["ProductCD", "card6"]
CONFIG = {"name": "lgbm_15", "num_leaves": 15, "scale_pos_weight": None}
BASE = {"learning_rate": 0.1, "n_estimators": 40, "min_child_samples": 50,
        "reg_lambda": 1.0, "n_jobs": 2}


def test_available_at_es_el_evento_mas_el_retraso():
    days = np.array([0, 10, 100])
    assert list(available_at_day(days, 30)) == [30, 40, 130]


def test_mutar_etiquetas_inmaduras_no_cambia_el_predictor(prepared_frame):
    """Prueba decisiva de que el fit respeta la madurez."""
    cutoff, delay = 90, 30
    days = prepared_frame["dia"].to_numpy()
    from fraud_adaptive.splits import Interval

    mask = mature_training_mask(days, Interval(0, 69), cutoff, delay)
    labels = prepared_frame["isFraud"].to_numpy()

    model_a = fit_model("lightgbm", CONFIG, BASE, prepared_frame.loc[mask], labels[mask],
                        NUMERIC, CATEGORICAL, seed=42, n_threads=2)

    # Se invierten todas las etiquetas aun inmaduras en el cutoff.
    mutated = labels.copy()
    immature = ~label_eligible(days, cutoff, delay)
    mutated[immature] = 1 - mutated[immature]
    assert immature.sum() > 0, "el escenario requiere etiquetas inmaduras"

    model_b = fit_model("lightgbm", CONFIG, BASE, prepared_frame.loc[mask], mutated[mask],
                        NUMERIC, CATEGORICAL, seed=42, n_threads=2)

    evaluation = prepared_frame[prepared_frame["dia"] >= 120]
    np.testing.assert_allclose(
        model_a.predict_proba(evaluation), model_b.predict_proba(evaluation), rtol=0, atol=0,
    )


def test_mutar_etiquetas_inmaduras_no_cambia_el_calibrador(prepared_frame):
    cutoff, delay = 90, 30
    days = prepared_frame["dia"].to_numpy()
    labels = prepared_frame["isFraud"].to_numpy()
    from fraud_adaptive.splits import Interval

    calibration_mask = mature_training_mask(days, Interval(69, 76), cutoff, delay)
    if calibration_mask.sum() == 0 or len(np.unique(labels[calibration_mask])) < 2:
        pytest.skip("sin soporte de calibracion en el sustituto reducido")

    scores = np.linspace(0.01, 0.99, int(calibration_mask.sum()))
    calibrator_a = PlattCalibrator().fit(scores, labels[calibration_mask])

    mutated = labels.copy()
    immature = ~label_eligible(days, cutoff, delay)
    mutated[immature] = 1 - mutated[immature]
    calibrator_b = PlattCalibrator().fit(scores, mutated[calibration_mask])

    assert calibrator_a.a == pytest.approx(calibrator_b.a)
    assert calibrator_a.b == pytest.approx(calibrator_b.b)


def test_la_accion_emitida_no_depende_de_la_etiqueta(cost_model):
    """La decision se toma con p, monto y cupo. La etiqueta solo entra al evaluar."""
    n = 300
    probability = np.linspace(0.01, 0.99, n)
    amounts = np.full(n, 150.0)
    policy = Policy(tau_low=0.2, tau_high=0.8, daily_capacity=150)

    decisions = decide_batch(probability, amounts, np.zeros(n, dtype=int), list(range(n)),
                             policy, cost_model, ledger=CapacityLedger(150))

    labels_a = np.zeros(n, dtype=int)
    labels_b = np.ones(n, dtype=int)
    outcomes_a = simulate_outcomes(decisions, labels_a, cost_model, seed=42)
    outcomes_b = simulate_outcomes(decisions, labels_b, cost_model, seed=42)

    # La accion es identica; solo cambia el costo observado.
    pd.testing.assert_series_equal(outcomes_a["accion"], outcomes_b["accion"])
    assert not outcomes_a["costo_observado"].equals(outcomes_b["costo_observado"])


def test_el_veredicto_del_analista_no_altera_decisiones_previas(cost_model):
    """El desenlace simulado se calcula al madurar, sin reescribir la accion."""
    n = 100
    policy = Policy(tau_low=0.0, tau_high=1.1, daily_capacity=n)
    decisions = decide_batch(np.full(n, 0.5), np.full(n, 100.0), np.zeros(n, dtype=int),
                             list(range(n)), policy, cost_model, ledger=CapacityLedger(n))
    original = decisions["accion"].copy()
    outcomes = simulate_outcomes(decisions, np.ones(n, dtype=int), cost_model, seed=42)
    pd.testing.assert_series_equal(outcomes["accion"], original)


def test_adwin_consume_cada_etiqueta_una_sola_vez():
    """Reanudar la corrida no puede alimentar dos veces el mismo detector."""
    detector = BrierAdwin(version_id="W30_T120", delta=0.002, clock=32)
    for event_id in range(100):
        detector.update(event_id, 0.2, 0, event_day=120, available_day=150)
    assert detector.n_updates == 100

    # Se reprocesa el mismo dia completo: no debe contar de nuevo.
    for event_id in range(100):
        detector.update(event_id, 0.2, 0, event_day=120, available_day=150)
    assert detector.n_updates == 100


def test_adwin_registra_los_dos_relojes():
    detector = BrierAdwin(version_id="W30_T120", delta=0.002, clock=32)
    rng = np.random.default_rng(42)
    for event_id in range(1200):
        # Salto de perdida a mitad del stream para forzar una deteccion.
        loss_center = 0.05 if event_id < 600 else 0.9
        detector.update(event_id, float(np.clip(loss_center + rng.normal(0, 0.05), 0, 1)),
                        0, event_day=120, available_day=150)
    if detector.detections:
        record = detector.detections[0]
        assert record["dia_evento"] == 120
        assert record["dia_disponibilidad"] == 150
        assert record["retraso_dias"] == 30


def test_s3_solo_cuenta_revisiones_maduras(cost_model):
    n = 50
    policy = Policy(tau_low=0.0, tau_high=1.1, daily_capacity=n)
    decisions = decide_batch(np.full(n, 0.5), np.full(n, 100.0), np.zeros(n, dtype=int),
                             list(range(n)), policy, cost_model, ledger=CapacityLedger(n))
    outcomes = simulate_outcomes(decisions, np.ones(n, dtype=int), cost_model, seed=42)
    signal = analyst_signal(outcomes)
    assert signal["senal"] == "S3"
    assert signal["n_revisiones"] == n
    assert "tardio" in signal["limitacion"]


def test_sin_revisiones_maduras_s3_no_inventa_una_tasa():
    empty = pd.DataFrame({"revision_admitida": pd.Series([], dtype=bool),
                          "analista_detecto_fraude": pd.Series([], dtype=bool)})
    signal = analyst_signal(empty)
    assert "motivo" in signal and signal["n_revisiones"] == 0
    assert "tasa_confirmacion" not in signal
