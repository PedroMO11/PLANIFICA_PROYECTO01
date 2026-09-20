"""Comportamiento de las ventanas de olvido y de los detectores de drift.

No se prueba que el sistema adaptativo GANE: eso es un resultado empirico que
puede ser negativo. Se prueba que las ventanas esten bien construidas, que los
detectores reaccionen ante un cambio conocido y que no se alarmen sin motivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_adaptive.data import psi
from fraud_adaptive.drift import (
    CovariateMonitor, adwin_selftest, domain_classifier, score_drift,
)
from fraud_adaptive.splits import build_version_roles
from fraud_adaptive.synthetic import make_adwin_streams


# --------------------------------------------------------------------------- ventanas

def test_la_ventana_deslizante_olvida_el_pasado_lejano(temporal_config):
    """W30 en el ultimo corte no puede seguir viendo los primeros dias."""
    last = temporal_config.update_times[-1]
    roles = build_version_roles("W30", last, temporal_config, window_days=30, kind="sliding")
    assert roles.predictor.start > 0
    assert not roles.predictor.contains(0)


def test_la_expansiva_nunca_olvida(temporal_config):
    for update_time in temporal_config.update_times:
        roles = build_version_roles("E15", update_time, temporal_config, kind="expanding")
        assert roles.predictor.start == 0


def test_la_estatica_no_se_mueve(temporal_config):
    first = temporal_config.update_times[0]
    roles = build_version_roles("S0", first, temporal_config, kind="static")
    assert roles.predictor == temporal_config.base_fit


def test_ventanas_mas_grandes_dan_mas_dias_de_fit(temporal_config):
    widths = [
        build_version_roles("W%d" % w, 150, temporal_config, window_days=w, kind="sliding").predictor.days
        for w in (30, 60, 90)
    ]
    assert widths == sorted(widths)
    assert len(set(widths)) == 3, "las tres ventanas deben ser distinguibles"


def test_ventana_deslizante_requiere_tamano(temporal_config):
    with pytest.raises(ValueError, match="window_days"):
        build_version_roles("W", 120, temporal_config, kind="sliding")


# --------------------------------------------------------------------------- PSI / KS

def test_psi_es_cero_sin_cambio():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=5000)
    assert psi(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_crece_con_el_desplazamiento():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    small = psi(reference, rng.normal(0.2, 1, 5000))
    large = psi(reference, rng.normal(2.0, 1, 5000))
    assert large > small
    assert large > 0.20, "un desplazamiento de 2 sigma debe superar el umbral de alerta"


def test_los_bordes_de_bin_vienen_de_la_referencia():
    """Si los bins se recalcularan en cada ventana, el PSI mediria el rebinning."""
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 4000)
    shifted = rng.normal(3.0, 1, 4000)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    assert psi(reference, shifted, bins=edges) > 0.5


def test_monitor_exige_dos_cierres_consecutivos():
    """Una alerta con un solo cierre seria demasiado sensible al ruido diario."""
    rng = np.random.default_rng(2)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 3000), "y": rng.normal(0, 1, 3000)})
    monitor = CovariateMonitor(reference=reference, columns=["x", "y"], consecutive_closes=2)

    shifted = pd.DataFrame({"x": rng.normal(4.0, 1, 1500), "y": rng.normal(4.0, 1, 1500)})
    first = monitor.evaluate(shifted, day=120)
    assert first["cierres_consecutivos"] == 1
    assert not first["alerta"], "un solo cierre no debe alertar"

    second = monitor.evaluate(shifted, day=121)
    assert second["alerta"], "dos cierres consecutivos si deben alertar"


def test_monitor_no_alerta_sin_cambio():
    rng = np.random.default_rng(3)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 4000), "y": rng.normal(0, 1, 4000)})
    monitor = CovariateMonitor(reference=reference, columns=["x", "y"], consecutive_closes=2)
    for day in (120, 121, 122):
        result = monitor.evaluate(
            pd.DataFrame({"x": rng.normal(0, 1, 1500), "y": rng.normal(0, 1, 1500)}), day=day
        )
        assert not result["alerta"]


# --------------------------------------------------------------------------- S1 / S2

def test_domain_classifier_detecta_cambio_de_covariables():
    rng = np.random.default_rng(4)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    recent = pd.DataFrame({"a": rng.normal(3, 1, 2000), "b": rng.normal(3, 1, 2000)})
    result = domain_classifier(reference, recent, ["a", "b"], seed=42)
    assert result["auc"] > 0.9
    assert result["alerta"] is True
    assert "P(X)" in result["limitacion"], "debe declarar que no prueba cambio en P(y|X)"


def test_domain_classifier_no_alerta_sin_cambio():
    rng = np.random.default_rng(5)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    recent = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    result = domain_classifier(reference, recent, ["a", "b"], seed=42)
    assert result["auc"] < 0.75
    assert result["alerta"] is False


def test_domain_classifier_declara_soporte_insuficiente():
    small = pd.DataFrame({"a": [1.0, 2.0]})
    result = domain_classifier(small, small, ["a"], seed=42)
    assert result.get("motivo") == "soporte_insuficiente"
    assert np.isnan(result["auc"])


def test_s2_detecta_desplazamiento_de_scores():
    rng = np.random.default_rng(6)
    calibration = rng.beta(2, 20, 3000)
    similar = score_drift(calibration, rng.beta(2, 20, 3000))
    shifted = score_drift(calibration, rng.beta(8, 4, 3000))
    assert not similar["alerta"]
    assert shifted["alerta"]
    assert "version" in shifted["limitacion"]


# --------------------------------------------------------------------------- ADWIN

def test_adwin_detecta_un_salto_conocido_y_no_alarma_sin_el():
    """Prueba del instrumento contra verdad conocida, no resultado sobre fraude."""
    streams = make_adwin_streams(n_obs=2000, seed=42, jump_at=1000)
    table = adwin_selftest(streams, delta=0.002, clock=32, jump_at=1000)

    jump = table[table["stream"] == "jump"].iloc[0]
    stationary = table[table["stream"] == "stationary"].iloc[0]

    assert jump["n_detecciones"] >= 1, "debe detectar el cambio inyectado"
    assert jump["retardo_obs"] is not None and jump["retardo_obs"] < 300
    assert stationary["n_detecciones"] == 0, "no debe alarmarse en un stream estacionario"


def test_el_selftest_reporta_retardo_y_falsas_alarmas():
    streams = make_adwin_streams(n_obs=2000, seed=42, jump_at=1000)
    table = adwin_selftest(streams, delta=0.002, clock=32, jump_at=1000)
    for column in ("retardo_obs", "falsas_alarmas", "delta", "clock"):
        assert column in table.columns
