"""Validacion de promocion sobre un holdout limpio.

H debe ser posterior a todo ajuste y no haber sido visto por ninguno de los dos
paquetes comparados. Estas pruebas verifican que un H contaminado haga rechazar el
gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_adaptive.adaptation import (
    ModelPackage, assert_package_roles_disjoint, authorization_manifest, promotion_gate,
)
from fraud_adaptive.calibration import PlattCalibrator
from fraud_adaptive.decision import Policy
from fraud_adaptive.splits import build_version_roles

GATES = {
    "cost_ratio_max": 1.01,
    "ap_drop_max": 0.01,
    "brier_increase_max": 0.005,
    "segment_block_increase_pp_max": 2.0,
}


class _FakeModel:
    """Predictor de prueba: devuelve una probabilidad fija o derivada del monto."""

    def __init__(self, base: float = 0.1, amount_weight: float = 0.0):
        self.base = base
        self.amount_weight = amount_weight
        self.fit_ids: set = set()
        self.pipeline = None
        self.estimator = None

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        value = self.base + self.amount_weight * (frame["TransactionAmt"].to_numpy() / 1000.0)
        return np.clip(value, 1e-6, 1 - 1e-6)

    def describe(self) -> dict:
        return {"family": "fake", "config": "fake"}


def _package(version: str, roles, *, base: float = 0.1, amount_weight: float = 0.0,
             ids_by_role=None) -> ModelPackage:
    return ModelPackage(
        version_id=version, strategy="W30", update_time=roles.update_time, roles=roles,
        model=_FakeModel(base, amount_weight), calibrator=PlattCalibrator(),
        policy=Policy(tau_low=0.3, tau_high=0.7, daily_capacity=150),
        valid=True, ids_by_role=ids_by_role or {},
    )


def _holdout(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "TransactionID": np.arange(10_000, 10_000 + n),
        "TransactionAmt": rng.lognormal(3.5, 0.8, n),
        "dia": np.repeat(np.arange(n // 50), 50)[:n],
    })
    labels = rng.binomial(1, 0.05, n)
    return frame, labels


@pytest.fixture
def roles(temporal_config):
    return build_version_roles("W30", 135, temporal_config, window_days=30, kind="sliding")


def test_h_contaminado_con_el_fit_rechaza_el_gate(roles, cost_model):
    frame, labels = _holdout()
    shared = set(frame["TransactionID"][:10])
    challenger = _package("W30_T135", roles, ids_by_role={"predictor": shared})

    result = promotion_gate(challenger, None, frame, labels,
                            cost_model=cost_model, gates=GATES)
    assert result["recomendacion"] == "rechazar"
    assert "H_contaminado" in result["motivo"]


def test_h_contaminado_con_la_calibracion_rechaza_el_gate(roles, cost_model):
    frame, labels = _holdout()
    challenger = _package("W30_T135", roles,
                          ids_by_role={"calibracion": set(frame["TransactionID"][:5])})
    result = promotion_gate(challenger, None, frame, labels,
                            cost_model=cost_model, gates=GATES)
    assert result["recomendacion"] == "rechazar"
    assert "calibracion" in result["motivo"]


def test_h_contaminado_en_el_champion_tambien_rechaza(roles, cost_model):
    """Ambos paquetes deben ser ciegos a H, no solo el challenger."""
    frame, labels = _holdout()
    challenger = _package("W30_T135", roles, ids_by_role={"predictor": {1, 2}})
    champion = _package("W30_T120", roles,
                        ids_by_role={"predictor": set(frame["TransactionID"][:5])})
    result = promotion_gate(challenger, champion, frame, labels,
                            cost_model=cost_model, gates=GATES)
    assert result["recomendacion"] == "rechazar"
    assert "champion" in result["motivo"]


def test_h_limpio_permite_evaluar(roles, cost_model):
    frame, labels = _holdout()
    challenger = _package("W30_T135", roles, ids_by_role={"predictor": {1, 2, 3}})
    result = promotion_gate(challenger, None, frame, labels,
                            cost_model=cost_model, gates=GATES)
    assert result["recomendacion"] in ("activar", "rechazar")
    assert "H_contaminado" not in str(result.get("motivo", ""))


def test_un_paquete_invalido_nunca_se_promueve(roles, cost_model):
    frame, labels = _holdout()
    challenger = _package("W30_T135", roles)
    challenger.valid = False
    challenger.invalid_reason = "soporte_insuficiente"
    result = promotion_gate(challenger, None, frame, labels,
                            cost_model=cost_model, gates=GATES)
    assert result["recomendacion"] == "rechazar"
    assert result["motivo"] == "challenger_no_valido"


def test_el_gate_nunca_promueve_por_si_solo(roles, cost_model):
    """La funcion recomienda; la promocion real exige autorizacion humana."""
    frame, labels = _holdout()
    challenger = _package("W30_T135", roles, ids_by_role={"predictor": {1}})
    result = promotion_gate(challenger, None, frame, labels,
                            cost_model=cost_model, gates=GATES)
    assert result["promovido"] is False
    assert result["aprobacion_humana_requerida"] is True


def test_holdout_vacio_rechaza(roles, cost_model):
    empty = pd.DataFrame({"TransactionID": [], "TransactionAmt": [], "dia": []})
    result = promotion_gate(_package("W30_T135", roles), None, empty, np.array([]),
                            cost_model=cost_model, gates=GATES)
    assert result["recomendacion"] == "rechazar"
    assert result["motivo"] == "holdout_vacio"


def test_un_challenger_peor_conserva_al_champion(roles, cost_model):
    """Regla de no inferioridad: si el challenger cuesta mas, se conserva el champion."""
    frame, labels = _holdout(n=600, seed=3)
    # El challenger bloquea casi todo: mucha friccion, costo alto.
    challenger = _package("W30_T135", roles, base=0.95, ids_by_role={"predictor": {1}})
    champion = _package("W30_T120", roles, base=0.05, ids_by_role={"predictor": {2}})
    result = promotion_gate(challenger, champion, frame, labels,
                            cost_model=cost_model, gates=GATES)
    assert result["recomendacion"] in ("conservar_champion", "rechazar")


def test_roles_solapados_en_un_paquete_son_fuga(roles):
    package = _package("W30_T135", roles, ids_by_role={
        "predictor": {1, 2, 3}, "calibracion": {3, 4},
    })
    with pytest.raises(ValueError, match="Fuga"):
        assert_package_roles_disjoint(package)


def test_el_manifest_de_autorizacion_declara_sus_limites():
    manifest = authorization_manifest(
        "run-1", [{"tarea": "fit_W30_T120", "tipo": "ajuste_offline"}],
        authorized_by="equipo", simulated=True,
    )
    assert manifest["promociones"] == "simuladas"
    assert any("despliegue" in item for item in manifest["no_autoriza"])
    assert any("pagos" in item for item in manifest["no_autoriza"])
    assert len(manifest["tareas_autorizadas"]) == 1
