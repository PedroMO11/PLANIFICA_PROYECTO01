"""Contrato del servicio HTTP.

Se comprueba lo que el servicio debe RECHAZAR tanto como lo que debe responder:
sin paquete valido no decide, no acepta la etiqueta y no acepta un esquema que no
sea el suyo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_adaptive.calibration import PlattCalibrator
from fraud_adaptive.models import FeaturePipeline
from fraud_adaptive.serving import PackageRegistry, create_app

SCHEMA_VERSION = "1.0.0"

# El servicio ya no lee la economia del config: la toma del manifiesto del
# paquete, que es lo que garantiza que decide igual que el backtest.
COSTOS = {"c_fp": 75.0, "c_review": 1.0, "r_h": 0.90, "f_h": 0.02}

CONFIGS = {
    "serving": {"service": {"schema_version": SCHEMA_VERSION}},
    "decision": {"analyst": {"r_h": 0.90, "f_h": 0.02}},
}


class _Estimator:
    """Predictor de prueba: probabilidad creciente con la primera columna."""

    def predict_proba(self, matrix) -> np.ndarray:
        values = np.asarray(matrix)[:, 0] if np.asarray(matrix).ndim == 2 else np.asarray(matrix)
        probability = np.clip(np.asarray(values, dtype=float) / 1000.0, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - probability, probability])


@pytest.fixture
def package_dir(tmp_path) -> Path:
    """Escribe en disco un paquete minimo pero completo."""
    import joblib

    frame = pd.DataFrame({
        "TransactionAmt": [10.0, 100.0, 900.0, 50.0],
        "monto_log": [2.4, 4.6, 6.8, 3.9],
        "ProductCD": ["W", "C", "W", "R"],
    })
    pipeline = FeaturePipeline(family="random_forest").fit(
        frame, ["TransactionAmt", "monto_log"], ["ProductCD"]
    )

    directory = tmp_path / "W30_T120"
    directory.mkdir(parents=True)
    joblib.dump(_Estimator(), directory / "predictor.joblib")
    joblib.dump(pipeline, directory / "preprocessor.joblib")
    joblib.dump(PlattCalibrator(), directory / "calibrator.joblib")
    (directory / "feature_schema.json").write_text(
        json.dumps(pipeline.schema()), encoding="utf-8"
    )
    policy = {"rule": "argmin", "tau_low": 0.0, "tau_high": 1.0, "daily_capacity": 150,
              "delta": 0.0, "version": "politica_v2"}
    (directory / "policy.json").write_text(
        json.dumps({**policy, "costos": COSTOS}), encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps({
        "version_id": "W30_T120", "strategy": "W30", "policy": policy,
        "costos": COSTOS, "package_hash": "abc123",
    }), encoding="utf-8")
    return directory


@pytest.fixture
def client(package_dir) -> TestClient:
    return TestClient(create_app(package_dir=package_dir, configs=CONFIGS))


def _payload(**overrides) -> dict:
    payload = {
        "event_id": "evt-1",
        "event_day": 120,
        "amount": 100.0,
        "features": {"TransactionAmt": 100.0, "monto_log": 4.6, "ProductCD": "W"},
        "schema_version": SCHEMA_VERSION,
        "capacity_context": {"remaining_reviews": 10, "day": 120},
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- health

def test_health_informa_version_y_hash(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "W30_T120"
    assert body["package_hash"] == "abc123"
    assert "no es autoritativo" in body["nota"]


def test_sin_paquete_health_reporta_indisponible():
    client = TestClient(create_app(package_dir=None, configs=CONFIGS))
    assert client.get("/health").json()["status"] == "model_unavailable"


# --------------------------------------------------------------------------- predict

def test_predict_devuelve_el_contrato_completo(client):
    body = client.post("/predict", json=_payload()).json()
    for field in ("p_raw", "p_calibrated", "action", "expected_costs", "reason",
                  "model_version", "calibrator_version", "policy_version", "timings_ms"):
        assert field in body
    assert body["action"] in ("aprobar", "revisar", "bloquear")
    assert set(body["expected_costs"]) == {"aprobar", "revisar", "bloquear"}
    assert 0.0 <= body["p_calibrated"] <= 1.0


def test_sin_paquete_valido_responde_503():
    """C21: nunca se decide un pago sin modelo. 503, no una accion por defecto."""
    client = TestClient(create_app(package_dir=None, configs=CONFIGS))
    response = client.post("/predict", json=_payload())
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "model_unavailable"


def test_el_servicio_rechaza_la_etiqueta(client):
    """Recibir isFraud seria una fuga por la interfaz."""
    response = client.post("/predict", json=_payload(
        features={"TransactionAmt": 100.0, "monto_log": 4.6, "ProductCD": "W", "isFraud": 1}
    ))
    assert response.status_code == 400
    assert response.json()["detail"]["campo"] == "isFraud"


def test_esquema_incompatible_es_rechazado(client):
    response = client.post("/predict", json=_payload(schema_version="9.9.9"))
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "schema_version_incompatible"


def test_categoria_no_vista_no_rompe_el_servicio(client):
    response = client.post("/predict", json=_payload(
        features={"TransactionAmt": 100.0, "monto_log": 4.6, "ProductCD": "JAMAS_VISTA"}
    ))
    assert response.status_code == 200
    assert response.json()["action"] in ("aprobar", "revisar", "bloquear")


def test_feature_faltante_se_imputa_sin_fallar(client):
    response = client.post("/predict", json=_payload(features={"TransactionAmt": 100.0}))
    assert response.status_code == 200


def test_sin_cupo_la_zona_gris_cae_a_accion_automatica(client):
    """Con p en zona gris y cupo agotado, la respuesta no puede ser 'revisar'."""
    con_cupo = client.post("/predict", json=_payload(
        event_id="a", features={"TransactionAmt": 500.0, "monto_log": 6.2, "ProductCD": "W"},
        capacity_context={"remaining_reviews": 10, "day": 120},
    )).json()
    sin_cupo = client.post("/predict", json=_payload(
        event_id="b", features={"TransactionAmt": 500.0, "monto_log": 6.2, "ProductCD": "W"},
        capacity_context={"remaining_reviews": 0, "day": 120},
    )).json()

    if con_cupo["action"] == "revisar":
        assert sin_cupo["action"] in ("aprobar", "bloquear")
        assert sin_cupo["reason"] == "overflow_cupo_menor_costo_esperado"


def test_monto_negativo_es_rechazado(client):
    assert client.post("/predict", json=_payload(amount=-5.0)).status_code == 422


def test_cupo_negativo_es_rechazado(client):
    response = client.post("/predict", json=_payload(
        capacity_context={"remaining_reviews": -1, "day": 120}
    ))
    assert response.status_code == 422


def test_paridad_entre_el_servicio_y_el_calculo_offline(client, package_dir):
    """La misma fila debe dar la misma p por HTTP que en proceso.

    Si divergieran, las metricas del backtest no describirian lo que el servicio
    hace en la demo.
    """
    import joblib

    pipeline = joblib.load(package_dir / "preprocessor.joblib")
    estimator = joblib.load(package_dir / "predictor.joblib")
    calibrator = joblib.load(package_dir / "calibrator.joblib")

    features = {"TransactionAmt": 250.0, "monto_log": 5.5, "ProductCD": "C"}
    frame = pd.DataFrame([features])
    offline_raw = float(estimator.predict_proba(pipeline.transform(frame))[:, 1][0])
    offline_calibrated = float(calibrator.transform(np.array([offline_raw]))[0])

    body = client.post("/predict", json=_payload(features=features, amount=250.0)).json()
    assert body["p_raw"] == pytest.approx(offline_raw, rel=1e-9)
    assert body["p_calibrated"] == pytest.approx(offline_calibrated, rel=1e-9)

    # La paridad de probabilidad no basta. El servicio debe emitir la MISMA accion
    # que el calculo offline: comparar solo `p` dejo pasar que el servicio decidiera
    # por umbrales mientras el backtest decidia por costo esperado.
    from fraud_adaptive.decision import CostModel, Policy, policy_from_dict

    policy = policy_from_dict(
        json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))["policy"])
    costs = CostModel(**COSTOS).expected_costs(
        np.array([offline_calibrated]), np.array([250.0]))
    assert body["action"] == str(policy.propose(np.array([offline_calibrated]), costs)[0])
    for accion, valor in body["expected_costs"].items():
        assert valor == pytest.approx(float(costs[accion][0]), rel=1e-9)


def test_el_servicio_usa_el_precio_del_cupo(client, package_dir):
    """Con `tau_low=0` y `tau_high=1`, una regla de umbrales mandaria todo a
    revision. La accion debe depender del monto y del precio del cupo."""
    barata = client.post("/predict", json=_payload(
        features={"TransactionAmt": 1.0, "monto_log": 0.7, "ProductCD": "C"},
        amount=1.0)).json()
    cara = client.post("/predict", json=_payload(
        features={"TransactionAmt": 5000.0, "monto_log": 8.5, "ProductCD": "C"},
        amount=5000.0)).json()
    assert barata["action"] == "aprobar", barata
    assert barata["action"] != cara["action"], (barata, cara)


# --------------------------------------------------------------------------- rollback

def test_el_registro_conserva_el_paquete_anterior(package_dir):
    registry = PackageRegistry(package_dir)
    assert registry.available and registry.previous is None
    # Sin version anterior no hay rollback posible.
    assert registry.rollback() is False

    registry.load(package_dir)
    assert registry.previous is not None
    assert registry.rollback() is True


def test_un_paquete_corrupto_no_tumba_el_servicio(tmp_path):
    broken = tmp_path / "roto"
    broken.mkdir()
    (broken / "manifest.json").write_text("{ esto no es json", encoding="utf-8")
    registry = PackageRegistry(broken)
    assert not registry.available
    assert registry.load_error is not None
