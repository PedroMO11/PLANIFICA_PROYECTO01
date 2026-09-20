"""Servicio HTTP local de scoring y decision.

Contrato del servicio
---------------------
El endpoint CALCULA una decision; NO es la autoridad del estado. El cupo diario
vive en el ledger del replay, no aqui, y el cliente lo envia en
``capacity_context``.

Esa separacion es deliberada y su limitacion esta declarada: un cliente arbitrario
podria enviar un contexto de cupo falso. Por eso la demo exige un unico
orquestador confiable (``replay.py``) y el servicio no se publica como endpoint
abierto de decision de pagos. Hacer el cupo autoritativo dentro del servicio
requeriria estado distribuido y transaccional, que es justamente lo que el plan
deja como diseno futuro (Firestore) y no como implementacion.

Contrato de Cloud Run que se respeta
------------------------------------
* escucha en ``0.0.0.0`` y en el puerto de ``$PORT`` (8080 por defecto);
* sin estado durable dentro del contenedor;
* ``/health`` informa version y hash del paquete activo;
* sin paquete valido responde 503 (C21), nunca una decision por defecto.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pydantic import BaseModel as _BaseModel, Field as _Field

from .decision import (
    APROBAR, BLOQUEAR, REVISAR, CostModel, Policy, cost_model_from_dict,
    fallback_action, policy_from_dict,
)

LOGGER = logging.getLogger("fraud_adaptive.serving")


class PackageRegistry:
    """Mantiene el paquete activo y el anterior, ambos elegibles para rollback.

    No reconstruye ni recalibra nada al arrancar: carga artefactos ya validados.
    Recalibrar al inicio produciria un modelo distinto del que paso los gates.
    """

    def __init__(self, package_dir: str | Path | None = None):
        self.active: dict[str, Any] | None = None
        self.previous: dict[str, Any] | None = None
        self.load_error: str | None = None
        if package_dir:
            self.load(package_dir)

    def load(self, package_dir: str | Path) -> None:
        from .adaptation import ModelPackage

        try:
            loaded = ModelPackage.load_for_serving(package_dir)
            self.previous = self.active
            self.active = loaded
            self.load_error = None
            LOGGER.info("Paquete activo: %s", loaded["manifest"]["version_id"])
        except Exception as exc:  # noqa: BLE001 - el servicio debe seguir respondiendo 503
            self.load_error = "%s: %s" % (type(exc).__name__, exc)
            LOGGER.error("No se pudo cargar el paquete %s: %s", package_dir, self.load_error)

    def rollback(self) -> bool:
        """Vuelve al paquete anterior. Solo si existe uno completo y valido."""
        if self.previous is None:
            return False
        self.active, self.previous = self.previous, self.active
        LOGGER.warning("Rollback al paquete %s", self.active["manifest"]["version_id"])
        return True

    @property
    def available(self) -> bool:
        return self.active is not None


def _score_frame(package: dict[str, Any], frame: pd.DataFrame) -> tuple[float, float]:
    pipeline = package["pipeline"]
    estimator = package["estimator"]
    calibrator = package["calibrator"]
    matrix = pipeline.transform(frame)
    raw = float(estimator.predict_proba(matrix)[:, 1][0])
    return raw, float(calibrator.transform(np.array([raw]))[0])


class CapacityContext(_BaseModel):
    """Cupo restante que reporta el orquestador. El servicio no lo persiste."""

    remaining_reviews: int = _Field(..., ge=0)
    day: int = _Field(..., ge=0)


class PredictRequest(_BaseModel):
    event_id: str
    event_day: int = _Field(..., ge=0)
    amount: float = _Field(..., ge=0)
    features: dict[str, Any]
    schema_version: str
    capacity_context: CapacityContext


class PredictResponse(_BaseModel):
    event_id: str
    p_raw: float
    p_calibrated: float
    action: str
    expected_costs: dict[str, float]
    reason: str
    model_version: str
    calibrator_version: str
    policy_version: str
    timings_ms: dict[str, float]


def create_app(package_dir: str | Path | None = None, configs: dict[str, Any] | None = None):
    """Construye la app FastAPI. Se separa de ``uvicorn.run`` para poder testearla.

    Los modelos Pydantic viven a nivel de modulo, no aqui dentro. Con
    ``from __future__ import annotations`` las anotaciones son cadenas y FastAPI
    las resuelve contra los globales del modulo: una clase definida dentro de esta
    funcion no seria visible ahi, y FastAPI degradaria el cuerpo de la peticion a
    un parametro de query.
    """
    from fastapi import FastAPI, HTTPException

    configs = configs or {}
    serving_config = configs.get("serving", {})
    decision_config = configs.get("decision", {})
    schema_version = serving_config.get("service", {}).get("schema_version", "1.0.0")

    registry = PackageRegistry(package_dir or os.environ.get("PACKAGE_DIR"))

    app = FastAPI(
        title="fraud-adaptive serving",
        version=schema_version,
        description="Scoring y decision locales. El cupo lo gobierna el replay, no este servicio.",
    )
    app.state.registry = registry

    @app.get("/health")
    def health() -> dict[str, Any]:
        if not registry.available:
            return {
                "status": "model_unavailable",
                "detalle": registry.load_error or "sin paquete cargado",
                "schema_version": schema_version,
            }
        manifest = registry.active["manifest"]
        return {
            "status": "ok",
            "schema_version": schema_version,
            "model_version": manifest["version_id"],
            "package_hash": manifest.get("package_hash"),
            "strategy": manifest.get("strategy"),
            "policy": manifest.get("policy"),
            "previous_version": (
                registry.previous["manifest"]["version_id"] if registry.previous else None
            ),
            "nota": "El cupo de revision no es autoritativo en este servicio.",
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> Any:
        # C21: sin paquete valido no se decide un pago. 503, no una accion por defecto.
        if not registry.available:
            raise HTTPException(
                status_code=503,
                detail={"error": "model_unavailable", "motivo": registry.load_error or "sin paquete"},
            )
        if request.schema_version != schema_version:
            raise HTTPException(
                status_code=400,
                detail={"error": "schema_version_incompatible",
                        "esperado": schema_version, "recibido": request.schema_version},
            )
        # El servicio nunca debe recibir la etiqueta.
        if "isFraud" in request.features:
            raise HTTPException(
                status_code=400,
                detail={"error": "campo_prohibido", "campo": "isFraud",
                        "motivo": "el servicio no puede recibir la etiqueta"},
            )

        started = time.perf_counter()
        manifest = registry.active["manifest"]
        schema = registry.active["pipeline"].schema()

        expected = list(schema["numeric_columns"]) + list(schema["categorical_columns"])
        row = {column: request.features.get(column, None) for column in expected}
        frame = pd.DataFrame([row])
        t_features = time.perf_counter()

        raw, calibrated = _score_frame(registry.active, frame)
        t_inference = time.perf_counter()

        # Politica y economia salen del manifiesto del paquete. El servicio no
        # tiene una economia propia: decide con la que se valido en el backtest.
        policy = policy_from_dict(manifest["policy"])
        cost_model = cost_model_from_dict(manifest["costos"])

        probability = np.array([calibrated])
        amount = np.array([request.amount])
        expected_costs = cost_model.expected_costs(probability, amount)
        # La regla real, que usa el monto y el precio del cupo. `zone` solo mira `p`
        # y con la politica vigente mandaria todo a revision.
        proposed = str(policy.propose(probability, expected_costs)[0])

        action, reason = proposed, "%s_%s" % (policy.rule, proposed)
        if proposed == REVISAR:
            if request.capacity_context.remaining_reviews > 0:
                reason = "revision_propuesta_cupo_disponible"
            else:
                action = fallback_action(expected_costs, 0)
                reason = "overflow_cupo_menor_costo_esperado"

        t_decision = time.perf_counter()
        return {
            "event_id": request.event_id,
            "p_raw": raw,
            "p_calibrated": calibrated,
            "action": action,
            "expected_costs": {k: float(v[0]) for k, v in expected_costs.items()},
            "reason": reason,
            "model_version": manifest["version_id"],
            "calibrator_version": manifest["version_id"] + "_platt",
            "policy_version": policy.version,
            "timings_ms": {
                "features": (t_features - started) * 1000,
                "inference": (t_inference - t_features) * 1000,
                "decision": (t_decision - t_inference) * 1000,
                "total": (t_decision - started) * 1000,
            },
        }

    return app
