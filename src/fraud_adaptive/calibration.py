"""Calibracion de Platt: un calibrador propio por version de paquete.

La politica compara costos esperados como ``p*monto`` y ``(1-p)*c_FP``, lo que
exige que ``p`` sea una probabilidad. Un modelo entrenado con ``class_weight`` o
``scale_pos_weight`` produce puntajes inflados que harian bloquear de mas sin que
el AP lo refleje.

La cola de calibracion cubre 7 dias, con unos 50 a 300 fraudes. Con ese soporte la
calibracion isotonica produce escalones inestables entre versiones; Platt tiene
dos parametros y es estable. La eleccion se fijo antes del test.

El ajuste se hace sobre ``logit(p)``: la transformacion es monotona y un modelo ya
calibrado corresponde a la identidad (a=1, b=0).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger("fraud_adaptive.calibration")

DEFAULT_CLIP = (1e-6, 1.0 - 1e-6)


def to_logit(probability: np.ndarray, clip: tuple[float, float] = DEFAULT_CLIP) -> np.ndarray:
    """logit con recorte, para que una probabilidad de 0 o 1 no produzca infinitos.

    Con el recorte por defecto el logit queda acotado a unos +/-13.8.
    """
    clipped = np.clip(np.asarray(probability, dtype=float), clip[0], clip[1])
    return np.log(clipped / (1.0 - clipped))


@dataclass
class PlattCalibrator:
    """Calibrador univariado ajustado sobre una cola temporal separada."""

    a: float = 1.0
    b: float = 0.0
    clip: tuple[float, float] = DEFAULT_CLIP
    n_fit: int = 0
    n_positives: int = 0
    fit_ids: set[Any] = field(default_factory=set)
    fitted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(
        self,
        raw_probability: np.ndarray,
        labels: np.ndarray,
        *,
        ids: set[Any] | None = None,
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
    ) -> "PlattCalibrator":
        raw_probability = np.asarray(raw_probability, dtype=float)
        labels = np.asarray(labels)

        self.n_fit = int(labels.size)
        self.n_positives = int((labels == 1).sum())
        self.fit_ids = set(ids or ())

        if len(np.unique(labels)) < 2:
            # Sin ambas clases no hay nada que estimar: se mantiene la identidad.
            LOGGER.warning("Calibracion con una sola clase (n=%d): se mantiene identidad", self.n_fit)
            self.a, self.b, self.fitted = 1.0, 0.0, False
            self.metadata["motivo_identidad"] = "una_sola_clase"
            return self

        logits = to_logit(raw_probability, self.clip).reshape(-1, 1)
        # Sin class_weight, para reproducir la prevalencia real de la cola.
        model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, class_weight=None)
        model.fit(logits, labels)

        self.a = float(model.coef_[0][0])
        self.b = float(model.intercept_[0])
        self.fitted = True
        self.metadata.update({
            "prevalencia_calibracion": float(labels.mean()),
            "a": self.a,
            "b": self.b,
        })
        LOGGER.debug("Platt ajustado: a=%.4f b=%.4f n=%d pos=%d", self.a, self.b, self.n_fit, self.n_positives)
        return self

    def transform(self, raw_probability: np.ndarray) -> np.ndarray:
        logits = to_logit(raw_probability, self.clip)
        return 1.0 / (1.0 + np.exp(-(self.a * logits + self.b)))

    __call__ = transform

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "clip": list(self.clip),
            "n_fit": self.n_fit,
            "n_positives": self.n_positives,
            "fitted": self.fitted,
            "metodo": "platt_logit_univariado",
            **self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlattCalibrator":
        calibrator = cls(
            a=float(data["a"]),
            b=float(data["b"]),
            clip=tuple(data.get("clip", DEFAULT_CLIP)),
            n_fit=int(data.get("n_fit", 0)),
            n_positives=int(data.get("n_positives", 0)),
        )
        calibrator.fitted = bool(data.get("fitted", True))
        return calibrator


# --------------------------------------------------------------------------- diagnostico

def brier_score(probability: np.ndarray, labels: np.ndarray) -> float:
    """Error cuadratico medio de la probabilidad. Menor es mejor."""
    probability = np.asarray(probability, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probability.size == 0:
        return float("nan")
    return float(np.mean((probability - labels) ** 2))


def reliability_curve(
    probability: np.ndarray, labels: np.ndarray, *, n_bins: int = 10, strategy: str = "quantile"
) -> dict[str, list[float]]:
    """Curva de confiabilidad: probabilidad predicha frente a frecuencia observada.

    Con una prevalencia de 3.5% los bins uniformes concentran casi todo en el
    primero, por eso se usan cuantiles y se reporta el soporte de cada bin.
    """
    probability = np.asarray(probability, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probability.size == 0:
        return {"pred": [], "obs": [], "n": [], "edges": []}

    if strategy == "quantile":
        edges = np.unique(np.quantile(probability, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    if edges.size < 2:
        edges = np.array([0.0, 1.0])

    indices = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, edges.size - 2)
    pred, obs, counts = [], [], []
    for b in range(edges.size - 1):
        mask = indices == b
        n = int(mask.sum())
        counts.append(n)
        pred.append(float(probability[mask].mean()) if n else float("nan"))
        obs.append(float(labels[mask].mean()) if n else float("nan"))
    return {"pred": pred, "obs": obs, "n": counts, "edges": [float(e) for e in edges]}


def expected_calibration_error(
    probability: np.ndarray, labels: np.ndarray, *, n_bins: int = 10, strategy: str = "quantile"
) -> float:
    """ECE ponderado por soporte.

    Depende de la eleccion de bins y se usa como complemento del Brier.
    """
    curve = reliability_curve(probability, labels, n_bins=n_bins, strategy=strategy)
    total = sum(curve["n"])
    if total == 0:
        return float("nan")
    error = 0.0
    for pred, obs, n in zip(curve["pred"], curve["obs"], curve["n"]):
        if n and np.isfinite(pred) and np.isfinite(obs):
            error += (n / total) * abs(pred - obs)
    return float(error)


def calibration_report(
    raw_probability: np.ndarray,
    calibrated_probability: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Metricas de calibracion antes y despues de aplicar Platt."""
    return {
        "n": int(np.asarray(labels).size),
        "prevalencia": float(np.mean(labels)) if np.size(labels) else float("nan"),
        "brier_crudo": brier_score(raw_probability, labels),
        "brier_calibrado": brier_score(calibrated_probability, labels),
        "ece_crudo": expected_calibration_error(raw_probability, labels, n_bins=n_bins),
        "ece_calibrado": expected_calibration_error(calibrated_probability, labels, n_bins=n_bins),
        "p_media_cruda": float(np.mean(raw_probability)) if np.size(raw_probability) else float("nan"),
        "p_media_calibrada": float(np.mean(calibrated_probability)) if np.size(calibrated_probability) else float("nan"),
        "curva_calibrada": reliability_curve(calibrated_probability, labels, n_bins=n_bins),
    }
