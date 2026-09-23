"""Preprocesamiento por fit y las tres familias de modelos.

``FeaturePipeline.fit`` recibe solo las filas del tramo de fit de su version. Las
medianas de imputacion, la media y desviacion del escalado, el vocabulario de las
categoricas y las columnas descartadas por faltantes se calculan ahi.

Si el vocabulario one-hot se ajustara sobre todo el dataset, el modelo reservaria
columnas para categorias que solo aparecen en test. ``tests/test_temporal_integrity.py``
comprueba que modificar filas futuras no cambia las transformaciones.

Familias: regresion logistica como referencia lineal, Random Forest como modelo no
lineal por bagging y LightGBM como modelo avanzado por boosting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger("fraud_adaptive.models")

UNKNOWN_TOKEN = "__unknown__"
MISSING_TOKEN = "__missing__"


# --------------------------------------------------------------------------- preprocesamiento

@dataclass
class FeaturePipeline:
    """Transformacion ajustada dentro de un unico tramo de fit.

    ``family`` decide la estrategia:

    * ``logistic``  -> imputacion por mediana + indicadores + escalado + one-hot disperso
    * ``random_forest`` -> imputacion por mediana + indicadores + one-hot denso
    * ``lightgbm``  -> sin imputacion ni escalado; faltantes y categorias nativos
    """

    family: str
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    vocabularies: dict[str, list[str]] = field(default_factory=dict)
    indicator_columns: list[str] = field(default_factory=list)
    min_frequency: int = 10
    max_categories: int = 64
    missing_threshold: float = 0.95
    fitted: bool = False
    n_fit_rows: int = 0

    # -- ajuste

    def fit(
        self,
        frame: pd.DataFrame,
        numeric_columns: Sequence[str],
        categorical_columns: Sequence[str],
    ) -> "FeaturePipeline":
        self.n_fit_rows = len(frame)
        if self.n_fit_rows == 0:
            raise ValueError("No se puede ajustar el preprocesamiento sobre un tramo vacio")

        # Las columnas se descartan segun sus faltantes en el fit, aunque esten
        # pobladas en test, porque esta version no pudo aprender de ellas.
        numeric_keep: list[str] = []
        self.dropped_columns = []
        for column in numeric_columns:
            missing_rate = float(frame[column].isna().mean())
            if missing_rate > self.missing_threshold:
                self.dropped_columns.append(column)
            else:
                numeric_keep.append(column)
        self.numeric_columns = numeric_keep
        self.categorical_columns = list(categorical_columns)

        if self.family in ("logistic", "random_forest"):
            for column in self.numeric_columns:
                series = pd.to_numeric(frame[column], errors="coerce")
                median = float(series.median()) if series.notna().any() else 0.0
                self.medians[column] = median
                if series.isna().any():
                    self.indicator_columns.append(column)

        if self.family == "logistic":
            for column in self.numeric_columns:
                series = pd.to_numeric(frame[column], errors="coerce").fillna(self.medians[column])
                mean = float(series.mean())
                std = float(series.std())
                self.means[column] = mean
                # Una columna constante tiene std 0; con 1 queda en cero tras centrar.
                self.stds[column] = std if np.isfinite(std) and std > 1e-12 else 1.0

        for column in self.categorical_columns:
            counts = frame[column].astype("string").fillna(MISSING_TOKEN).value_counts()
            frequent = counts[counts >= self.min_frequency]
            vocabulary = list(frequent.index[: self.max_categories])
            if MISSING_TOKEN not in vocabulary:
                vocabulary.append(MISSING_TOKEN)
            # UNKNOWN recibe en test las categorias no vistas o poco frecuentes en
            # el fit, de modo que el ancho de la matriz no cambia.
            vocabulary.append(UNKNOWN_TOKEN)
            self.vocabularies[column] = vocabulary

        self.fitted = True
        LOGGER.debug(
            "Pipeline %s ajustado: %d numericas, %d categoricas, %d descartadas",
            self.family, len(self.numeric_columns), len(self.categorical_columns), len(self.dropped_columns),
        )
        return self

    # -- transformacion

    def transform(self, frame: pd.DataFrame) -> Any:
        if not self.fitted:
            raise RuntimeError("El pipeline no esta ajustado")
        if self.family == "lightgbm":
            return self._transform_native(frame)
        return self._transform_matrix(frame)

    def _transform_native(self, frame: pd.DataFrame) -> pd.DataFrame:
        """LightGBM: faltantes nativos y categoricas como dtype category.

        Las categorias se fijan al vocabulario del fit, y una categoria nueva en test
        se asigna al token desconocido.
        """
        # El DataFrame se construye de una vez para no fragmentarlo con cientos de
        # asignaciones de columna.
        columns: dict[str, Any] = {}
        for column in self.numeric_columns:
            columns[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
        for column in self.categorical_columns:
            vocabulary = self.vocabularies[column]
            values = frame[column].astype("string").fillna(MISSING_TOKEN)
            values = values.where(values.isin(vocabulary), UNKNOWN_TOKEN)
            columns[column] = pd.Categorical(values, categories=vocabulary)
        return pd.DataFrame(columns, index=frame.index)

    def _transform_matrix(self, frame: pd.DataFrame) -> Any:
        blocks: list[Any] = []

        numeric_block = np.empty((len(frame), len(self.numeric_columns)), dtype="float32")
        for i, column in enumerate(self.numeric_columns):
            series = pd.to_numeric(frame[column], errors="coerce")
            filled = series.fillna(self.medians[column]).to_numpy(dtype="float32")
            if self.family == "logistic":
                filled = (filled - self.means[column]) / self.stds[column]
            # Los infinitos se reemplazan para que el solver no falle.
            numeric_block[:, i] = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
        blocks.append(numeric_block)

        if self.indicator_columns:
            indicators = np.empty((len(frame), len(self.indicator_columns)), dtype="float32")
            for i, column in enumerate(self.indicator_columns):
                indicators[:, i] = frame[column].isna().to_numpy(dtype="float32")
            blocks.append(indicators)

        for column in self.categorical_columns:
            vocabulary = self.vocabularies[column]
            index = {token: i for i, token in enumerate(vocabulary)}
            values = frame[column].astype("string").fillna(MISSING_TOKEN)
            positions = values.map(lambda v: index.get(v, index[UNKNOWN_TOKEN])).to_numpy(dtype="int32")
            onehot = np.zeros((len(frame), len(vocabulary)), dtype="float32")
            onehot[np.arange(len(frame)), positions] = 1.0
            blocks.append(onehot)

        dense = np.hstack(blocks) if blocks else np.empty((len(frame), 0), dtype="float32")
        # La logistica usa saga sobre una matriz dispersa, que con cientos de
        # columnas one-hot reduce la memoria en un orden de magnitud.
        return sparse.csr_matrix(dense) if self.family == "logistic" else dense

    # -- metadatos

    def feature_names(self) -> list[str]:
        if self.family == "lightgbm":
            return list(self.numeric_columns) + list(self.categorical_columns)
        names = list(self.numeric_columns)
        names += ["%s__isna" % c for c in self.indicator_columns]
        for column in self.categorical_columns:
            names += ["%s__%s" % (column, token) for token in self.vocabularies[column]]
        return names

    def schema(self) -> dict[str, Any]:
        """Esquema serializable, que el servicio usa para validar el payload."""
        return {
            "family": self.family,
            "n_fit_rows": self.n_fit_rows,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "dropped_columns": self.dropped_columns,
            "indicator_columns": self.indicator_columns,
            "n_features": len(self.feature_names()),
            "vocabulary_sizes": {c: len(v) for c, v in self.vocabularies.items()},
            "min_frequency": self.min_frequency,
            "max_categories": self.max_categories,
            "missing_threshold": self.missing_threshold,
        }


# --------------------------------------------------------------------------- modelos

def _lightgbm_classifier(**kwargs: Any):
    from lightgbm import LGBMClassifier  # import local: LightGBM tarda en cargar

    return LGBMClassifier(**kwargs)


def build_estimator(family: str, config: dict[str, Any], base: dict[str, Any], *, seed: int = 42,
                    n_threads: int = 4, y_fit: np.ndarray | None = None) -> Any:
    """Instancia un estimador con la configuracion de configs/models.yaml.

    Los pesos de clase se derivan de ``y_fit``, para no usar la prevalencia de
    periodos futuros.
    """
    params = {k: v for k, v in config.items() if k != "name"}

    if family == "logistic":
        return LogisticRegression(
            C=params.get("C", 1.0),
            class_weight=params.get("class_weight"),
            solver=base.get("solver", "saga"),
            max_iter=base.get("max_iter", 300),
            tol=base.get("tol", 1e-3),
            random_state=seed,
            n_jobs=n_threads,
        )

    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth"),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            class_weight=params.get("class_weight"),
            bootstrap=base.get("bootstrap", True),
            random_state=seed,
            n_jobs=n_threads,
        )

    if family == "lightgbm":
        scale_pos_weight = params.get("scale_pos_weight")
        if scale_pos_weight == "auto":
            if y_fit is None:
                raise ValueError("scale_pos_weight=auto requiere y_fit")
            n_positive = int((y_fit == 1).sum())
            n_negative = int((y_fit == 0).sum())
            scale_pos_weight = (n_negative / n_positive) if n_positive else 1.0
        return _lightgbm_classifier(
            num_leaves=params.get("num_leaves", 31),
            learning_rate=base.get("learning_rate", 0.05),
            n_estimators=base.get("n_estimators", 300),
            min_child_samples=base.get("min_child_samples", 100),
            reg_lambda=base.get("reg_lambda", 1.0),
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=n_threads,
            verbose=-1,
        )

    raise ValueError("Familia desconocida: %s" % family)


def stratified_day_subsample(
    frame: pd.DataFrame, labels: np.ndarray, max_rows: int, *, seed: int = 42
) -> np.ndarray:
    """Submuestreo estratificado por dia y clase para acotar el costo de Random Forest.

    Preserva la distribucion temporal del tramo y la prevalencia; un muestreo
    uniforme podria dejar dias sin filas. Devuelve posiciones enteras y se aplica
    solo al fit.
    """
    n_rows = len(frame)
    if n_rows <= max_rows:
        return np.arange(n_rows)

    rng = np.random.default_rng(seed)
    fraction = max_rows / n_rows
    days = frame["dia"].to_numpy()
    selected: list[np.ndarray] = []

    for day in np.unique(days):
        for label in (0, 1):
            positions = np.where((days == day) & (labels == label))[0]
            if positions.size == 0:
                continue
            # Al menos una fila por estrato, para conservar los dias con pocos fraudes.
            take = max(1, int(round(positions.size * fraction)))
            take = min(take, positions.size)
            selected.append(rng.choice(positions, size=take, replace=False))

    out = np.sort(np.concatenate(selected)) if selected else np.arange(0)
    LOGGER.info("Submuestreo RF: %d -> %d filas (estratificado por dia y clase)", n_rows, out.size)
    return out


@dataclass
class FittedModel:
    """Predictor, su preprocesamiento y los datos con que se ajusto."""

    family: str
    config_name: str
    pipeline: FeaturePipeline
    estimator: Any
    fit_rows: int
    fit_positives: int
    fit_ids: set[Any] = field(default_factory=set)
    seed: int = 42
    subsampled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.pipeline.transform(frame)
        return self.estimator.predict_proba(matrix)[:, 1]

    def describe(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "config": self.config_name,
            "fit_rows": self.fit_rows,
            "fit_positives": self.fit_positives,
            "fit_prevalence": (self.fit_positives / self.fit_rows) if self.fit_rows else None,
            "subsampled": self.subsampled,
            "seed": self.seed,
            "schema": self.pipeline.schema(),
            **self.metadata,
        }


def fit_model(
    family: str,
    config: dict[str, Any],
    base: dict[str, Any],
    frame: pd.DataFrame,
    labels: np.ndarray,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    seed: int = 42,
    n_threads: int = 4,
    max_rows: int | None = None,
    id_column: str = "TransactionID",
    encoding: dict[str, Any] | None = None,
) -> FittedModel:
    """Ajusta preprocesamiento y estimador sobre un unico tramo.

    El submuestreo se decide primero y el preprocesamiento se ajusta sobre las filas
    que efectivamente entrenan el modelo.
    """
    encoding = encoding or {}
    labels = np.asarray(labels)

    positions = np.arange(len(frame))
    subsampled = False
    if max_rows is not None and len(frame) > max_rows:
        positions = stratified_day_subsample(frame, labels, max_rows, seed=seed)
        subsampled = True

    fit_frame = frame.iloc[positions]
    fit_labels = labels[positions]

    if len(np.unique(fit_labels)) < 2:
        raise ValueError(
            "El tramo de fit tiene una sola clase (%d filas): no se puede entrenar"
            % len(fit_labels)
        )

    pipeline = FeaturePipeline(
        family=family,
        min_frequency=encoding.get("onehot_min_frequency", 10),
        max_categories=encoding.get("onehot_max_categories", 64),
        missing_threshold=encoding.get("drop_column_missing_above", 0.95),
    )
    pipeline.fit(fit_frame, numeric_columns, categorical_columns)

    estimator = build_estimator(family, config, base, seed=seed, n_threads=n_threads, y_fit=fit_labels)
    matrix = pipeline.transform(fit_frame)

    if family == "lightgbm":
        estimator.fit(matrix, fit_labels, categorical_feature=list(pipeline.categorical_columns))
    else:
        estimator.fit(matrix, fit_labels)

    fit_ids = set(fit_frame[id_column].tolist()) if id_column in fit_frame.columns else set()

    return FittedModel(
        family=family,
        config_name=config.get("name", "sin_nombre"),
        pipeline=pipeline,
        estimator=estimator,
        fit_rows=int(len(fit_frame)),
        fit_positives=int(fit_labels.sum()),
        fit_ids=fit_ids,
        seed=seed,
        subsampled=subsampled,
        metadata={
            "n_rows_disponibles": int(len(frame)),
            "dia_min_fit": int(fit_frame["dia"].min()) if "dia" in fit_frame.columns else None,
            "dia_max_fit": int(fit_frame["dia"].max()) if "dia" in fit_frame.columns else None,
        },
    )


def iter_family_configs(models_config: dict[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Aplana el YAML a ``(familia, config, base)`` preservando el orden declarado."""
    out: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for family, spec in models_config["families"].items():
        base = spec.get("base", {})
        for config in spec["configs"]:
            out.append((family, config, base))
    return out
