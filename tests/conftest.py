"""Fixtures compartidas: un dataset pequeno pero con la misma forma que el real.

Las pruebas usan un sustituto reducido con el mismo esquema, con dias suficientes
para construir los roles temporales y ambas clases en cada tramo, de modo que la
suite corre en segundos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_adaptive import data as data_module  # noqa: E402
from fraud_adaptive import features as features_module  # noqa: E402
from fraud_adaptive.splits import TemporalConfig  # noqa: E402
from fraud_adaptive.synthetic import SyntheticConfig, generate  # noqa: E402


@pytest.fixture(scope="session")
def temporal_config() -> TemporalConfig:
    return TemporalConfig.from_yaml(ROOT / "configs" / "temporal.yaml")


@pytest.fixture(scope="session")
def raw_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sustituto reducido: 182 dias para conservar toda la grilla temporal."""
    config = SyntheticConfig(
        scale=0.02,            # ~64 tx/dia: rapido y con ambas clases por tramo
        n_days=182,
        n_v_columns=20,        # panel estrecho: las pruebas no miden calidad predictiva
        n_cards=400,
        n_devices=200,
        base_prevalence=0.10,  # prevalencia alta para garantizar positivos por dia
        seed=7,
    )
    return generate(config)


@pytest.fixture(scope="session")
def prepared_frame(raw_frames) -> pd.DataFrame:
    """Frame integrado con features causales, como lo produce el pipeline real."""
    transactions, identity = raw_frames
    merged, _ = data_module.join_sources(transactions, identity, join_key="TransactionID")
    merged = data_module.derive_time_columns(merged, time_col="TransactionDT")
    merged = data_module.order_events(merged, time_col="TransactionDT", join_key="TransactionID")
    prepared, _ = features_module.prepare_features(merged)
    return prepared


@pytest.fixture(scope="session")
def feature_columns(prepared_frame) -> tuple[list[str], list[str]]:
    numeric, categorical = features_module.build_feature_panel(prepared_frame)
    # Se recorta el panel para que los fits de prueba sean rapidos.
    return numeric[:40], categorical[:8]


@pytest.fixture
def cost_model():
    """Economia de referencia de las pruebas.

    ``c_fp=75`` esta en el orden del valor que la calibracion produce sobre datos
    reales al exigir un 1 % de bloqueo de legitimas. Con montos tipicos hace que
    revisar sea la accion mas barata en la franja intermedia de ``p``, que es la
    condicion que ejercitan las pruebas de cupo.
    """
    from fraud_adaptive.decision import CostModel

    return CostModel(c_fp=75.0, c_review=1.0, r_h=0.90, f_h=0.02)


@pytest.fixture
def cost_model_barato():
    """Economia con friccion baja, para comprobar que entonces no se revisa."""
    from fraud_adaptive.decision import CostModel

    return CostModel(c_fp=5.0, c_review=1.0, r_h=0.90, f_h=0.02)
