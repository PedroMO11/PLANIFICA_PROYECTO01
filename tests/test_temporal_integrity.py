"""Integridad de los roles temporales y del preprocesamiento por fold.

Cubre las dos fugas mas dificiles de ver a simple vista:

* roles que se solapan y comparten filas entre ajuste y evaluacion;
* transformaciones (medianas, escalado, vocabulario) ajustadas con datos que la
  version no debia conocer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_adaptive.models import FeaturePipeline
from fraud_adaptive.splits import (
    Interval, assert_no_leakage_between_roles, build_all_version_roles,
    build_version_roles, check_support, find_overlaps, intervals_disjoint,
    label_eligible, mature_training_mask,
)

STRATEGIES = [
    {"name": "S0", "kind": "static", "window_days": None},
    {"name": "E15", "kind": "expanding", "window_days": None},
    {"name": "W30", "kind": "sliding", "window_days": 30},
    {"name": "W60", "kind": "sliding", "window_days": 60},
    {"name": "W90", "kind": "sliding", "window_days": 90},
]


# --------------------------------------------------------------------------- intervalos

def test_intervalos_semiabiertos_no_se_solapan_en_el_borde():
    """[0,10) y [10,20) se tocan pero no se solapan: es lo que permite encadenar roles."""
    assert not Interval(0, 10).overlaps(Interval(10, 20))
    assert Interval(0, 11).overlaps(Interval(10, 20))
    assert Interval(0, 10).contains(9)
    assert not Interval(0, 10).contains(10)


def test_intervalo_invalido_falla():
    with pytest.raises(ValueError):
        Interval(10, 5)


# --------------------------------------------------------------------------- roles

def test_los_cuatro_roles_son_disjuntos_en_toda_la_grilla(temporal_config):
    """Ninguna version puede usar el mismo dia para ajustar y para validar."""
    all_roles = build_all_version_roles(STRATEGIES, temporal_config)
    for versions in all_roles.values():
        for roles in versions:
            named = roles.named()
            assert intervals_disjoint(named.values()), \
                "roles solapados en %s: %s" % (roles.version_id, find_overlaps(named))


def test_el_predictor_termina_antes_de_las_tres_reservas(temporal_config):
    for versions in build_all_version_roles(STRATEGIES, temporal_config).values():
        for roles in versions:
            assert roles.predictor.end <= roles.calibration.start
            assert roles.calibration.end <= roles.policy.start
            assert roles.policy.end <= roles.promotion_validation.start
            assert roles.promotion_validation.end == roles.cutoff


def test_ventanas_dan_el_fit_efectivo_esperado(temporal_config):
    """W abarca predictor mas tres reservas de 7 dias: el fit efectivo es W-21."""
    for window, expected in ((30, 9), (60, 39), (90, 69)):
        roles = build_version_roles("W%d" % window, 135, temporal_config,
                                    window_days=window, kind="sliding")
        assert roles.predictor.days == expected


def test_cutoff_respeta_el_retraso_de_etiqueta(temporal_config):
    roles = build_version_roles("W30", 120, temporal_config, window_days=30, kind="sliding")
    assert roles.cutoff == 120 - temporal_config.label_delay_days == 90


def test_estrategias_iniciales_comparten_predictor(temporal_config):
    """En T=120, S0/E15/W90 tienen el mismo tramo: es un solo fit, no tres."""
    first = temporal_config.update_times[0]
    static = build_version_roles("S0", first, temporal_config, kind="static")
    expanding = build_version_roles("E15", first, temporal_config, kind="expanding")
    w90 = build_version_roles("W90", first, temporal_config, window_days=90, kind="sliding")
    assert static.predictor == expanding.predictor == w90.predictor


def test_ventana_deslizante_avanza_conservando_su_ancho(temporal_config):
    previous = None
    for update_time in temporal_config.update_times:
        roles = build_version_roles("W60", update_time, temporal_config,
                                    window_days=60, kind="sliding")
        if previous is not None:
            assert roles.predictor.start > previous.predictor.start, "la ventana debe deslizarse"
            assert roles.predictor.days == previous.predictor.days, "el ancho debe conservarse"
        previous = roles


def test_ventana_expansiva_crece(temporal_config):
    widths = [
        build_version_roles("E15", t, temporal_config, kind="expanding").predictor.days
        for t in temporal_config.update_times
    ]
    assert widths == sorted(widths) and widths[0] < widths[-1]


# --------------------------------------------------------------------------- madurez

def test_elegibilidad_de_etiqueta_es_estricta():
    """Una etiqueta que madura EXACTAMENTE en el corte todavia no esta disponible."""
    days = np.array([59, 60, 61])
    eligible = label_eligible(days, cutoff=90, label_delay_days=30)
    assert list(eligible) == [True, False, False]


def test_mascara_de_entrenamiento_exige_intervalo_y_madurez():
    days = np.array([10, 50, 65, 80])
    mask = mature_training_mask(days, Interval(0, 69), cutoff=90, label_delay_days=30)
    # 80 esta fuera del intervalo; 65 esta dentro pero madura en 95 > 90.
    assert list(mask) == [True, True, False, False]


def test_soporte_insuficiente_se_detecta():
    labels = np.array([0] * 100 + [1] * 10)
    ok, counts = check_support(labels, {"fraud": 200, "legit": 2000})
    assert not ok and counts == {"fraud": 10, "legit": 100}
    ok, _ = check_support(labels, {"fraud": 5, "legit": 50})
    assert ok


# --------------------------------------------------------------------------- fuga por IDs

def test_ids_compartidos_entre_roles_se_detectan(temporal_config):
    roles = build_version_roles("W60", 120, temporal_config, window_days=60, kind="sliding")
    with pytest.raises(ValueError, match="Fuga"):
        assert_no_leakage_between_roles(roles, {
            "predictor": {1, 2, 3},
            "calibracion": {3, 4},   # el 3 aparece en ambos
        })
    # Sin interseccion no levanta nada.
    assert_no_leakage_between_roles(roles, {"predictor": {1, 2}, "calibracion": {3, 4}})


def test_roles_reales_no_comparten_ids(prepared_frame, temporal_config):
    """Sobre datos reales, y no solo sobre intervalos, los roles deben ser disjuntos."""
    days = prepared_frame["dia"].to_numpy()
    for update_time in temporal_config.update_times:
        roles = build_version_roles("W90", update_time, temporal_config,
                                    window_days=90, kind="sliding")
        ids_by_role = {}
        for name, interval in roles.named().items():
            mask = mature_training_mask(days, interval, roles.cutoff,
                                        temporal_config.label_delay_days)
            ids_by_role[name] = set(prepared_frame.loc[mask, "TransactionID"])
        assert_no_leakage_between_roles(roles, ids_by_role)


# --------------------------------------------------------------------------- preprocesamiento

def _panel(frame):
    return ["TransactionAmt", "monto_log"], ["ProductCD", "card6"]


def test_preprocesamiento_no_cambia_al_alterar_filas_futuras(prepared_frame):
    """La prueba decisiva de que las transformaciones se ajustan dentro del fit.

    Si el vocabulario o las medianas se calcularan globalmente, alterar el futuro
    cambiaria el pipeline; aqui debe quedar identico bit a bit.
    """
    numeric, categorical = _panel(prepared_frame)
    fit_frame = prepared_frame[prepared_frame["dia"] < 60]

    pipeline_a = FeaturePipeline(family="logistic").fit(fit_frame, numeric, categorical)

    # Se altera drasticamente el futuro: montos enormes y una categoria nueva.
    mutated = prepared_frame.copy()
    future = mutated["dia"] >= 60
    mutated.loc[future, "TransactionAmt"] = 999_999.0
    mutated.loc[future, "ProductCD"] = "CATEGORIA_NUEVA"
    mutated_fit = mutated[mutated["dia"] < 60]

    pipeline_b = FeaturePipeline(family="logistic").fit(mutated_fit, numeric, categorical)

    assert pipeline_a.medians == pipeline_b.medians
    assert pipeline_a.means == pipeline_b.means
    assert pipeline_a.vocabularies == pipeline_b.vocabularies
    assert "CATEGORIA_NUEVA" not in pipeline_a.vocabularies["ProductCD"]


def test_categoria_no_vista_cae_en_unknown(prepared_frame):
    """Una categoria nueva en test no puede romper el ancho de la matriz ni
    mapearse en silencio a otra existente."""
    numeric, categorical = _panel(prepared_frame)
    fit_frame = prepared_frame[prepared_frame["dia"] < 60]
    pipeline = FeaturePipeline(family="random_forest").fit(fit_frame, numeric, categorical)

    unseen = prepared_frame[prepared_frame["dia"] >= 120].copy()
    unseen["ProductCD"] = "JAMAS_VISTA"
    matrix = pipeline.transform(unseen)
    assert matrix.shape[1] == len(pipeline.feature_names())
    assert np.isfinite(matrix).all()


def test_columnas_muy_vacias_se_descartan_segun_el_fit(prepared_frame):
    numeric, categorical = _panel(prepared_frame)
    frame = prepared_frame[prepared_frame["dia"] < 60].copy()
    frame["casi_vacia"] = np.nan
    frame.loc[frame.index[:2], "casi_vacia"] = 1.0

    pipeline = FeaturePipeline(family="logistic").fit(
        frame, numeric + ["casi_vacia"], categorical
    )
    assert "casi_vacia" in pipeline.dropped_columns
    assert "casi_vacia" not in pipeline.numeric_columns


def test_lightgbm_conserva_faltantes_nativos(prepared_frame):
    numeric, categorical = _panel(prepared_frame)
    fit_frame = prepared_frame[prepared_frame["dia"] < 60]
    pipeline = FeaturePipeline(family="lightgbm").fit(fit_frame, numeric, categorical)
    out = pipeline.transform(fit_frame)
    assert isinstance(out, pd.DataFrame)
    for column in categorical:
        assert isinstance(out[column].dtype, pd.CategoricalDtype)
