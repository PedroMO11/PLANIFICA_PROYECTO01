"""Las features solo pueden mirar al pasado estricto.

Cada prueba ataca una forma concreta en que una implementacion ingenua filtra
futuro. Son las que justifican la afirmacion de causalidad del informe; sin ellas,
esa afirmacion seria solo una intencion documentada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_adaptive import features as features_module


def _frame(times, amounts, *, card="A", ids=None) -> pd.DataFrame:
    n = len(times)
    return pd.DataFrame({
        "TransactionID": ids if ids is not None else list(range(1, n + 1)),
        "TransactionDT": times,
        "TransactionAmt": amounts,
        "card1": [card] * n, "card2": [1] * n, "card3": [1] * n,
        "card4": ["visa"] * n, "card5": [1] * n, "card6": ["debit"] * n, "addr1": [1] * n,
        "DeviceType": ["desktop"] * n, "DeviceInfo": ["Windows"] * n,
        "has_identity": [1] * n, "isFraud": [0] * n,
    })


def test_primer_evento_no_tiene_historial():
    """El primer evento de una entidad no puede tener pasado."""
    out, _ = features_module.prepare_features(_frame([100], [10.0]))
    assert out.loc[0, "card_cnt_expansivo"] == 0
    assert out.loc[0, "card_cnt_24h"] == 0
    assert np.isnan(out.loc[0, "card_amt_rezago"])
    assert np.isnan(out.loc[0, "card_segundos_desde_ultimo"])


def test_la_fila_actual_nunca_entra_en_su_propio_agregado():
    """Un conteo que incluyera la fila actual seria fuga de la etiqueta por la puerta de atras."""
    out, _ = features_module.prepare_features(_frame([100, 200, 300], [10.0, 20.0, 30.0]))
    assert list(out["card_cnt_expansivo"]) == [0.0, 1.0, 2.0]
    # La suma en 24h del tercer evento son los dos anteriores: 10+20, no 10+20+30.
    assert out.loc[2, "card_amt_24h"] == pytest.approx(30.0)


def test_empates_temporales_leen_el_mismo_pasado():
    """Dos eventos con el mismo timestamp no pueden verse entre si."""
    out, _ = features_module.prepare_features(_frame([100, 200, 200, 200], [10.0, 20.0, 30.0, 40.0]))
    tied = out.loc[1:3, "card_cnt_expansivo"].tolist()
    assert tied == [1.0, 1.0, 1.0], "los empatados deben ver solo el evento de t=100"
    assert out.loc[1:3, "card_amt_24h"].nunique() == 1


def test_permutar_ids_empatados_no_cambia_las_features():
    """El desempate por ID es solo reproducibilidad: no puede alterar un valor.

    Si cambiara, el resultado dependeria de un orden arbitrario y dejaria de ser
    una propiedad del fenomeno.
    """
    columns = ["card_cnt_expansivo", "card_amt_24h", "card_cnt_1h", "card_amt_rezago"]

    base = _frame([100, 200, 200, 200, 300], [10.0, 20.0, 30.0, 40.0, 50.0], ids=[1, 2, 3, 4, 5])
    out_a, _ = features_module.prepare_features(base)

    permuted = _frame([100, 200, 200, 200, 300], [10.0, 40.0, 30.0, 20.0, 50.0], ids=[1, 4, 3, 2, 5])
    out_b, _ = features_module.prepare_features(permuted)

    # El evento posterior (t=300) ve exactamente el mismo pasado agregado.
    for column in columns:
        assert out_a.loc[4, column] == pytest.approx(out_b.loc[4, column], nan_ok=True)


def test_un_evento_futuro_no_altera_features_anteriores():
    """Anadir un monto extremo al final no puede mover ninguna fila previa."""
    short = _frame([100, 200, 300], [10.0, 20.0, 30.0])
    long = _frame([100, 200, 300, 400], [10.0, 20.0, 30.0, 999999.0])

    out_short, _ = features_module.prepare_features(short)
    out_long, _ = features_module.prepare_features(long)

    columns = [c for c in out_short.columns if c.startswith(("card_", "device_"))]
    pd.testing.assert_frame_equal(
        out_short[columns].reset_index(drop=True),
        out_long[columns].head(3).reset_index(drop=True),
        check_dtype=False,
    )


def test_ventana_es_semiabierta_hacia_atras():
    """[t-w, t): el evento exactamente en el borde entra; el actual no."""
    # t=3700 con ventana 1h=3600 => borde en 100. El evento en t=100 debe contar.
    out, _ = features_module.prepare_features(_frame([100, 3700], [10.0, 20.0]))
    assert out.loc[1, "card_cnt_1h"] == 1.0

    # t=3701 => borde en 101. El evento en t=100 ya queda fuera.
    out, _ = features_module.prepare_features(_frame([100, 3701], [10.0, 20.0]))
    assert out.loc[1, "card_cnt_1h"] == 0.0


def test_entidades_distintas_no_comparten_historial():
    frame = pd.concat([
        _frame([100, 200], [10.0, 20.0], card="A", ids=[1, 2]),
        _frame([150, 250], [30.0, 40.0], card="B", ids=[3, 4]),
    ]).sort_values("TransactionDT").reset_index(drop=True)
    out, _ = features_module.prepare_features(frame)
    for _, row in out.iterrows():
        # Cada tarjeta tiene a lo sumo un evento previo propio.
        assert row["card_cnt_expansivo"] <= 1.0


def test_sin_identidad_no_hay_entidad_de_dispositivo():
    """Sin identidad no existe dispositivo: agrupar todos esos eventos en una
    entidad comun inventaria un historial que no se observo."""
    frame = _frame([100, 200], [10.0, 20.0])
    frame["has_identity"] = [0, 0]
    out, _ = features_module.prepare_features(frame)
    assert out["device_proxy"].isna().all()
    assert out["device_cnt_24h"].isna().all()


def test_columnas_prohibidas_son_rechazadas():
    """El panel del predictor no puede contener ID ni tiempo absoluto."""
    with pytest.raises(ValueError, match="prohibidas"):
        features_module.assert_no_forbidden_features(["monto_log", "TransactionID"])
    with pytest.raises(ValueError, match="prohibidas"):
        features_module.assert_no_forbidden_features(["monto_log", "dia"])
    # Un panel limpio no levanta nada.
    features_module.assert_no_forbidden_features(["monto_log", "hora_sin", "card_cnt_24h"])


def test_panel_excluye_identificadores_y_objetivo(prepared_frame):
    numeric, categorical = features_module.build_feature_panel(prepared_frame)
    prohibited = {"TransactionID", "TransactionDT", "dia", "semana", "isFraud",
                  "card_proxy", "device_proxy", "s_rel"}
    assert not (set(numeric) | set(categorical)) & prohibited


def test_frame_desordenado_es_rechazado():
    """Calcular historial sobre filas fuera de orden daria resultados silenciosamente
    incorrectos, asi que se falla de forma explicita."""
    frame = _frame([300, 100, 200], [10.0, 20.0, 30.0])
    frame["card_proxy"] = "A"
    frame["device_proxy"] = "D"
    with pytest.raises(ValueError, match="ordenado"):
        features_module.compute_history_features(frame)
