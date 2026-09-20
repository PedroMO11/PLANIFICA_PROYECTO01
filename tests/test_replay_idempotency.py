"""Ledger del replay: idempotencia, atomicidad del cupo y durabilidad.

La propiedad que se prueba es la que hace segura una reanudacion: reenviar un
evento ya resuelto devuelve su decision anterior y NO consume un segundo cupo. Sin
ella, matar y relanzar la corrida nocturna inflaria las revisiones y falsearia el
costo reportado.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from fraud_adaptive.decision import APROBAR, BLOQUEAR, REVISAR, CostModel, Policy
from fraud_adaptive.replay import Ledger, contract_fixtures


def _record(event_id: str, day: int = 120, *, action: str = REVISAR,
            overflow: str = APROBAR, amount: float = 100.0) -> dict:
    return {
        "event_id": event_id, "dia": day, "p_cruda": 0.4, "p_calibrada": 0.45,
        "monto": amount, "accion": action, "accion_overflow": overflow,
        "motivo": "zona_" + action, "model_version": "W30_T120",
        "policy_version": "politica_v1", "costo_aprobar": 45.0,
        "costo_revisar": 5.5, "costo_bloquear": 2.75, "origen": "dataset",
    }


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    instance = Ledger(tmp_path / "replay.sqlite", daily_capacity=5)
    yield instance
    instance.close()


# --------------------------------------------------------------------------- idempotencia

def test_reenviar_un_evento_devuelve_su_decision_y_no_consume_cupo(ledger):
    stored, reserved = ledger.commit_decision(_record("evt-1"), reserve_review=True)
    assert reserved is True
    assert ledger.remaining(120) == 4

    again, reserved_again = ledger.commit_decision(_record("evt-1"), reserve_review=True)
    assert reserved_again is False, "un reintento no puede reservar otro cupo"
    assert ledger.remaining(120) == 4, "el cupo no debe moverse"
    assert again["accion"] == stored["accion"]
    assert again["event_id"] == stored["event_id"]


def test_muchos_reintentos_no_mueven_el_estado(ledger):
    for _ in range(20):
        ledger.commit_decision(_record("evt-1"), reserve_review=True)
    assert ledger.remaining(120) == 4
    assert ledger.summary()["n_decisiones"] == 1


def test_reintentar_no_cambia_una_decision_ya_confirmada(ledger):
    ledger.commit_decision(_record("evt-1", action=APROBAR), reserve_review=False)
    # Se intenta reescribir la misma decision con otra accion: debe ignorarse.
    stored, _ = ledger.commit_decision(_record("evt-1", action=BLOQUEAR), reserve_review=False)
    assert stored["accion"] == APROBAR


# --------------------------------------------------------------------------- cupo

def test_el_cupo_se_agota_y_cae_a_la_accion_automatica(ledger):
    for i in range(5):
        _, reserved = ledger.commit_decision(_record("evt-%d" % i), reserve_review=True)
        assert reserved is True
    assert ledger.remaining(120) == 0

    stored, reserved = ledger.commit_decision(
        _record("evt-overflow", overflow=BLOQUEAR), reserve_review=True
    )
    assert reserved is False
    assert stored["accion"] == BLOQUEAR
    assert stored["motivo"] == "overflow_cupo_menor_costo_esperado"
    assert stored["revision"] == 0


def test_el_cupo_es_independiente_por_dia(ledger):
    for i in range(5):
        ledger.commit_decision(_record("d120-%d" % i, day=120), reserve_review=True)
    assert ledger.remaining(120) == 0
    assert ledger.remaining(121) == 5

    _, reserved = ledger.commit_decision(_record("d121-0", day=121), reserve_review=True)
    assert reserved is True
    assert ledger.remaining(121) == 4


def test_el_cupo_nunca_se_excede_bajo_carga(ledger):
    for i in range(200):
        ledger.commit_decision(_record("evt-%d" % i), reserve_review=True)
    summary = ledger.summary()
    assert summary["n_revisiones"] == 5
    assert summary["excedio_capacidad"] is False
    assert summary["cupo_max_usado"] <= summary["capacidad_diaria"]


# --------------------------------------------------------------------------- durabilidad

def test_el_ledger_sobrevive_al_reinicio_del_proceso(tmp_path):
    path = tmp_path / "replay.sqlite"

    first = Ledger(path, daily_capacity=5)
    for i in range(3):
        first.commit_decision(_record("evt-%d" % i), reserve_review=True)
    first.close()

    # Un proceso nuevo lee el mismo archivo y continua donde estaba.
    second = Ledger(path, daily_capacity=5)
    assert second.remaining(120) == 2
    assert second.existing_decision("evt-0") is not None

    # Reprocesar los mismos eventos tras reanudar no consume cupo.
    for i in range(3):
        _, reserved = second.commit_decision(_record("evt-%d" % i), reserve_review=True)
        assert reserved is False
    assert second.remaining(120) == 2
    second.close()


def test_los_eventos_del_sistema_quedan_registrados(ledger):
    ledger.log_event("inicio_replay", {"paquete": "W30_T120"})
    ledger.log_event("fin_replay", {"n_eventos": 10})
    assert ledger.summary()["n_eventos_sistema"] == 2


def test_event_id_es_clave_primaria(ledger):
    """La unicidad la garantiza el esquema, no solo el codigo de aplicacion."""
    ledger.commit_decision(_record("evt-1"), reserve_review=False)
    with pytest.raises(sqlite3.IntegrityError):
        ledger.connection.execute(
            "INSERT INTO decisiones (event_id, dia, accion, creado_en) VALUES (?,?,?,?)",
            ("evt-1", 120, APROBAR, "2026-09-20T00:00:00+00:00"),
        )


def test_el_resumen_cuenta_por_accion(ledger):
    ledger.commit_decision(_record("a", action=APROBAR), reserve_review=False)
    ledger.commit_decision(_record("b", action=APROBAR), reserve_review=False)
    ledger.commit_decision(_record("c", action=BLOQUEAR), reserve_review=False)
    counts = ledger.summary()["por_accion"]
    assert counts[APROBAR] == 2 and counts[BLOQUEAR] == 1


# --------------------------------------------------------------------------- fixtures

def test_los_fixtures_estan_rotulados_como_sinteticos(cost_model):
    """El plan exige que un caso de prueba no pueda confundirse con un resultado."""
    policy = Policy(tau_low=0.0, tau_high=1.0, daily_capacity=150)
    fixtures = contract_fixtures(policy, cost_model)
    assert fixtures["es_fixture"].all()
    assert fixtures["nota"].str.contains("sintetico").all()
    # Cubren las tres acciones mas overflow.
    assert {"aprobar_p_baja", "bloquear_p_alta", "revisar_zona_intermedia",
            "overflow_sin_cupo"} <= set(fixtures["caso"])


def test_los_fixtures_reciben_la_accion_que_dicen(cost_model):
    """Cada fixture se evalua con la regla real, incluido el monto, y no contra una
    zona aproximada. El caso de monto cero es el que distingue una regla de la otra."""
    policy = Policy(tau_low=0.0, tau_high=1.0, daily_capacity=150)
    fixtures = contract_fixtures(policy, cost_model)
    for _, fixture in fixtures.iterrows():
        if fixture["esperado"] == "automatica":
            continue
        probability = np.array([float(fixture["p_forzada"])])
        amount = np.array([float(fixture["monto"])])
        accion = policy.propose(probability, cost_model.expected_costs(probability, amount))
        assert str(accion[0]) == fixture["esperado"], fixture["caso"]
