"""Politica economica, cupo de revision y causalidad de la cola.

El cupo es el punto donde una implementacion puede volverse irreal sin que nadie
lo note: basta ordenar el dia completo por prioridad y quedarse con los 150
mejores. Estas pruebas fijan que la admision ocurre con la informacion disponible
en el momento de cada evento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_adaptive.decision import (
    APROBAR, BLOQUEAR, REVISAR, CapacityLedger, CostModel, Policy,
    decide_batch, select_thresholds, service_order, simulate_outcomes, total_cost,
)


# --------------------------------------------------------------------------- costos

def test_formula_de_costos_esperados(cost_model):
    probability = np.array([0.5])
    amount = np.array([100.0])
    costs = cost_model.expected_costs(probability, amount)
    assert costs[APROBAR][0] == pytest.approx(50.0)                 # p*m
    assert costs[BLOQUEAR][0] == pytest.approx(0.5 * 5.0)           # (1-p)*c_fp
    # c_R + p*(1-r_H)*m + (1-p)*f_H*c_FP = 1 + 0.5*0.1*100 + 0.5*0.02*5
    assert costs[REVISAR][0] == pytest.approx(1.0 + 5.0 + 0.05)


def test_revisar_no_siempre_gana(cost_model):
    """Con montos bajos, revisar cuesta mas que asumir la perdida: la politica debe
    poder preferir una accion automatica."""
    costs = cost_model.expected_costs(np.array([0.5]), np.array([1.0]))
    assert costs[REVISAR][0] > costs[APROBAR][0]


def test_umbrales_invalidos_fallan():
    with pytest.raises(ValueError):
        Policy(tau_low=0.9, tau_high=0.1)


def test_zonas_de_la_politica():
    policy = Policy(tau_low=0.2, tau_high=0.8)
    zones = policy.zone(np.array([0.05, 0.2, 0.5, 0.79, 0.8, 0.99]))
    assert list(zones) == [APROBAR, REVISAR, REVISAR, REVISAR, BLOQUEAR, BLOQUEAR]


def test_umbrales_iguales_eliminan_la_zona_gris():
    policy = Policy(tau_low=0.5, tau_high=0.5)
    zones = policy.zone(np.array([0.49, 0.5, 0.51]))
    assert list(zones) == [APROBAR, BLOQUEAR, BLOQUEAR]
    assert REVISAR not in set(zones)


# --------------------------------------------------------------------------- cupo

def test_el_cupo_diario_nunca_se_excede(cost_model):
    n = 500
    policy = Policy(tau_low=0.0, tau_high=1.1, daily_capacity=150)  # todo cae en revisar
    decisions = decide_batch(
        np.full(n, 0.5), np.full(n, 100.0), np.zeros(n, dtype=int),
        list(range(n)), policy, cost_model, ledger=CapacityLedger(150),
    )
    assert int(decisions["revision_admitida"].sum()) == 150


def test_el_cupo_se_reinicia_cada_dia(cost_model):
    days = np.array([0] * 200 + [1] * 200)
    n = len(days)
    policy = Policy(tau_low=0.0, tau_high=1.1, daily_capacity=150)
    decisions = decide_batch(
        np.full(n, 0.5), np.full(n, 100.0), days, list(range(n)),
        policy, cost_model, ledger=CapacityLedger(150),
    )
    by_day = decisions[decisions["revision_admitida"]].groupby("dia").size()
    assert by_day.to_dict() == {0: 150, 1: 150}


def test_overflow_elige_la_accion_automatica_mas_barata(cost_model):
    """Agotado el cupo, el caso cae a aprobar o bloquear segun costo esperado."""
    n = 160
    policy = Policy(tau_low=0.0, tau_high=1.1, daily_capacity=150)
    # Monto alto: con p=0.5, aprobar cuesta 500 y bloquear 2.5 => debe bloquear.
    decisions = decide_batch(
        np.full(n, 0.5), np.full(n, 1000.0), np.zeros(n, dtype=int),
        list(range(n)), policy, cost_model, ledger=CapacityLedger(150),
    )
    overflow = decisions[decisions["motivo"] == "overflow_cupo_menor_costo_esperado"]
    assert len(overflow) == 10
    assert set(overflow["accion"]) == {BLOQUEAR}

    # Monto bajo: aprobar cuesta 0.05 y bloquear 2.5 => debe aprobar.
    decisions = decide_batch(
        np.full(n, 0.1), np.full(n, 0.5), np.zeros(n, dtype=int),
        list(range(n)), policy, cost_model, ledger=CapacityLedger(150),
    )
    overflow = decisions[decisions["motivo"] == "overflow_cupo_menor_costo_esperado"]
    assert set(overflow["accion"]) == {APROBAR}


def test_admision_es_causal_no_top_k_del_dia(cost_model):
    """La admision es por llegada mientras haya cupo.

    Si el sistema ordenara el dia completo por p*monto y admitiera los mejores 150,
    estaria usando transacciones que aun no habian ocurrido. Aqui los 150 primeros
    en llegar son los admitidos, aunque lleguen despues casos de mayor prioridad.
    """
    n = 200
    # Prioridad creciente: los ultimos son los mas valiosos.
    probability = np.linspace(0.30, 0.70, n)
    amounts = np.linspace(10.0, 5000.0, n)
    policy = Policy(tau_low=0.0, tau_high=1.1, daily_capacity=150)
    decisions = decide_batch(
        probability, amounts, np.zeros(n, dtype=int), list(range(n)),
        policy, cost_model, ledger=CapacityLedger(150),
    )
    admitted = decisions[decisions["revision_admitida"]]
    assert list(admitted["event_id"]) == list(range(150))
    assert 199 not in set(admitted["event_id"]), "un oracle habria admitido el de mayor prioridad"


def test_prioridad_ordena_el_servicio_no_la_admision(cost_model):
    """p*monto decide a quien mira primero el analista, entre los ya admitidos."""
    n = 10
    probability = np.linspace(0.3, 0.6, n)
    amounts = np.linspace(100.0, 1000.0, n)
    policy = Policy(tau_low=0.0, tau_high=1.1, daily_capacity=150)
    decisions = decide_batch(
        probability, amounts, np.zeros(n, dtype=int), list(range(n)),
        policy, cost_model, ledger=CapacityLedger(150),
    )
    ordered = service_order(decisions)
    assert len(ordered) == n
    priorities = ordered["prioridad"].to_numpy()
    assert np.all(np.diff(priorities) <= 0), "el servicio va de mayor a menor prioridad"
    assert ordered.iloc[0]["event_id"] == 9


def test_reintento_es_idempotente_y_no_consume_cupo(cost_model):
    ledger = CapacityLedger(150)
    policy = Policy(tau_low=0.0, tau_high=1.1, daily_capacity=150)
    args = (np.array([0.5]), np.array([100.0]), np.array([0]), ["evento-1"], policy, cost_model)

    first = decide_batch(*args, ledger=ledger)
    remaining_after_first = ledger.remaining(0)
    second = decide_batch(*args, ledger=ledger)

    assert bool(second.iloc[0]["idempotente"]) is True
    assert ledger.remaining(0) == remaining_after_first
    assert first.iloc[0]["accion"] == second.iloc[0]["accion"]


# --------------------------------------------------------------------------- desenlaces

def test_costo_observado_suma_sus_tres_componentes(cost_model):
    decisions = pd.DataFrame([
        {"event_id": 1, "dia": 0, "p": 0.9, "monto": 100.0, "accion": APROBAR,
         "accion_propuesta": APROBAR, "motivo": "zona", "revision_admitida": False,
         "prioridad": 90.0, "costo_esp_aprobar": 90.0, "costo_esp_revisar": 0.0,
         "costo_esp_bloquear": 0.5, "cupo_restante": 150, "idempotente": False},
        {"event_id": 2, "dia": 0, "p": 0.9, "monto": 200.0, "accion": BLOQUEAR,
         "accion_propuesta": BLOQUEAR, "motivo": "zona", "revision_admitida": False,
         "prioridad": 180.0, "costo_esp_aprobar": 180.0, "costo_esp_revisar": 0.0,
         "costo_esp_bloquear": 0.5, "cupo_restante": 150, "idempotente": False},
    ])
    outcomes = simulate_outcomes(decisions, np.array([1, 0]), cost_model, seed=42)
    # Fraude aprobado: se pierde el monto. Legitima bloqueada: friccion c_FP.
    assert outcomes.loc[0, "costo_observado"] == pytest.approx(100.0)
    assert outcomes.loc[1, "costo_observado"] == pytest.approx(5.0)
    summary = total_cost(outcomes)
    assert summary["costo_total"] == pytest.approx(105.0)


def test_el_analista_simulado_es_determinista(cost_model):
    n = 300
    policy = Policy(tau_low=0.0, tau_high=1.1, daily_capacity=1000)
    decisions = decide_batch(
        np.full(n, 0.5), np.full(n, 100.0), np.zeros(n, dtype=int), list(range(n)),
        policy, cost_model, ledger=CapacityLedger(1000),
    )
    labels = np.array([1] * 150 + [0] * 150)
    first = simulate_outcomes(decisions, labels, cost_model, seed=42)
    second = simulate_outcomes(decisions, labels, cost_model, seed=42)
    pd.testing.assert_series_equal(first["analista_detecto_fraude"], second["analista_detecto_fraude"])


def test_tasa_de_deteccion_del_analista_se_aproxima_a_r_h(cost_model):
    n = 4000
    policy = Policy(tau_low=0.0, tau_high=1.1, daily_capacity=n)
    decisions = decide_batch(
        np.full(n, 0.5), np.full(n, 100.0), np.zeros(n, dtype=int), list(range(n)),
        policy, cost_model, ledger=CapacityLedger(n),
    )
    outcomes = simulate_outcomes(decisions, np.ones(n, dtype=int), cost_model, seed=42)
    rate = outcomes["analista_detecto_fraude"].mean()
    assert rate == pytest.approx(cost_model.r_h, abs=0.03)


def test_fraude_revisado_y_no_detectado_cuenta_como_aprobado(cost_model):
    """El sistema no puede acreditarse un fraude que el analista dejo pasar."""
    n = 200
    policy = Policy(tau_low=0.0, tau_high=1.1, daily_capacity=n)
    decisions = decide_batch(
        np.full(n, 0.5), np.full(n, 100.0), np.zeros(n, dtype=int), list(range(n)),
        policy, cost_model, ledger=CapacityLedger(n),
    )
    outcomes = simulate_outcomes(decisions, np.ones(n, dtype=int), cost_model, seed=42)
    missed = outcomes[~outcomes["analista_detecto_fraude"]]
    assert missed["fraude_aprobado"].all()
    assert (missed["monto_fraude_evitado"] == 0).all()


# --------------------------------------------------------------------------- umbrales

def test_seleccion_de_umbrales_minimiza_costo(cost_model):
    rng = np.random.default_rng(0)
    n = 1200
    labels = rng.binomial(1, 0.05, size=n)
    # Score informativo: los fraudes puntuan mas alto.
    probability = np.clip(rng.normal(0.1, 0.05, n) + labels * 0.45, 0.001, 0.999)
    amounts = rng.lognormal(3.5, 1.0, n)
    days = np.repeat(np.arange(8), n // 8)

    selection = select_thresholds(
        probability, amounts, days, labels, list(range(n)), cost_model,
        quantiles=[0.0, 0.5, 0.9, 0.95, 0.99, 1.0], daily_capacity=150, seed=42,
    )
    grid = selection["grid"]
    best = selection["best"]
    assert best["costo_por_tx"] == pytest.approx(grid["costo_por_tx"].min())
    assert selection["policy"].tau_low <= selection["policy"].tau_high
    # La eleccion no puede ser peor que bloquear o aprobar todo, que estan en la grilla.
    assert best["costo_por_tx"] <= grid["costo_por_tx"].max()


def test_los_umbrales_no_se_asumen_en_medio_punto(cost_model):
    """El plan prohibe el 0.5 por defecto: el umbral sale del costo."""
    rng = np.random.default_rng(1)
    n = 800
    labels = rng.binomial(1, 0.03, size=n)
    probability = np.clip(rng.normal(0.05, 0.03, n) + labels * 0.3, 0.001, 0.999)
    selection = select_thresholds(
        probability, rng.lognormal(3, 1, n), np.repeat(np.arange(8), n // 8),
        labels, list(range(n)), cost_model,
        quantiles=[0.0, 0.5, 0.9, 0.99, 1.0], daily_capacity=150, seed=42,
    )
    assert selection["policy"].tau_high != 0.5
