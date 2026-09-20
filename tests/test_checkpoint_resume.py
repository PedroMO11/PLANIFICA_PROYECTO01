"""Reanudacion, presupuesto y escritura atomica.

La corrida nocturna puede interrumpirse. Lo que no puede pasar es que al reanudar
se repita trabajo ya pagado, se reinicie el contador de computo o se lea un
manifest escrito a medias.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from fraud_adaptive.tracking import (
    BudgetTracker, RunContext, atomic_write, sha256_obj, source_code_hash, write_json,
)


def test_escritura_atomica_no_deja_temporales(tmp_path):
    target = tmp_path / "salida.json"
    atomic_write(target, '{"ok": true}')
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
    assert list(tmp_path.glob(".*tmp")) == []


def test_escritura_atomica_preserva_el_contenido_previo_si_falla(tmp_path):
    """Un fallo a mitad de escritura no puede dejar el archivo truncado."""
    target = tmp_path / "salida.json"
    write_json(target, {"version": 1})

    class Explosivo:
        def __str__(self) -> str:
            raise RuntimeError("fallo simulado durante la serializacion")

    with pytest.raises(RuntimeError):
        write_json(target, {"version": Explosivo()})

    # El archivo anterior sigue intacto y legible.
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}


def test_hash_de_objeto_es_estable_ante_el_orden_de_claves():
    """Sin orden estable, el prerregistro de configuracion no detectaria nada."""
    assert sha256_obj({"a": 1, "b": 2}) == sha256_obj({"b": 2, "a": 1})
    assert sha256_obj({"a": 1}) != sha256_obj({"a": 2})


def test_el_presupuesto_persiste_entre_procesos(tmp_path):
    """El tope de 8 h es del proyecto, no de la sesion: reanudar no lo reinicia."""
    path = tmp_path / "budget.json"

    first = BudgetTracker(path, total_minutes=10.0, reserve_minutes=2.0)
    with first.task("tarea_1", kind="fit"):
        time.sleep(0.05)
    consumed = first.state.consumed_seconds
    assert consumed > 0

    # Un proceso nuevo que lee el mismo archivo hereda el consumo.
    second = BudgetTracker(path, total_minutes=10.0, reserve_minutes=2.0)
    assert second.state.consumed_seconds == pytest.approx(consumed)
    assert len(second.state.tasks) == 1

    with second.task("tarea_2", kind="fit"):
        time.sleep(0.05)
    assert second.state.consumed_seconds > consumed
    assert len(second.state.tasks) == 2


def test_una_tarea_que_falla_igual_consume_presupuesto(tmp_path):
    """Un fit fallido gasto CPU: ocultarlo falsearia el costo reportado."""
    tracker = BudgetTracker(tmp_path / "budget.json", total_minutes=10.0)
    with pytest.raises(ValueError):
        with tracker.task("tarea_rota", kind="fit"):
            raise ValueError("fallo")
    assert tracker.state.consumed_seconds > 0
    assert tracker.state.tasks[-1]["status"] == "error"
    assert "ValueError" in tracker.state.tasks[-1]["error"]


def test_una_tarea_que_excede_su_limite_queda_marcada(tmp_path):
    """Un timeout no puede presentarse como un fit exitoso."""
    tracker = BudgetTracker(tmp_path / "budget.json", total_minutes=10.0)
    with tracker.task("lenta", kind="fit", limit_seconds=0.01):
        time.sleep(0.05)
    assert tracker.state.tasks[-1]["exceeded_limit"] is True
    assert tracker.state.tasks[-1]["status"] == "ok"


def test_estados_del_presupuesto(tmp_path):
    tracker = BudgetTracker(tmp_path / "budget.json", total_minutes=1.0, reserve_minutes=0.9)
    assert not tracker.state.exhausted
    tracker.state.consumed_seconds = 10.0   # 0.167 min de 1.0 => quedan 0.83 < 0.9
    assert tracker.state.in_reserve
    tracker.state.consumed_seconds = 61.0
    assert tracker.state.exhausted


def test_checkpoints_permiten_saltar_trabajo_hecho(tmp_path):
    run = RunContext("prueba", root=tmp_path, config={"seed": 42})
    assert not run.has_checkpoint("tuning_x")

    run.save_checkpoint("tuning_x", {"ap": 0.5, "segundos": 12.0})
    assert run.has_checkpoint("tuning_x")
    assert run.load_checkpoint("tuning_x")["ap"] == 0.5

    # Un contexto nuevo sobre el mismo directorio ve el checkpoint.
    reopened = RunContext("prueba", root=tmp_path, config={"seed": 42})
    assert reopened.has_checkpoint("tuning_x")
    assert reopened.load_checkpoint("tuning_x")["ap"] == 0.5


def test_checkpoint_ausente_devuelve_none(tmp_path):
    run = RunContext("prueba", root=tmp_path)
    assert run.load_checkpoint("no_existe") is None


def test_el_manifest_registra_artefactos_con_hash(tmp_path):
    run = RunContext("prueba", root=tmp_path, config={"seed": 42})
    artifact = tmp_path / "tabla.csv"
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")

    run.register_artifact("tabla", artifact)
    entry = run.manifest()["artifacts"]["tabla"]
    assert entry["exists"] is True
    assert len(entry["sha256"]) == 64
    assert entry["bytes"] == artifact.stat().st_size


def test_el_manifest_guarda_entorno_y_hash_de_codigo(tmp_path):
    run = RunContext("prueba", root=tmp_path, config={"seed": 42})
    manifest = run.manifest()
    assert manifest["config_hash"] == sha256_obj({"seed": 42})
    assert manifest["code_hash"] == source_code_hash()
    assert "python" in manifest["environment"]
    assert "packages" in manifest["environment"]


def test_los_subruns_se_enlazan_desde_el_manifest(tmp_path):
    run = RunContext("prueba", root=tmp_path)
    run.register_subrun("W30_T120", "paquete", estrategia="W30")
    subruns = run.manifest()["subruns"]
    assert len(subruns) == 1 and subruns[0]["subrun_id"] == "W30_T120"


def test_los_eventos_se_acumulan_en_jsonl(tmp_path):
    run = RunContext("prueba", root=tmp_path)
    run.log("inicio", detalle="a")
    run.log("fin", detalle="b")
    lines = run.events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "inicio"
    assert json.loads(lines[1])["event"] == "fin"
