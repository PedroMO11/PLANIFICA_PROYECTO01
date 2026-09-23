"""Comparacion de corridas.

La herramienta distingue que salgan los mismos numeros de que el entorno sea
idéntico: editar documentacion cambia el hash del codigo sin afectar los resultados.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraud_adaptive.tracking import write_json
from fraud_adaptive.verification import comparar_runs, formatear


def _crear_run(root: Path, nombre: str, *, code_hash: str, aps: dict[str, float],
               prerregistro: str = "hash-pre") -> Path:
    directorio = root / nombre
    (directorio / "checkpoints").mkdir(parents=True)
    write_json(directorio / "manifest.json", {
        "run_id": nombre, "code_hash": code_hash, "config_hash": "cfg-1",
    })
    for tarea, ap in aps.items():
        write_json(directorio / "checkpoints" / ("tuning_%s.json" % tarea),
                   {"payload": {"ap_validacion": ap, "n_fit": 100, "n_val": 50}})
    write_json(directorio / "checkpoints" / "prerregistro.json",
               {"payload": {"hash": prerregistro, "contenido": {}}})
    return directorio


def test_dos_corridas_identicas_son_reproducibles(tmp_path):
    aps = {"a_fold1": 0.5, "b_fold1": 0.4}
    _crear_run(tmp_path, "uno", code_hash="abc", aps=aps)
    _crear_run(tmp_path, "dos", code_hash="abc", aps=aps)

    resultado = comparar_runs("uno", "dos", runs_root=tmp_path)
    assert resultado["resultados_identicos"] is True
    assert resultado["entorno_identico"] is True
    assert resultado["reproducible"] is True
    assert resultado["tuning"]["n_comparados"] == 2
    assert "advertencia" not in resultado


def test_un_ap_distinto_rompe_la_reproducibilidad(tmp_path):
    _crear_run(tmp_path, "uno", code_hash="abc", aps={"a_fold1": 0.5})
    _crear_run(tmp_path, "dos", code_hash="abc", aps={"a_fold1": 0.5000001})

    resultado = comparar_runs("uno", "dos", runs_root=tmp_path)
    assert resultado["resultados_identicos"] is False
    assert len(resultado["tuning"]["diferencias"]) == 1
    assert resultado["tuning"]["diferencias"][0]["tarea"] == "tuning_a_fold1"


def test_codigo_distinto_no_invalida_resultados_iguales(tmp_path):
    """El caso que motiva separar ambos veredictos."""
    aps = {"a_fold1": 0.5}
    _crear_run(tmp_path, "uno", code_hash="abc", aps=aps)
    _crear_run(tmp_path, "dos", code_hash="xyz", aps=aps)

    resultado = comparar_runs("uno", "dos", runs_root=tmp_path)
    assert resultado["resultados_identicos"] is True
    assert resultado["entorno_identico"] is False
    assert resultado["reproducible"] is True
    # La advertencia no debe afirmar que los cambios estaban fuera del camino de
    # entrenamiento: eso exigiria revisarlos uno a uno y el verificador no lo hace.
    assert "neutrales para el resultado" in resultado["advertencia"]
    assert "no participan del entrenamiento" not in resultado["advertencia"]


def test_prerregistro_distinto_rompe_la_reproducibilidad(tmp_path):
    """Si la selección no es determinista, todo lo que viene después pierde sentido."""
    aps = {"a_fold1": 0.5}
    _crear_run(tmp_path, "uno", code_hash="abc", aps=aps, prerregistro="pre-1")
    _crear_run(tmp_path, "dos", code_hash="abc", aps=aps, prerregistro="pre-2")

    resultado = comparar_runs("uno", "dos", runs_root=tmp_path)
    assert resultado["prerregistro"]["identico"] is False
    assert resultado["resultados_identicos"] is False


def test_una_corrida_inexistente_falla_claro(tmp_path):
    _crear_run(tmp_path, "uno", code_hash="abc", aps={"a_fold1": 0.5})
    with pytest.raises(FileNotFoundError, match="No existe la corrida"):
        comparar_runs("uno", "no_existe", runs_root=tmp_path)


def test_checkpoints_desbalanceados_se_reportan(tmp_path):
    _crear_run(tmp_path, "uno", code_hash="abc", aps={"a_fold1": 0.5, "b_fold1": 0.4})
    _crear_run(tmp_path, "dos", code_hash="abc", aps={"a_fold1": 0.5})

    resultado = comparar_runs("uno", "dos", runs_root=tmp_path)
    assert resultado["tuning"]["n_comparados"] == 1
    assert resultado["tuning"]["n_solo_en_a"] == 1
    assert resultado["tuning"]["n_solo_en_b"] == 0


def test_el_informe_es_legible(tmp_path):
    aps = {"a_fold1": 0.5}
    _crear_run(tmp_path, "uno", code_hash="abc", aps=aps)
    _crear_run(tmp_path, "dos", code_hash="abc", aps=aps)

    texto = formatear(comparar_runs("uno", "dos", runs_root=tmp_path))
    assert "RESULTADOS IDENTICOS : SI" in texto
    assert "ENTORNO IDENTICO     : SI" in texto
    # El informe debe declarar el límite, no prometer portabilidad entre máquinas.
    assert "maquinas distintas" in texto
