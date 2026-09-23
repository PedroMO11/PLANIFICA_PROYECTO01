"""Comparacion de dos corridas independientes.

Dos corridas con la misma semilla, los mismos datos y la misma configuracion deben
producir los mismos numeros. La comparacion indica donde divergen si no coinciden.

Se comparan:

* el hash de codigo y de configuracion, que explica diferencias posteriores;
* el AP de cada uno de los 18 fits de tuning;
* el prerregistro: familia, configuracion y economia calibrada;
* el costo observado por estrategia.

Las duraciones no se comparan porque dependen de la carga de la maquina.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .tracking import read_json

LOGGER = logging.getLogger("fraud_adaptive.verification")

# Igualdad exacta por defecto: en la misma maquina y el mismo entorno, una
# diferencia en el ultimo bit indica no determinismo.
TOLERANCIA_EXACTA = 0.0


def _cargar_tuning(run_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    checkpoints = run_dir / "checkpoints"
    if not checkpoints.exists():
        return out
    for path in sorted(checkpoints.glob("tuning_*.json")):
        if path.stem == "tuning_resumen":
            continue
        out[path.stem] = read_json(path)["payload"]
    return out


def comparar_runs(
    run_a: str, run_b: str, *, runs_root: str | Path = "runs",
    reports_root: str | Path = "reports", tolerancia: float = TOLERANCIA_EXACTA,
) -> dict[str, Any]:
    """Compara dos corridas y devuelve un informe estructurado."""
    dir_a = Path(runs_root) / run_a
    dir_b = Path(runs_root) / run_b
    for directory in (dir_a, dir_b):
        if not directory.exists():
            raise FileNotFoundError("No existe la corrida %s" % directory)

    resultado: dict[str, Any] = {"run_a": run_a, "run_b": run_b, "tolerancia": tolerancia}

    # -- entorno
    manifest_a, manifest_b = read_json(dir_a / "manifest.json"), read_json(dir_b / "manifest.json")
    resultado["entorno"] = {
        "code_hash_igual": manifest_a.get("code_hash") == manifest_b.get("code_hash"),
        "config_hash_igual": manifest_a.get("config_hash") == manifest_b.get("config_hash"),
        "code_hash_a": manifest_a.get("code_hash", "")[:16],
        "code_hash_b": manifest_b.get("code_hash", "")[:16],
    }

    # -- tuning
    tuning_a, tuning_b = _cargar_tuning(dir_a), _cargar_tuning(dir_b)
    comunes = sorted(set(tuning_a) & set(tuning_b))
    diferencias = []
    for clave in comunes:
        ap_a, ap_b = tuning_a[clave]["ap_validacion"], tuning_b[clave]["ap_validacion"]
        if abs(ap_a - ap_b) > tolerancia:
            diferencias.append({"tarea": clave, "a": ap_a, "b": ap_b, "delta": ap_a - ap_b})
    resultado["tuning"] = {
        "n_comparados": len(comunes),
        "n_solo_en_a": len(set(tuning_a) - set(tuning_b)),
        "n_solo_en_b": len(set(tuning_b) - set(tuning_a)),
        "identicos": len(diferencias) == 0,
        "diferencias": diferencias,
    }

    # -- prerregistro
    pre_a = (dir_a / "checkpoints" / "prerregistro.json")
    pre_b = (dir_b / "checkpoints" / "prerregistro.json")
    if pre_a.exists() and pre_b.exists():
        hash_a = read_json(pre_a)["payload"]["hash"]
        hash_b = read_json(pre_b)["payload"]["hash"]
        resultado["prerregistro"] = {
            "hash_a": hash_a[:16], "hash_b": hash_b[:16], "identico": hash_a == hash_b,
        }
    else:
        resultado["prerregistro"] = {"disponible": False}

    # -- resultados finales
    tabla_a = dir_a / "outcomes.parquet"
    tabla_b = dir_b / "outcomes.parquet"
    if tabla_a.exists() and tabla_b.exists():
        resumen = []
        for estrategia in ("S0", "E15", "W30", "W60", "W90"):
            costos = []
            for ruta in (tabla_a, tabla_b):
                frame = pd.read_parquet(ruta, columns=["estrategia", "costo_observado"])
                subset = frame[frame["estrategia"] == estrategia]
                costos.append(float(subset["costo_observado"].mean()) if len(subset) else np.nan)
            resumen.append({
                "estrategia": estrategia,
                "costo_a": costos[0], "costo_b": costos[1],
                "delta": costos[0] - costos[1],
                "identico": bool(np.isclose(costos[0], costos[1], rtol=0, atol=max(tolerancia, 0))
                                 or (np.isnan(costos[0]) and np.isnan(costos[1]))),
            })
        resultado["resultados"] = {
            "por_estrategia": resumen,
            "todos_identicos": all(r["identico"] for r in resumen),
        }
    else:
        resultado["resultados"] = {"disponible": False}

    # El veredicto de reproducibilidad depende de los resultados. Un cambio de hash
    # de codigo se reporta aparte, porque editar documentacion o reportes lo mueve
    # sin afectar el entrenamiento.
    resultado["resultados_identicos"] = all([
        resultado["tuning"]["identicos"],
        resultado["prerregistro"].get("identico", True),
        resultado["resultados"].get("todos_identicos", True),
    ])
    resultado["entorno_identico"] = bool(
        resultado["entorno"]["code_hash_igual"] and resultado["entorno"]["config_hash_igual"]
    )
    resultado["reproducible"] = resultado["resultados_identicos"]
    if resultado["resultados_identicos"] and not resultado["entorno_identico"]:
        resultado["advertencia"] = (
            "Los resultados coinciden pese a que el hash de codigo difiere. Eso prueba "
            "que los cambios entre las dos corridas fueron neutrales para el resultado, "
            "no que no los hubiera: afirmar que estaban fuera del camino de "
            "entrenamiento exigiria revisarlos uno a uno. Para una comparacion estricta, "
            "ejecutar ambas corridas sin editar el codigo entre ellas."
        )
    elif not resultado["resultados_identicos"] and not resultado["entorno_identico"]:
        resultado["advertencia"] = (
            "Los resultados difieren Y el codigo cambio entre corridas: la diferencia no "
            "es atribuible a no determinismo mientras no se repita con el mismo codigo."
        )
    return resultado


def formatear(resultado: dict[str, Any]) -> str:
    """Informe legible del resultado de la comparacion."""
    lineas = [
        "Comparacion de reproducibilidad: '%s' frente a '%s'" % (resultado["run_a"], resultado["run_b"]),
        "=" * 72,
        "",
        "Entorno",
        "  hash de codigo identico       : %s (%s / %s)" % (
            resultado["entorno"]["code_hash_igual"],
            resultado["entorno"]["code_hash_a"], resultado["entorno"]["code_hash_b"]),
        "  hash de configuracion identico: %s" % resultado["entorno"]["config_hash_igual"],
        "",
        "Tuning",
        "  fits comparados               : %d" % resultado["tuning"]["n_comparados"],
        "  identicos                     : %s" % resultado["tuning"]["identicos"],
    ]
    for diferencia in resultado["tuning"]["diferencias"][:10]:
        lineas.append("    DIFIERE %s: %.10f vs %.10f" % (
            diferencia["tarea"], diferencia["a"], diferencia["b"]))

    if resultado["prerregistro"].get("disponible", True):
        lineas += [
            "",
            "Prerregistro",
            "  hash identico                 : %s (%s / %s)" % (
                resultado["prerregistro"]["identico"],
                resultado["prerregistro"]["hash_a"], resultado["prerregistro"]["hash_b"]),
        ]

    if resultado["resultados"].get("disponible", True):
        lineas += ["", "Costo observado por estrategia (UM/tx)",
                   "  %-8s %14s %14s %12s" % ("estrat.", resultado["run_a"], resultado["run_b"], "identico")]
        for fila in resultado["resultados"]["por_estrategia"]:
            lineas.append("  %-8s %14.10f %14.10f %12s" % (
                fila["estrategia"], fila["costo_a"], fila["costo_b"],
                "SI" if fila["identico"] else "NO"))

    lineas += [
        "",
        "=" * 72,
        "RESULTADOS IDENTICOS : %s" % ("SI" if resultado["resultados_identicos"] else "NO"),
        "ENTORNO IDENTICO     : %s" % ("SI" if resultado["entorno_identico"] else "NO"),
    ]
    if resultado.get("advertencia"):
        lineas += ["", "Advertencia: " + resultado["advertencia"]]
    lineas += [
        "",
        "Nota: la igualdad se verifica dentro de la misma maquina y el mismo entorno.",
        "No se promete igualdad bit a bit entre maquinas distintas: BLAS, version de",
        "CPU y orden de reduccion en punto flotante pueden diferir.",
    ]
    return "\n".join(lineas)
