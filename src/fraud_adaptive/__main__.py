"""Interfaz de linea de comandos del sistema.

Todos los comandos son LOCALES. Ninguno crea recursos cloud, ejecuta ``gcloud`` ni
despliega nada: el runbook de ``deploy/`` describe los pasos que el equipo hace a
mano.

Flujo habitual:

    fraud-adaptive data surrogate       # solo si no hay datos reales de Kaggle
    fraud-adaptive data prepare
    fraud-adaptive train tune
    fraud-adaptive train static
    fraud-adaptive adapt run
    fraud-adaptive report build

o bien ``fraud-adaptive all`` para la cadena completa, reanudable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .tracking import RunContext, setup_logging, write_json

LOGGER = logging.getLogger("fraud_adaptive.cli")

DEFAULT_RUN_ID = "principal"


def _build_run(args: argparse.Namespace, configs: dict[str, Any]) -> RunContext:
    budget = configs["base"]["budget"]
    return RunContext(
        args.run_id,
        root=configs["base"]["paths"]["runs"],
        config=configs,
        budget_minutes=budget["total_minutes"],
        reserve_minutes=budget["reserve_minutes"],
    )


def _feature_columns(frame, configs: dict[str, Any]) -> tuple[list[str], list[str]]:
    from .features import assert_no_forbidden_features, build_feature_panel

    numeric, categorical = build_feature_panel(frame, target=configs["base"]["data"]["target"])
    assert_no_forbidden_features(numeric + categorical)
    return numeric, categorical


# --------------------------------------------------------------------------- data

def cmd_data(args: argparse.Namespace) -> int:
    from . import data as data_module
    from .pipeline import load_configs, prepare_data

    configs = load_configs(args.config_dir)
    paths = configs["base"]["paths"]

    if args.action == "instructions":
        print(data_module.kaggle_download_instructions(paths["data_root"]))
        return 0

    if args.action == "surrogate":
        from .synthetic import SyntheticConfig, write_surrogate

        target_dir = Path(paths["data_root"])
        if (target_dir / "train_transaction.csv").exists() and not args.force:
            print("Ya existen CSV en %s. Usa --force para sobrescribir." % target_dir)
            return 1
        print("Generando datos SUSTITUTOS (no son IEEE-CIS). Escala=%.2f" % args.scale)
        info = write_surrogate(target_dir, SyntheticConfig(scale=args.scale, seed=args.seed))
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    if args.action == "inspect":
        sources = data_module.resolve_sources(
            paths["data_root"],
            configs["base"]["data"]["transactions_file"],
            configs["base"]["data"]["identity_file"],
        )
        manifest = data_module.build_sources_manifest(sources)
        manifest["data_source"] = (
            "sintetico_sustituto"
            if (Path(paths["data_root"]) / "LEEME_DATOS_SUSTITUTOS.txt").exists()
            else "ieee_cis_real"
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        missing = [k for k, v in manifest["sources"].items() if not v["exists"]]
        if missing:
            print("\nFaltan fuentes: %s" % missing, file=sys.stderr)
            print(data_module.kaggle_download_instructions(paths["data_root"]), file=sys.stderr)
            return 1
        return 0

    if args.action == "prepare":
        run = _build_run(args, configs)
        frame = prepare_data(configs, run, rebuild=args.rebuild, nrows=args.nrows)
        print("Eventos preparados: %d filas, %d columnas, %d dias"
              % (len(frame), frame.shape[1], int(frame["dia"].max()) + 1))
        print("Presupuesto consumido: %.1f min" % run.budget.state.consumed_minutes)
        return 0

    raise ValueError("Accion desconocida: %s" % args.action)


# --------------------------------------------------------------------------- train

def cmd_train(args: argparse.Namespace) -> int:
    import pandas as pd

    from .pipeline import (
        fit_static_models, load_configs, prepare_data, run_tuning, select_best_configs,
    )
    from .splits import TemporalConfig

    configs = load_configs(args.config_dir)
    run = _build_run(args, configs)
    temporal = TemporalConfig.from_yaml(Path(args.config_dir) / "temporal.yaml")

    frame = prepare_data(configs, run)
    numeric, categorical = _feature_columns(frame, configs)
    LOGGER.info("Panel: %d numericas, %d categoricas", len(numeric), len(categorical))

    if args.action == "tune":
        summary = run_tuning(frame, configs, run, temporal, numeric, categorical)
        out = Path(configs["base"]["paths"]["reports"]) / "tables" / "tuning.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        run.register_artifact("tuning_tabla", out)
        print(summary.to_string(index=False))
        print("\nMejores por familia:", select_best_configs(summary))
        return 0

    if args.action == "static":
        summary_path = Path(configs["base"]["paths"]["reports"]) / "tables" / "tuning.csv"
        if not summary_path.exists():
            print("Falta la tabla de tuning. Ejecuta antes: fraud-adaptive train tune", file=sys.stderr)
            return 1
        best = select_best_configs(pd.read_csv(summary_path))
        results = fit_static_models(frame, configs, run, temporal, numeric, categorical, best)

        payload = {
            "familia_elegida": results["familia_elegida"],
            "prerregistro": results["prerregistro"],
            "baselines": results["baselines"],
            "familias": {
                name: {
                    "config": info["config"],
                    "costo_politica": info["costo_politica"],
                    "ap_politica": info["ap_politica"],
                    "policy": info["policy"].to_dict(),
                    "umbral_fpr": info["umbral_fpr"],
                    "umbral_precision": info["umbral_precision"],
                    # La curva de confiabilidad se conserva: son 10 bins y es lo
                    # que alimenta la figura de calibracion del informe.
                    "calibracion": info["calibracion"],
                    "segundos": info["segundos"],
                    "n_fit": info["n_fit"],
                }
                for name, info in results["familias"].items()
            },
        }
        out = run.dir / "static_results.json"
        write_json(out, payload)
        run.register_artifact("static_results", out)
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return 0

    raise ValueError("Accion desconocida: %s" % args.action)


# --------------------------------------------------------------------------- adapt

def cmd_adapt(args: argparse.Namespace) -> int:
    import pandas as pd

    from .pipeline import (
        fit_static_models, load_configs, prepare_data, run_adaptation, select_best_configs,
    )
    from .splits import TemporalConfig

    configs = load_configs(args.config_dir)
    run = _build_run(args, configs)
    temporal = TemporalConfig.from_yaml(Path(args.config_dir) / "temporal.yaml")

    frame = prepare_data(configs, run)
    numeric, categorical = _feature_columns(frame, configs)

    summary_path = Path(configs["base"]["paths"]["reports"]) / "tables" / "tuning.csv"
    if not summary_path.exists():
        print("Falta tuning. Ejecuta: fraud-adaptive train tune", file=sys.stderr)
        return 1
    best = select_best_configs(pd.read_csv(summary_path))
    static_results = fit_static_models(frame, configs, run, temporal, numeric, categorical, best)

    # La ventana desplegable se elige en desarrollo, antes de abrir el test.
    from .pipeline import select_deployable_window

    seleccion = select_deployable_window(
        frame, configs, run, temporal, numeric, categorical, static_results
    )
    write_json(run.dir / "seleccion_ventana.json",
               {k: (v.to_dict("records") if hasattr(v, "to_dict") else v)
                for k, v in seleccion.items()})
    run.register_artifact("seleccion_ventana", run.dir / "seleccion_ventana.json")
    print("Ventana elegida en desarrollo: %s" % seleccion.get("elegida"))

    result = run_adaptation(frame, configs, run, temporal, numeric, categorical, static_results)

    runs_dir = run.dir
    result.predictions.to_parquet(runs_dir / "predictions.parquet", index=False)
    result.outcomes.to_parquet(runs_dir / "outcomes.parquet", index=False)
    run.register_artifact("predicciones", runs_dir / "predictions.parquet")
    run.register_artifact("desenlaces", runs_dir / "outcomes.parquet")

    for name, table in (
        ("drift_log", result.drift_log),
        ("version_changes", result.version_changes),
        ("gates", result.gates),
        ("adwin_detections", result.adwin_detections),
    ):
        if not table.empty:
            path = runs_dir / (name + ".csv")
            table.to_csv(path, index=False)
            run.register_artifact(name, path)

    write_json(runs_dir / "coverage.json", {"cobertura": result.coverage, "capacidad": result.capacity})
    print(json.dumps({"cobertura": result.coverage, "capacidad": result.capacity},
                     indent=2, ensure_ascii=False, default=str))
    print("\nPresupuesto consumido: %.1f min de %.0f"
          % (run.budget.state.consumed_minutes, run.budget.state.total_minutes))
    return 0


# --------------------------------------------------------------------------- report

def cmd_package(args: argparse.Namespace) -> int:
    """Exporta un paquete desplegable, listo para el servicio o el despliegue manual."""
    import pandas as pd

    from .pipeline import (
        export_package, fit_static_models, load_configs, prepare_data, select_best_configs,
    )
    from .splits import TemporalConfig
    from .tracking import read_json

    configs = load_configs(args.config_dir)
    run = _build_run(args, configs)
    temporal = TemporalConfig.from_yaml(Path(args.config_dir) / "temporal.yaml")

    frame = prepare_data(configs, run)
    numeric, categorical = _feature_columns(frame, configs)

    static_path = run.dir / "static_results.json"
    if static_path.exists():
        # Reutiliza la seleccion ya congelada: exportar no puede reabrir decisiones.
        static_results = read_json(static_path)
    else:
        summary_path = Path(configs["base"]["paths"]["reports"]) / "tables" / "tuning.csv"
        if not summary_path.exists():
            print("Falta tuning. Ejecuta: fraud-adaptive train tune", file=sys.stderr)
            return 1
        best = select_best_configs(pd.read_csv(summary_path))
        static_results = fit_static_models(frame, configs, run, temporal, numeric, categorical, best)
        static_results = {
            "familia_elegida": static_results["familia_elegida"],
            "familias": {
                name: {"config": info["config"], "policy": info["policy"].to_dict()}
                for name, info in static_results["familias"].items()
            },
        }

    info = export_package(
        frame, configs, run, temporal, numeric, categorical, static_results,
        strategy=args.strategy, update_time=args.update_time,
    )
    print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .pipeline import load_configs
    from .reporting import build_all_reports

    configs = load_configs(args.config_dir)
    run = _build_run(args, configs)
    paths = build_all_reports(configs, run)
    print("Artefactos generados:")
    for key, path in paths.items():
        print("  %-28s %s" % (key, path))
    return 0


# --------------------------------------------------------------------------- selftest

def cmd_deliver(args: argparse.Namespace) -> int:
    """Cierra el paquete de entrega: notebooks, PDF del informe y manifiesto."""
    from .export import export_report
    from .notebooks import write_notebooks
    from .pipeline import load_configs
    from .reporting import build_delivery_manifest, write_reproduction_report
    from .tracking import write_json

    configs = load_configs(args.config_dir)
    run = _build_run(args, configs)
    reports_dir = Path(configs["base"]["paths"]["reports"])
    no_verificado: list[str] = []

    # Regenerar sin ejecutar borraria las salidas de la corrida anterior y dejaria
    # notebooks vacios en la entrega, que es peor que no tocarlos.
    if not args.execute_notebooks:
        print("1/4  Notebooks: se conservan los existentes (no se regeneran sin ejecutar).")
        notebooks = {}
    else:
        print("1/4  Generando y ejecutando notebooks...")
        notebooks = write_notebooks("notebooks")

    if args.execute_notebooks:
        import nbformat
        from nbclient import NotebookClient

        for name, path in notebooks.items():
            notebook = nbformat.read(path, as_version=4)
            try:
                NotebookClient(
                    notebook, timeout=900, kernel_name="python3",
                    resources={"metadata": {"path": "notebooks"}},
                ).execute()
                nbformat.write(notebook, path)
                print("     ejecutado: %s" % name)
            except Exception as exc:  # noqa: BLE001 - se registra y se sigue
                print("     FALLO: %s (%s)" % (name, str(exc)[:160]), file=sys.stderr)
                no_verificado.append("Notebook %s no pudo ejecutarse" % name)

    print("2/4  Exportando el informe a PDF...")
    informe_paginas: Any = "?"
    try:
        resultado = export_report(
            reports_dir / "informe_final.md", reports_dir / "informe_final.pdf",
            title="Sistema adaptativo de decision ante fraude transaccional",
            strict=not args.allow_overflow,
        )
        informe_paginas = resultado.get("paginas", "?")
        if not resultado["generado"]:
            no_verificado.append("PDF del informe: %s" % resultado.get("motivo"))
        else:
            print("     %s paginas (limite 8)" % informe_paginas)
    except ValueError as exc:
        print("     %s" % exc, file=sys.stderr)
        return 1

    print("3/4  Comprobando el inventario de entrega...")
    if not (reports_dir / "infografia.pdf").exists():
        no_verificado.append("PDF de la infografia (falta un renderizador de SVG)")
    delivery = build_delivery_manifest(configs, run, extra={
        "informe_paginas": informe_paginas,
        "no_verificado": no_verificado,
    })
    write_json(reports_dir / "manifest_entrega.json", delivery)

    print("4/4  Escribiendo el reporte de reproduccion...")
    write_reproduction_report(
        configs, run, reports_dir / "reproduccion_final.md", delivery=delivery
    )

    print("\nInventario: %d categorias" % len(delivery["inventario"]))
    if delivery["faltantes"]:
        print("FALTAN %d artefactos:" % len(delivery["faltantes"]))
        for ruta in delivery["faltantes"]:
            print("   - %s" % ruta)
    else:
        print("Todos los artefactos obligatorios estan presentes.")
    if no_verificado:
        print("\nNo pudo verificarse:")
        for item in no_verificado:
            print("   - %s" % item)
    print("\nComputo acumulado: %.1f min de %d" % (
        delivery["computo_minutos"], delivery["computo_limite_minutos"]))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Compara dos corridas para comprobar que el proyecto es reproducible."""
    from .pipeline import load_configs
    from .tracking import write_json
    from .verification import comparar_runs, formatear

    configs = load_configs(args.config_dir)
    resultado = comparar_runs(
        args.run_id, args.against,
        runs_root=configs["base"]["paths"]["runs"],
        reports_root=configs["base"]["paths"]["reports"],
    )
    print(formatear(resultado))

    destino = Path(configs["base"]["paths"]["reports"]) / "verificacion_reproducibilidad.json"
    write_json(destino, resultado)
    print("\nInforme guardado en %s" % destino)
    return 0 if resultado["reproducible"] else 1


def cmd_selftest(args: argparse.Namespace) -> int:
    from .drift import adwin_selftest
    from .pipeline import load_configs
    from .synthetic import make_adwin_streams

    configs = load_configs(args.config_dir)
    adwin_config = configs["adaptation"]["detectors"]["adwin"]
    selftest_config = configs["adaptation"]["adwin_selftest"]

    streams = make_adwin_streams(
        n_obs=selftest_config["n_obs"], seed=selftest_config["seed"],
        jump_at=selftest_config["jump_at"],
    )
    table = adwin_selftest(
        streams, delta=adwin_config["delta"], clock=adwin_config["clock"],
        jump_at=selftest_config["jump_at"],
    )
    out = Path(configs["base"]["paths"]["reports"]) / "tables" / "adwin_selftest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print("\nPrueba del INSTRUMENTO sobre streams sinteticos con cambio conocido.")
    print("No son resultados sobre fraude y sus parametros no se retocan despues.")
    return 0


# --------------------------------------------------------------------------- serve / replay

def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .pipeline import load_configs
    from .serving import create_app

    configs = load_configs(args.config_dir)
    app = create_app(package_dir=args.package, configs=configs)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from .pipeline import load_configs
    from .replay import run_replay

    configs = load_configs(args.config_dir)
    run = _build_run(args, configs)
    summary = run_replay(
        configs, run,
        package_dir=args.package,
        endpoint=args.endpoint,
        max_events=args.max_events,
        ledger_path=args.ledger,
        fixtures=args.fixtures,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


# --------------------------------------------------------------------------- all

def cmd_all(args: argparse.Namespace) -> int:
    """Cadena completa y reanudable. Cada fase deja checkpoints."""
    for step, function in (
        ("data prepare", lambda: cmd_data(argparse.Namespace(**{**vars(args), "action": "prepare",
                                                               "rebuild": False, "nrows": None}))),
        ("train tune", lambda: cmd_train(argparse.Namespace(**{**vars(args), "action": "tune"}))),
        ("train static", lambda: cmd_train(argparse.Namespace(**{**vars(args), "action": "static"}))),
        ("adapt run", lambda: cmd_adapt(args)),
        ("selftest adwin", lambda: cmd_selftest(args)),
        ("report build", lambda: cmd_report(args)),
    ):
        print("\n" + "=" * 70)
        print("FASE: %s" % step)
        print("=" * 70)
        code = function()
        if code != 0:
            print("Fase '%s' fallo con codigo %d. La cadena se detiene." % (step, code), file=sys.stderr)
            return code
    return 0


# --------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud-adaptive",
        description="Sistema adaptativo de decision ante fraude transaccional (UTEC).",
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="adquisicion, inspeccion y preparacion")
    p_data.add_argument("action", choices=["inspect", "instructions", "surrogate", "prepare"])
    p_data.add_argument("--rebuild", action="store_true", help="reconstruye el Parquet desde los CSV")
    p_data.add_argument("--nrows", type=int, default=None, help="limita filas (solo para pruebas)")
    p_data.add_argument("--scale", type=float, default=1.0, help="escala del generador sustituto")
    p_data.add_argument("--seed", type=int, default=42)
    p_data.add_argument("--force", action="store_true")
    p_data.set_defaults(func=cmd_data)

    p_train = sub.add_parser("train", help="tuning y modelos estaticos")
    p_train.add_argument("action", choices=["tune", "static"])
    p_train.set_defaults(func=cmd_train)

    p_adapt = sub.add_parser("adapt", help="experimento de adaptacion por ventanas")
    p_adapt.add_argument("action", choices=["run"], nargs="?", default="run")
    p_adapt.set_defaults(func=cmd_adapt)

    p_package = sub.add_parser("package", help="exporta un paquete desplegable")
    p_package.add_argument("action", choices=["export"], nargs="?", default="export")
    p_package.add_argument("--strategy", default="W30", help="solo estrategias desplegables")
    p_package.add_argument("--update-time", type=int, default=None,
                           help="corte T; por defecto el ultimo de la grilla")
    p_package.set_defaults(func=cmd_package)

    p_report = sub.add_parser("report", help="tablas y figuras")
    p_report.add_argument("action", choices=["build"], nargs="?", default="build")
    p_report.set_defaults(func=cmd_report)

    p_verify = sub.add_parser("verify", help="compara dos corridas (reproducibilidad)")
    p_verify.add_argument("--against", required=True,
                          help="run_id de la segunda corrida a comparar")
    p_verify.set_defaults(func=cmd_verify)

    p_deliver = sub.add_parser("deliver", help="cierra el paquete de entrega")
    p_deliver.add_argument("action", choices=["package"], nargs="?", default="package")
    p_deliver.add_argument("--execute-notebooks", action="store_true", default=True)
    p_deliver.add_argument("--no-execute-notebooks", dest="execute_notebooks",
                           action="store_false")
    p_deliver.add_argument("--allow-overflow", action="store_true",
                           help="no fallar si el informe supera las 8 paginas")
    p_deliver.set_defaults(func=cmd_deliver)

    p_self = sub.add_parser("selftest", help="prueba de detectores contra verdad conocida")
    p_self.add_argument("action", choices=["adwin"], nargs="?", default="adwin")
    p_self.set_defaults(func=cmd_selftest)

    p_serve = sub.add_parser("serve", help="servicio HTTP local de scoring")
    p_serve.add_argument("--package", required=True, help="directorio del paquete de modelo")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.set_defaults(func=cmd_serve)

    p_replay = sub.add_parser("replay", help="replay local secuencial con ledger")
    p_replay.add_argument("--package", required=True)
    p_replay.add_argument("--endpoint", default=None, help="URL del servicio; sin ella usa el modelo en proceso")
    p_replay.add_argument("--max-events", type=int, default=5000)
    p_replay.add_argument("--ledger", default=None)
    p_replay.add_argument("--fixtures", action="store_true",
                          help="anade casos sinteticos rotulados para ejercitar ramas ausentes")
    p_replay.set_defaults(func=cmd_replay)

    p_all = sub.add_parser("all", help="cadena completa reanudable")
    p_all.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nInterrumpido. Los checkpoints permiten reanudar.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
