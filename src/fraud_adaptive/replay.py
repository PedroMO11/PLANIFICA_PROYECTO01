"""Replay local secuencial: el unico orquestador autorizado del estado.

Reparto de responsabilidades
----------------------------
El servicio calcula la decision. El replay la CONFIRMA, reserva el cupo y la
registra. Solo el replay escribe en el ledger.

El ledger es SQLite con una transaccion por evento. ``event_id`` es clave
primaria, asi que reintentar un evento ya resuelto devuelve su decision anterior
sin consumir un segundo cupo. Esa es la propiedad que permite matar el proceso a
mitad de una corrida y reanudarlo sin inflar las revisiones ni duplicar costos.

Limite declarado: un unico proceso secuencial garantiza el cupo. La demo NO
certifica decisiones concurrentes de produccion, que requerirían estado
distribuido (el diseno futuro con Firestore).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .decision import (
    APROBAR, BLOQUEAR, REVISAR, CostModel, Policy, cost_model_from_dict,
    fallback_action, implied_thresholds, policy_from_dict,
)
from .tracking import RunContext, utc_now, write_json

LOGGER = logging.getLogger("fraud_adaptive.replay")

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisiones (
    event_id        TEXT PRIMARY KEY,
    dia             INTEGER NOT NULL,
    p_cruda         REAL,
    p_calibrada     REAL,
    monto           REAL,
    accion          TEXT NOT NULL,
    motivo          TEXT,
    revision        INTEGER NOT NULL DEFAULT 0,
    model_version   TEXT,
    policy_version  TEXT,
    costo_aprobar   REAL,
    costo_revisar   REAL,
    costo_bloquear  REAL,
    origen          TEXT,
    creado_en       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cupo (
    dia        INTEGER PRIMARY KEY,
    usados     INTEGER NOT NULL DEFAULT 0,
    capacidad  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS eventos_sistema (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    tipo     TEXT NOT NULL,
    detalle  TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisiones_dia ON decisiones(dia);
"""


class Ledger:
    """Estado durable del replay. Fuera del contenedor de serving."""

    def __init__(self, path: str | Path, daily_capacity: int = 150):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_capacity = daily_capacity
        self.connection = sqlite3.connect(str(self.path), isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        # WAL permite que una lectura concurrente no bloquee al escritor y hace
        # mas robusta la reanudacion tras una interrupcion.
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(SCHEMA)

    def close(self) -> None:
        self.connection.close()

    # -- consultas

    def existing_decision(self, event_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM decisiones WHERE event_id = ?", (event_id,)
        ).fetchone()
        return dict(row) if row else None

    def remaining(self, day: int) -> int:
        row = self.connection.execute("SELECT usados FROM cupo WHERE dia = ?", (int(day),)).fetchone()
        used = row["usados"] if row else 0
        return max(0, self.daily_capacity - used)

    def log_event(self, kind: str, detail: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO eventos_sistema (ts, tipo, detalle) VALUES (?, ?, ?)",
            (utc_now(), kind, json.dumps(detail, ensure_ascii=False, default=str)),
        )

    # -- escritura

    def commit_decision(
        self, record: dict[str, Any], *, reserve_review: bool
    ) -> tuple[dict[str, Any], bool]:
        """Confirma una decision de forma atomica.

        Devuelve ``(registro, reservado)``. Toda la operacion -comprobar cupo,
        incrementarlo e insertar la decision- ocurre dentro de una unica
        transaccion. Si el proceso muere a mitad, el cupo no queda incrementado
        con una decision sin registrar.
        """
        event_id = str(record["event_id"])
        day = int(record["dia"])

        existing = self.existing_decision(event_id)
        if existing is not None:
            return existing, False

        cursor = self.connection.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            # Releer dentro de la transaccion: entre la comprobacion previa y este
            # punto otro intento pudo insertar el mismo evento.
            row = cursor.execute(
                "SELECT * FROM decisiones WHERE event_id = ?", (event_id,)
            ).fetchone()
            if row is not None:
                cursor.execute("COMMIT")
                return dict(row), False

            reserved = False
            if reserve_review:
                cursor.execute(
                    "INSERT INTO cupo (dia, usados, capacidad) VALUES (?, 0, ?) "
                    "ON CONFLICT(dia) DO NOTHING",
                    (day, self.daily_capacity),
                )
                used = cursor.execute("SELECT usados FROM cupo WHERE dia = ?", (day,)).fetchone()["usados"]
                if used < self.daily_capacity:
                    cursor.execute("UPDATE cupo SET usados = usados + 1 WHERE dia = ?", (day,))
                    reserved = True

            if reserve_review and not reserved:
                # El cupo se agoto: la decision cae a la accion automatica mas barata.
                record = {**record, "accion": record["accion_overflow"],
                          "motivo": "overflow_cupo_menor_costo_esperado", "revision": 0}

            cursor.execute(
                "INSERT INTO decisiones (event_id, dia, p_cruda, p_calibrada, monto, accion, motivo,"
                " revision, model_version, policy_version, costo_aprobar, costo_revisar, costo_bloquear,"
                " origen, creado_en) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event_id, day, record.get("p_cruda"), record.get("p_calibrada"), record.get("monto"),
                    record["accion"], record.get("motivo"), 1 if reserved else 0,
                    record.get("model_version"), record.get("policy_version"),
                    record.get("costo_aprobar"), record.get("costo_revisar"), record.get("costo_bloquear"),
                    record.get("origen", "replay"), utc_now(),
                ),
            )
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise

        stored = self.existing_decision(event_id)
        return stored or record, reserved

    # -- resumen

    def summary(self) -> dict[str, Any]:
        cursor = self.connection.cursor()
        total = cursor.execute("SELECT COUNT(*) AS n FROM decisiones").fetchone()["n"]
        by_action = {
            row["accion"]: row["n"]
            for row in cursor.execute("SELECT accion, COUNT(*) AS n FROM decisiones GROUP BY accion")
        }
        quota = [dict(row) for row in cursor.execute("SELECT * FROM cupo ORDER BY dia")]
        over = [q for q in quota if q["usados"] > q["capacidad"]]
        return {
            "n_decisiones": total,
            "por_accion": by_action,
            "n_revisiones": cursor.execute(
                "SELECT COUNT(*) AS n FROM decisiones WHERE revision = 1"
            ).fetchone()["n"],
            "dias_con_cupo": len(quota),
            "cupo_max_usado": max([q["usados"] for q in quota], default=0),
            "capacidad_diaria": self.daily_capacity,
            "excedio_capacidad": len(over) > 0,
            "n_eventos_sistema": cursor.execute(
                "SELECT COUNT(*) AS n FROM eventos_sistema"
            ).fetchone()["n"],
        }


# --------------------------------------------------------------------------- clientes

@dataclass
class InProcessClient:
    """Puntua con el paquete cargado en el proceso. Evita el costo HTTP."""

    package: dict[str, Any]

    def score(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = self.package["pipeline"].transform(frame)
        raw = self.package["estimator"].predict_proba(matrix)[:, 1]
        return raw, self.package["calibrator"].transform(raw)

    @property
    def version(self) -> str:
        return self.package["manifest"]["version_id"]


@dataclass
class HttpClient:
    """Cliente del endpoint HTTP, usado tal cual contra el servicio remoto.

    Es el mismo cliente que el equipo apunta al URL de Cloud Run: el ledger sigue
    siendo local y autoritativo aunque el scoring ocurra en la nube.
    """

    endpoint: str
    schema_version: str = "1.0.0"
    timeout: float = 10.0

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        response = httpx.post(self.endpoint.rstrip("/") + "/predict", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


# --------------------------------------------------------------------------- fixtures

def version_switch_demo(
    package_dir: str | Path, models_root: str | Path, frame: pd.DataFrame
) -> dict[str, Any]:
    """Demuestra el cambio de versión activa y el rollback al paquete anterior.

    Es una **prueba de contrato**, no una promoción de negocio aprobada: comprueba
    que el registro conmuta de paquete, que la versión reportada cambia y que se
    puede volver atrás. No afirma que el paquete nuevo sea mejor; eso lo decide el
    gate sobre H y, después, una persona.

    Si no hay una segunda versión disponible, se reporta así en vez de fabricar
    una: el plan prohíbe completar ramas inventando evidencia.
    """
    from .serving import PackageRegistry

    models_root = Path(models_root)
    package_dir = Path(package_dir)
    candidates = sorted(
        d for d in models_root.iterdir()
        if d.is_dir() and (d / "manifest.json").exists() and d.name != package_dir.name
    )
    if not candidates:
        return {
            "demostrado": False,
            "motivo": "no hay una segunda version disponible en %s" % models_root.as_posix(),
            "nota": "Se reporta la ausencia en lugar de fabricar un cambio de version.",
        }

    other = candidates[-1]
    registry = PackageRegistry(package_dir)
    inicial = registry.active["manifest"]["version_id"]

    schema = registry.active["pipeline"].schema()
    columns = [c for c in (list(schema["numeric_columns"]) + list(schema["categorical_columns"]))
               if c in frame.columns]
    muestra = frame[columns].head(1)

    def puntuar(reg: PackageRegistry) -> float:
        raw = reg.active["estimator"].predict_proba(reg.active["pipeline"].transform(muestra))[:, 1]
        return float(reg.active["calibrator"].transform(raw)[0])

    p_inicial = puntuar(registry)

    registry.load(other)
    activada = registry.active["manifest"]["version_id"]
    p_activada = puntuar(registry)

    revertido = registry.rollback()
    tras_rollback = registry.active["manifest"]["version_id"]
    p_tras_rollback = puntuar(registry)

    return {
        "demostrado": True,
        "version_inicial": inicial,
        "version_activada": activada,
        "rollback_ejecutado": revertido,
        "version_tras_rollback": tras_rollback,
        "rollback_correcto": tras_rollback == inicial,
        "p_inicial": p_inicial,
        "p_version_activada": p_activada,
        "p_tras_rollback": p_tras_rollback,
        "score_restaurado": abs(p_inicial - p_tras_rollback) < 1e-12,
        "naturaleza": "prueba_de_contrato",
        "nota": ("Cambio técnico de paquete. NO es una promoción de negocio aprobada: "
                 "esa requiere pasar el gate sobre H y autorización humana registrada."),
    }


def measure_inference_latency(
    package: dict[str, Any], frame: pd.DataFrame, *, warmup: int = 100, measured: int = 1000
) -> dict[str, Any]:
    """Latencia de inferencia UNA FILA A LA VEZ, como la atenderia el endpoint.

    Puntuar el lote completo y dividir daria un numero mucho mas bajo que lo que
    experimenta una peticion real: la vectorizacion amortiza el coste fijo de
    transformar y predecir. Esta funcion mide ese coste fijo, que es el que el
    objetivo de p95 <= 300 ms pretende acotar.

    El calentamiento importa: las primeras llamadas pagan la carga perezosa de
    LightGBM y de los buffers de NumPy, y contarlas inflaria la cola alta.
    """
    pipeline = package["pipeline"]
    estimator = package["estimator"]
    calibrator = package["calibrator"]

    schema = pipeline.schema()
    columns = [c for c in (list(schema["numeric_columns"]) + list(schema["categorical_columns"]))
               if c in frame.columns]
    sample = frame[columns].head(max(1, warmup + measured)).reset_index(drop=True)
    if sample.empty:
        return {"motivo_na": "sin_filas_para_medir"}

    def score_one(index: int) -> None:
        row = sample.iloc[[index % len(sample)]]
        raw = estimator.predict_proba(pipeline.transform(row))[:, 1]
        calibrator.transform(raw)

    cold_started = time.perf_counter()
    score_one(0)
    cold_ms = (time.perf_counter() - cold_started) * 1000

    for i in range(warmup):
        score_one(i)

    timings = np.empty(measured, dtype=float)
    for i in range(measured):
        started = time.perf_counter()
        score_one(i)
        timings[i] = (time.perf_counter() - started) * 1000

    return {
        "p50": float(np.percentile(timings, 50)),
        "p95": float(np.percentile(timings, 95)),
        "p99": float(np.percentile(timings, 99)),
        "media": float(timings.mean()),
        "max": float(timings.max()),
        "primera_llamada_ms": cold_ms,
        "n_warmup": warmup,
        "n_medidas": measured,
        "modo": "una_fila_por_peticion, en proceso",
        "nota": "Medicion local. No representa una region cloud ni un SLA.",
    }


def contract_fixtures(policy: Policy, cost_model: CostModel) -> pd.DataFrame:
    """Casos sinteticos ROTULADOS para ejercitar ramas que el replay natural
    puede no producir.

    El plan lo exige de forma explicita: si el dataset no genera las tres acciones
    o una alerta, se completan con fixtures claramente identificados, y no se
    cambia la politica ni se presentan como resultados de fraude.

    Los casos se construyen a partir de los umbrales que la regla economica induce
    en cada monto, de modo que ejercitan la decision real y no una aproximacion.
    """
    bajo = implied_thresholds(cost_model, 25.0, policy)
    alto = implied_thresholds(cost_model, 900.0, policy)
    medio = implied_thresholds(cost_model, 400.0, policy)
    zona_gris = (medio["tau_low"] + medio["tau_high"]) / 2.0
    rows = [
        {"caso": "aprobar_p_baja", "p_forzada": max(0.0, bajo["tau_low"] * 0.5),
         "monto": 25.0, "esperado": APROBAR},
        {"caso": "bloquear_p_alta", "p_forzada": min(1.0, alto["tau_high"] + (1 - alto["tau_high"]) * 0.5),
         "monto": 900.0, "esperado": BLOQUEAR},
        {"caso": "revisar_zona_intermedia", "p_forzada": zona_gris, "monto": 400.0,
         "esperado": REVISAR},
        {"caso": "overflow_sin_cupo", "p_forzada": zona_gris, "monto": 400.0,
         "esperado": "automatica"},
        # Monto cero: E[aprobar] = p*0 = 0 es siempre el minimo, asi que el caso
        # nunca consume cupo. Con la regla anterior, de dos umbrales sobre p, caia
        # en la zona de revision y si lo consumia. El fixture se conserva porque
        # ahora comprueba que la decision usa el monto y no solo el score.
        {"caso": "monto_cero", "p_forzada": zona_gris, "monto": 0.0, "esperado": APROBAR,
         "observacion": "E[aprobar]=0 domina; la regla economica no gasta cupo aqui"},
    ]
    frame = pd.DataFrame(rows)
    frame["es_fixture"] = True
    frame["nota"] = "Caso sintetico de prueba de contrato; no es una transaccion del dataset."
    if "observacion" not in frame.columns:
        frame["observacion"] = ""
    frame["observacion"] = frame["observacion"].fillna("")
    return frame


# --------------------------------------------------------------------------- replay

def run_replay(
    configs: dict[str, Any],
    run: RunContext,
    *,
    package_dir: str | Path,
    endpoint: str | None = None,
    max_events: int = 5000,
    ledger_path: str | Path | None = None,
    fixtures: bool = False,
) -> dict[str, Any]:
    """Ejecuta el replay local sobre los primeros eventos del periodo de test."""
    from .adaptation import ModelPackage
    from .splits import TemporalConfig

    base = configs["base"]
    decision_config = configs["decision"]
    temporal = TemporalConfig.from_yaml("configs/temporal.yaml")

    capacity = decision_config["capacity"]["daily_reviews"]

    package = ModelPackage.load_for_serving(package_dir)
    policy = policy_from_dict(package["manifest"]["policy"], daily_capacity=capacity)
    cost_model = cost_model_from_dict(package["manifest"]["costos"])

    ledger = Ledger(ledger_path or (run.dir / "replay.sqlite"), daily_capacity=capacity)
    ledger.log_event("inicio_replay", {
        "paquete": str(package_dir),
        "version": package["manifest"]["version_id"],
        "endpoint": endpoint or "en_proceso",
        "max_events": max_events,
    })

    frame = pd.read_parquet(Path(base["paths"]["processed"]) / "eventos.parquet")
    test_start = temporal.test_blocks[0][1].start
    subset = frame[frame["dia"] >= test_start].sort_values(
        ["TransactionDT", "TransactionID"], kind="mergesort"
    ).head(max_events).reset_index(drop=True)

    client = InProcessClient(package)
    http_client = HttpClient(endpoint) if endpoint else None

    started = time.perf_counter()
    raw, calibrated = client.score(subset)
    scoring_seconds = time.perf_counter() - started

    amounts = subset["TransactionAmt"].to_numpy()
    days = subset["dia"].to_numpy()
    costs = cost_model.expected_costs(calibrated, amounts)
    zones = policy.zone(calibrated)

    latencies: list[float] = []
    records: list[dict[str, Any]] = []
    idempotent_hits = 0

    for i in range(len(subset)):
        event_id = str(subset.at[i, "TransactionID"])
        request_started = time.perf_counter()

        if http_client is not None:
            # Contra el endpoint remoto: el cupo lo aporta el ledger local.
            schema = package["pipeline"].schema()
            feature_columns = list(schema["numeric_columns"]) + list(schema["categorical_columns"])
            payload = {
                "event_id": event_id,
                "event_day": int(days[i]),
                "amount": float(amounts[i]),
                "features": {
                    c: (None if pd.isna(subset.at[i, c]) else subset.at[i, c])
                    for c in feature_columns if c in subset.columns
                },
                "schema_version": configs["serving"]["service"]["schema_version"],
                "capacity_context": {
                    "remaining_reviews": ledger.remaining(int(days[i])), "day": int(days[i])
                },
            }
            response = http_client.predict(payload)
            proposed = response["action"]
            p_raw, p_cal = response["p_raw"], response["p_calibrated"]
        else:
            proposed = str(zones[i])
            p_raw, p_cal = float(raw[i]), float(calibrated[i])

        record = {
            "event_id": event_id,
            "dia": int(days[i]),
            "p_cruda": p_raw,
            "p_calibrada": p_cal,
            "monto": float(amounts[i]),
            "accion": proposed,
            "accion_overflow": fallback_action(costs, i),
            "motivo": "zona_" + proposed,
            "model_version": package["manifest"]["version_id"],
            "policy_version": policy.version,
            "costo_aprobar": float(costs[APROBAR][i]),
            "costo_revisar": float(costs[REVISAR][i]),
            "costo_bloquear": float(costs[BLOQUEAR][i]),
            "origen": "dataset",
        }
        stored, _ = ledger.commit_decision(record, reserve_review=(proposed == REVISAR))
        latencies.append((time.perf_counter() - request_started) * 1000)
        records.append(dict(stored))

    # Prueba de idempotencia: reenviar los primeros eventos no debe crear ni
    # consumir nada nuevo.
    replayed = min(50, len(subset))
    quota_before = {d: ledger.remaining(d) for d in sorted(set(days[:replayed].tolist()))}
    for i in range(replayed):
        event_id = str(subset.at[i, "TransactionID"])
        record = {
            "event_id": event_id, "dia": int(days[i]), "accion": str(zones[i]),
            "accion_overflow": fallback_action(costs, i), "monto": float(amounts[i]),
        }
        _, reserved = ledger.commit_decision(record, reserve_review=(str(zones[i]) == REVISAR))
        if not reserved:
            idempotent_hits += 1
    quota_after = {d: ledger.remaining(d) for d in quota_before}
    quota_stable = quota_before == quota_after

    fixture_results: list[dict[str, Any]] = []
    if fixtures:
        fixture_frame = contract_fixtures(policy, cost_model)
        for _, fixture in fixture_frame.iterrows():
            probability = np.array([float(fixture["p_forzada"])])
            amount = np.array([float(fixture["monto"])])
            fixture_costs = cost_model.expected_costs(probability, amount)
            # Se usa la regla real, no `zone`. Esta ultima ignora el monto y el
            # precio del cupo, de modo que evaluaba los fixtures con una politica
            # distinta de la que el servicio aplica.
            proposed = str(policy.propose(probability, fixture_costs)[0])
            # El caso de overflow se fuerza sin cupo para ejercitar esa rama.
            no_quota = fixture["caso"] == "overflow_sin_cupo"
            action = proposed
            if proposed == REVISAR and no_quota:
                action = fallback_action(fixture_costs, 0)
            expected = fixture["esperado"]
            fixture_results.append({
                "caso": fixture["caso"],
                "p_forzada": float(fixture["p_forzada"]),
                "monto": float(fixture["monto"]),
                "accion_obtenida": action,
                "accion_esperada": expected,
                "coincide": bool(expected == "automatica" and action in (APROBAR, BLOQUEAR))
                            or bool(action == expected),
                "es_fixture": True,
                "observacion": fixture.get("observacion", ""),
                "nota": fixture["nota"],
            })
        ledger.log_event("fixtures_contrato", {"n": len(fixture_results)})

    ledger_array = np.array(latencies)
    inference = measure_inference_latency(package, subset, warmup=100, measured=1000)
    decisions = pd.DataFrame(records)
    action_counts = decisions["accion"].value_counts().to_dict() if not decisions.empty else {}

    summary = {
        "paquete": str(package_dir),
        "version_modelo": package["manifest"]["version_id"],
        "politica": policy.to_dict(),
        "modo": "http:" + endpoint if endpoint else "en_proceso",
        "n_eventos": int(len(subset)),
        "dias_cubiertos": int(subset["dia"].nunique()),
        "acciones": action_counts,
        "acciones_presentes": sorted(action_counts),
        "tres_acciones_en_datos_reales": len(
            {APROBAR, REVISAR, BLOQUEAR} & set(action_counts)
        ) == 3,
        # Latencia POR PETICION, una fila a la vez, como la serviria el endpoint.
        # Es la cifra que debe compararse con el objetivo de p95 <= 300 ms.
        "latencia_ms": inference,
        # El commit al ledger se mide aparte: el replay puntua el lote por
        # adelantado, asi que mezclar ambas daria una latencia irrealmente baja.
        "latencia_ledger_ms": {
            "p50": float(np.percentile(ledger_array, 50)),
            "p95": float(np.percentile(ledger_array, 95)),
            "media": float(ledger_array.mean()),
        },
        "scoring_lote_segundos": scoring_seconds,
        "throughput_lote_tx_por_segundo": float(len(subset) / scoring_seconds) if scoring_seconds else None,
        "idempotencia": {
            "eventos_reenviados": replayed,
            "sin_nueva_reserva": idempotent_hits,
            "cupo_estable": quota_stable,
            "aprobado": bool(idempotent_hits == replayed and quota_stable),
        },
        "ledger": ledger.summary(),
        "cambio_de_version": version_switch_demo(
            package_dir, Path(base["paths"]["models"]), subset
        ),
        "fixtures": fixture_results,
        "limitaciones": [
            "Un unico orquestador secuencial garantiza el cupo; no certifica concurrencia.",
            "La latencia medida es local y no representa una region cloud.",
            "Las acciones son simuladas: no hay pagos ni bloqueos reales.",
        ],
    }

    if fixture_results:
        summary["nota_fixtures"] = (
            "Los fixtures son casos sinteticos de prueba de contrato, claramente "
            "separados de las acciones obtenidas sobre el dataset."
        )

    ledger.log_event("fin_replay", {"n_eventos": len(subset), "acciones": action_counts})
    ledger.close()

    out_dir = Path(base["paths"]["reports"])
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(out_dir / "tables" / "replay_decisiones.csv", index=False)
    write_json(run.dir / "replay_summary.json", summary)
    run.register_artifact("replay_summary", run.dir / "replay_summary.json")
    return summary
