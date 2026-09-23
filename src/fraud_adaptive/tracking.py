"""Trazabilidad sin servidor de tracking: manifests JSON, logs JSONL y presupuesto.

Los archivos de estado se escriben con ``atomic_write``, que escribe a un temporal
en el mismo directorio y lo renombra, de modo que una interrupcion no deja un
manifest a medias.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

LOGGER = logging.getLogger("fraud_adaptive")


# --------------------------------------------------------------------------- utilidades

def utc_now() -> str:
    """Marca de tiempo ISO-8601 en UTC.

    Es el reloj de pared de la ejecucion, independiente del reloj de evento del
    backtest definido en splits.py.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 leido por bloques, para no cargar CSV de cientos de MB en memoria."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_obj(obj: Any) -> str:
    """Hash estable de una estructura de datos.

    ``sort_keys`` hace que dos dicts equivalentes con distinto orden de insercion
    tengan el mismo hash, requisito para que el prerregistro detecte un reajuste.
    """
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return sha256_bytes(payload.encode("utf-8"))


def atomic_write(path: str | Path, data: str | bytes, *, binary: bool = False) -> Path:
    """Escribe a un temporal y lo renombra.

    Si la corrida se interrumpe, el archivo anterior queda intacto y al reanudar no
    se lee un estado truncado.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if binary else "w"
    encoding = None if binary else "utf-8"
    handle_fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".", suffix=".tmp")
    os.close(handle_fd)
    tmp = Path(tmp_name)
    try:
        with open(tmp, mode, encoding=encoding) as handle:
            handle.write(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    return atomic_write(path, json.dumps(obj, indent=indent, ensure_ascii=False, default=str))


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Agrega una linea JSON a un log. A diferencia de los checkpoints, no es atomico."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def environment_fingerprint() -> dict[str, Any]:
    """Versiones, hardware y commit, guardados en cada manifest.

    Asi cada resultado queda asociado al entorno que lo produjo.
    """
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "sklearn", "scipy", "lightgbm", "pyarrow", "river", "matplotlib"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "desconocida")
        except ImportError:
            versions[name] = "no instalada"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
        commit = result.stdout.strip() or "sin-commit"
    except (OSError, subprocess.SubprocessError):
        commit = "git-no-disponible"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "git_commit": commit,
        "packages": versions,
    }


def source_code_hash(package_dir: str | Path | None = None) -> str:
    """Hash del arbol de codigo.

    Reutilizar un artefacto de cache exige el mismo codigo, datos y configuracion.
    """
    package_dir = Path(package_dir) if package_dir else Path(__file__).parent
    parts = []
    for file in sorted(package_dir.rglob("*.py")):
        parts.append(file.relative_to(package_dir).as_posix() + ":" + sha256_file(file))
    return sha256_bytes("\n".join(parts).encode("utf-8"))


# --------------------------------------------------------------------------- presupuesto

@dataclass
class BudgetState:
    """Contador acumulado de computo.

    ``consumed_seconds`` se conserva al reanudar, porque el tope de 8 horas
    corresponde al proyecto completo.
    """

    total_minutes: float = 480.0
    reserve_minutes: float = 98.0
    consumed_seconds: float = 0.0
    tasks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def consumed_minutes(self) -> float:
        return self.consumed_seconds / 60.0

    @property
    def remaining_minutes(self) -> float:
        return self.total_minutes - self.consumed_minutes

    @property
    def exhausted(self) -> bool:
        return self.remaining_minutes <= 0

    @property
    def in_reserve(self) -> bool:
        """True cuando se agoto el trabajo previsto y se esta gastando la reserva."""
        return self.remaining_minutes <= self.reserve_minutes


class BudgetTracker:
    """Mide y persiste el consumo por tarea, con limite opcional por tarea."""

    def __init__(self, path: str | Path, total_minutes: float = 480.0, reserve_minutes: float = 98.0):
        self.path = Path(path)
        if self.path.exists():
            raw = read_json(self.path)
            self.state = BudgetState(
                total_minutes=raw.get("total_minutes", total_minutes),
                reserve_minutes=raw.get("reserve_minutes", reserve_minutes),
                consumed_seconds=raw.get("consumed_seconds", 0.0),
                tasks=raw.get("tasks", []),
            )
        else:
            self.state = BudgetState(total_minutes=total_minutes, reserve_minutes=reserve_minutes)
            self.flush()

    def flush(self) -> None:
        payload = asdict(self.state)
        payload["consumed_minutes"] = round(self.state.consumed_minutes, 2)
        payload["remaining_minutes"] = round(self.state.remaining_minutes, 2)
        payload["updated_at"] = utc_now()
        write_json(self.path, payload)

    @contextmanager
    def task(
        self,
        name: str,
        *,
        kind: str = "generic",
        limit_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Cronometra un bloque y lo registra aunque falle.

        Un limite excedido se marca con ``exceeded_limit`` y cuenta contra el
        presupuesto.
        """
        started = time.perf_counter()
        record: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "started_at": utc_now(),
            "metadata": metadata or {},
            "status": "running",
        }
        try:
            yield record
            record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - se registra y se re-lanza
            record["status"] = "error"
            record["error"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            elapsed = time.perf_counter() - started
            record["seconds"] = round(elapsed, 3)
            record["finished_at"] = utc_now()
            if limit_seconds is not None and elapsed > limit_seconds:
                record["exceeded_limit"] = True
                record["limit_seconds"] = limit_seconds
                LOGGER.warning("Tarea %s excedio su limite: %.1fs > %.1fs", name, elapsed, limit_seconds)
            self.state.consumed_seconds += elapsed
            self.state.tasks.append(record)
            self.flush()
            if self.state.exhausted:
                LOGGER.error(
                    "Presupuesto de computo agotado: %.1f min consumidos", self.state.consumed_minutes
                )
            elif self.state.in_reserve:
                LOGGER.warning("Consumiendo reserva: quedan %.1f min", self.state.remaining_minutes)


# --------------------------------------------------------------------------- runs

class RunContext:
    """Un run_id agrupa manifest, logs, checkpoints y presupuesto bajo ``runs/<run_id>/``."""

    def __init__(
        self,
        run_id: str,
        root: str | Path = "runs",
        *,
        config: dict[str, Any] | None = None,
        budget_minutes: float = 480.0,
        reserve_minutes: float = 98.0,
    ):
        self.run_id = run_id
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "checkpoints").mkdir(exist_ok=True)
        self.config = config or {}
        self.budget = BudgetTracker(self.dir / "budget.json", budget_minutes, reserve_minutes)
        self.events_path = self.dir / "events.jsonl"
        self.manifest_path = self.dir / "manifest.json"
        self.tasks_path = self.dir / "tasks.json"
        if not self.manifest_path.exists():
            self._init_manifest()

    def _init_manifest(self) -> None:
        write_json(
            self.manifest_path,
            {
                "run_id": self.run_id,
                "created_at": utc_now(),
                "config": self.config,
                "config_hash": sha256_obj(self.config),
                "code_hash": source_code_hash(),
                "environment": environment_fingerprint(),
                "subruns": [],
                "artifacts": {},
                "status": "created",
            },
        )

    # -- manifest

    def manifest(self) -> dict[str, Any]:
        return read_json(self.manifest_path)

    def update_manifest(self, **fields: Any) -> dict[str, Any]:
        data = self.manifest()
        data.update(fields)
        data["updated_at"] = utc_now()
        write_json(self.manifest_path, data)
        return data

    def register_artifact(self, key: str, path: str | Path, *, extra: dict[str, Any] | None = None) -> None:
        """Registra un artefacto con su hash, que el indice de evidencia cita."""
        path = Path(path)
        data = self.manifest()
        entry: dict[str, Any] = {
            "path": path.as_posix(),
            "exists": path.exists(),
            "registered_at": utc_now(),
        }
        if path.exists() and path.is_file():
            entry["sha256"] = sha256_file(path)
            entry["bytes"] = path.stat().st_size
        if extra:
            entry.update(extra)
        data.setdefault("artifacts", {})[key] = entry
        data["updated_at"] = utc_now()
        write_json(self.manifest_path, data)

    def register_subrun(self, subrun_id: str, kind: str, **fields: Any) -> None:
        data = self.manifest()
        data.setdefault("subruns", []).append(
            {"subrun_id": subrun_id, "kind": kind, "at": utc_now(), **fields}
        )
        write_json(self.manifest_path, data)

    # -- eventos y checkpoints

    def log(self, event: str, **fields: Any) -> None:
        append_jsonl(self.events_path, {"ts": utc_now(), "run_id": self.run_id, "event": event, **fields})

    def checkpoint_path(self, name: str) -> Path:
        return self.dir / "checkpoints" / (name + ".json")

    def save_checkpoint(self, name: str, payload: dict[str, Any]) -> Path:
        return write_json(self.checkpoint_path(name), {"saved_at": utc_now(), "payload": payload})

    def load_checkpoint(self, name: str) -> dict[str, Any] | None:
        path = self.checkpoint_path(name)
        if not path.exists():
            return None
        return read_json(path).get("payload")

    def has_checkpoint(self, name: str) -> bool:
        return self.checkpoint_path(name).exists()

    @contextmanager
    def task(self, name: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        with self.budget.task(name, **kwargs) as record:
            self.log("task_start", task=name, kind=kwargs.get("kind", "generic"))
            yield record
        self.log("task_end", task=name, seconds=record.get("seconds"), status=record.get("status"))
        write_json(self.tasks_path, {"tasks": self.budget.state.tasks, "updated_at": utc_now()})


def setup_logging(level: str = "INFO", jsonl_path: str | Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if jsonl_path:
        Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(jsonl_path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
