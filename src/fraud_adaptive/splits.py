"""Roles temporales, relojes y elegibilidad de etiquetas.

Este modulo es el arbitro de la causalidad del proyecto. Todo intervalo es
SEMIABIERTO ``[inicio, fin)`` y se expresa en dias relativos al primer evento.

Dos relojes conviven durante el backtest y nunca se mezclan:

* **reloj de evento**: el dia en que ocurrio la transaccion y en que se emitio la
  decision. Determina que features son visibles.
* **reloj de disponibilidad**: ``available_at = dia_evento + L``. Determina cuando
  una etiqueta puede entrar a un fit, a una metrica o a ADWIN.

La regla de elegibilidad es ``available_at < cutoff`` con desigualdad ESTRICTA.
Usar ``<=`` admitiria una etiqueta que madura exactamente el dia del corte, que en
una operacion real todavia no estaria confirmada al lanzar el job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml


# --------------------------------------------------------------------------- intervalos

@dataclass(frozen=True)
class Interval:
    """Intervalo semiabierto de dias relativos ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("Intervalo invalido: end (%s) < start (%s)" % (self.end, self.start))

    @property
    def days(self) -> int:
        return self.end - self.start

    @property
    def empty(self) -> bool:
        return self.end <= self.start

    def contains(self, day: int | float) -> bool:
        return self.start <= day < self.end

    def overlaps(self, other: "Interval") -> bool:
        """Solapamiento de intervalos semiabiertos.

        Dos intervalos que solo se tocan en el borde ([0,10) y [10,20)) NO se
        solapan: es exactamente lo que permite encadenar roles sin fuga.
        """
        if self.empty or other.empty:
            return False
        return self.start < other.end and other.start < self.end

    def mask(self, days: "pd.Series[Any] | np.ndarray") -> np.ndarray:
        values = np.asarray(days)
        return (values >= self.start) & (values < self.end)

    def clip(self, max_day: int) -> "Interval":
        """Recorta el extremo final a la cobertura real de los datos.

        El plan es explicito: si faltan periodos se reporta como limitacion, nunca
        se mueven los cortes para mejorar metricas.
        """
        return Interval(min(self.start, max_day), min(self.end, max_day))

    def as_list(self) -> list[int]:
        return [self.start, self.end]

    def __repr__(self) -> str:
        return "[%d,%d)" % (self.start, self.end)


def intervals_disjoint(intervals: Iterable[Interval]) -> bool:
    items = [iv for iv in intervals if not iv.empty]
    for i, first in enumerate(items):
        for second in items[i + 1:]:
            if first.overlaps(second):
                return False
    return True


def find_overlaps(named: dict[str, Interval]) -> list[tuple[str, str]]:
    """Devuelve los pares de roles que se solapan, para mensajes de error utiles."""
    keys = list(named)
    clashes: list[tuple[str, str]] = []
    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1:]:
            if named[key_a].overlaps(named[key_b]):
                clashes.append((key_a, key_b))
    return clashes


# --------------------------------------------------------------------------- configuracion

@dataclass
class TemporalConfig:
    """Parametros temporales cerrados del plan, cargados desde configs/temporal.yaml."""

    label_delay_days: int = 30
    cadence_days: int = 15
    tuning_folds: tuple[dict[str, Any], ...] = ()
    base_fit: Interval = Interval(0, 69)
    calibration: Interval = Interval(69, 76)
    policy: Interval = Interval(76, 83)
    promotion_validation: Interval = Interval(83, 90)
    warmup: Interval = Interval(90, 120)
    test_blocks: tuple[tuple[str, Interval], ...] = ()
    update_times: tuple[int, ...] = (120, 135, 150, 165)
    reserves_days: dict[str, list[int]] = None  # type: ignore[assignment]
    predictor_offset_end: int = -21
    min_support: dict[str, dict[str, int]] = None  # type: ignore[assignment]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TemporalConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        dev = raw["development"]
        blocks = tuple(
            (block["name"], Interval(*block["interval"])) for block in raw["test_blocks"]
        )
        return cls(
            label_delay_days=int(raw["label_delay_days"]),
            cadence_days=int(raw["cadence_days"]),
            tuning_folds=tuple(dev["tuning_folds"]),
            base_fit=Interval(*dev["base_fit"]),
            calibration=Interval(*dev["calibration"]),
            policy=Interval(*dev["policy"]),
            promotion_validation=Interval(*dev["promotion_validation"]),
            warmup=Interval(*raw["warmup"]["interval"]),
            test_blocks=blocks,
            update_times=tuple(int(t) for t in raw["update_times"]),
            reserves_days=raw["reserves_days"],
            predictor_offset_end=int(raw["predictor_offset_end"]),
            min_support=raw["min_support"],
        )

    @property
    def test_span(self) -> Interval:
        if not self.test_blocks:
            return Interval(0, 0)
        return Interval(self.test_blocks[0][1].start, self.test_blocks[-1][1].end)


# --------------------------------------------------------------------------- roles por version

@dataclass(frozen=True)
class VersionRoles:
    """Los cuatro roles disjuntos de una version de paquete en un cutoff dado.

    ``predictor`` -> ``calibration`` -> ``policy`` -> ``promotion_validation``
    se suceden en el tiempo sin solaparse. Que la validacion de promocion sea la
    cola mas reciente es lo que permite comparar champion y challenger sobre datos
    que ninguno de los dos vio (C20), sin recurrir a shadow deployment.
    """

    strategy: str
    update_time: int
    cutoff: int
    window_days: int | None
    predictor: Interval
    calibration: Interval
    policy: Interval
    promotion_validation: Interval

    def named(self) -> dict[str, Interval]:
        return {
            "predictor": self.predictor,
            "calibration": self.calibration,
            "policy": self.policy,
            "promotion_validation": self.promotion_validation,
        }

    def validate(self) -> None:
        clashes = find_overlaps(self.named())
        if clashes:
            raise ValueError(
                "Roles solapados en %s@T=%d: %s" % (self.strategy, self.update_time, clashes)
            )

    @property
    def version_id(self) -> str:
        return "%s_T%d" % (self.strategy, self.update_time)

    @property
    def job_time(self) -> int:
        """Momento en que corre el job. Es contra ESTE reloj que se mide la madurez.

        Distincion que es facil de equivocar y rompe el pipeline en silencio:

        * ``cutoff = T - L`` delimita QUE DIAS DE EVENTO pueden usarse, y es el
          origen desde el que se miden los cuatro roles.
        * ``job_time = T`` es cuando se lanza el ajuste, y es el valor contra el
          que se compara ``available_at``.

        Con T=120 y L=30, el ultimo dia de evento con etiqueta confirmada es el 89,
        justamente ``cutoff - 1``. Comparar ``available_at`` contra ``cutoff`` en
        lugar de contra ``job_time`` exigiria ``dia < 60`` y dejaria vacias las
        colas de calibracion, politica y validacion.
        """
        return self.update_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "strategy": self.strategy,
            "update_time": self.update_time,
            "cutoff": self.cutoff,
            "window_days": self.window_days,
            "predictor_days": self.predictor.days,
            **{name: iv.as_list() for name, iv in self.named().items()},
        }


def build_version_roles(
    strategy: str,
    update_time: int,
    config: TemporalConfig,
    *,
    window_days: int | None = None,
    kind: str = "sliding",
) -> VersionRoles:
    """Construye los roles de una version.

    ``kind`` decide donde EMPIEZA el predictor; el final y las tres reservas de 7
    dias son identicos para todas las estrategias. Mantenerlas iguales es lo que
    aisla el efecto del tamano de ventana: si W30 tuviera una cola de calibracion
    distinta de W90, la diferencia de costo ya no seria atribuible al olvido.
    """
    cutoff = update_time - config.label_delay_days
    predictor_end = cutoff + config.predictor_offset_end

    if kind == "static":
        # S0 se ajusta una sola vez en el tramo base y no vuelve a moverse.
        predictor = config.base_fit
    elif kind == "expanding":
        # E15 acumula toda la historia disponible hasta el corte del predictor.
        predictor = Interval(0, predictor_end)
    elif kind == "sliding":
        if window_days is None:
            raise ValueError("Una estrategia deslizante requiere window_days")
        # W abarca predictor + las tres reservas: el fit efectivo es W-21 dias.
        predictor = Interval(max(0, cutoff - window_days), predictor_end)
    else:
        raise ValueError("kind desconocido: %s" % kind)

    reserves = config.reserves_days
    roles = VersionRoles(
        strategy=strategy,
        update_time=update_time,
        cutoff=cutoff,
        window_days=window_days,
        predictor=predictor,
        calibration=Interval(cutoff + reserves["calibration"][0], cutoff + reserves["calibration"][1]),
        policy=Interval(cutoff + reserves["policy"][0], cutoff + reserves["policy"][1]),
        promotion_validation=Interval(
            cutoff + reserves["promotion_validation"][0], cutoff + reserves["promotion_validation"][1]
        ),
    )
    roles.validate()
    return roles


def build_all_version_roles(
    strategies: Sequence[dict[str, Any]], config: TemporalConfig
) -> dict[str, list[VersionRoles]]:
    """Genera la grilla completa de versiones: estrategia x tiempo de actualizacion.

    S0 es la excepcion deliberada: se crea solo en el primer corte porque, por
    definicion, no se reentrena.
    """
    out: dict[str, list[VersionRoles]] = {}
    for spec in strategies:
        name = spec["name"]
        kind = spec["kind"]
        window = spec.get("window_days")
        times = [config.update_times[0]] if kind == "static" else list(config.update_times)
        out[name] = [
            build_version_roles(name, t, config, window_days=window, kind=kind) for t in times
        ]
    return out


# --------------------------------------------------------------------------- elegibilidad

def available_at_day(event_day: "pd.Series[Any] | np.ndarray", label_delay_days: int) -> np.ndarray:
    """Dia en que la etiqueta de cada evento queda confirmada."""
    return np.asarray(event_day) + label_delay_days


def label_eligible(
    event_day: "pd.Series[Any] | np.ndarray", cutoff: int | float, label_delay_days: int
) -> np.ndarray:
    """Mascara de etiquetas ya maduras en ``cutoff``.

    Desigualdad estricta: una etiqueta que madura exactamente el dia del corte no
    esta disponible para el job que se lanza ese dia.
    """
    return available_at_day(event_day, label_delay_days) < cutoff


def mature_training_mask(
    event_day: "pd.Series[Any] | np.ndarray",
    interval: Interval,
    cutoff: int | float,
    label_delay_days: int,
) -> np.ndarray:
    """Filas utilizables para entrenar: dentro del rol Y con etiqueta madura.

    Ambas condiciones son necesarias. Estar dentro del intervalo no basta: con
    L=30 e intervalos cercanos al corte, parte del tramo todavia no tiene
    desenlace confirmado.
    """
    return interval.mask(event_day) & label_eligible(event_day, cutoff, label_delay_days)


def check_support(
    labels: "pd.Series[Any] | np.ndarray", requirement: dict[str, int]
) -> tuple[bool, dict[str, int]]:
    """Verifica el soporte minimo de fraudes y legitimas de un rol.

    Si falla, la version se marca no valida. El plan prohibe explicitamente la
    salida facil de ampliar W o de incorporar etiquetas inmaduras para completar.
    """
    values = np.asarray(labels)
    counts = {"fraud": int((values == 1).sum()), "legit": int((values == 0).sum())}
    ok = counts["fraud"] >= requirement.get("fraud", 0) and counts["legit"] >= requirement.get("legit", 0)
    return ok, counts


# --------------------------------------------------------------------------- manifest

def build_splits_manifest(
    config: TemporalConfig,
    strategies: Sequence[dict[str, Any]],
    *,
    max_day: int | None = None,
) -> dict[str, Any]:
    """Manifest auditable de todas las particiones, para data/manifests/splits.json."""
    roles = build_all_version_roles(strategies, config)
    test_blocks = []
    for name, interval in config.test_blocks:
        clipped = interval.clip(max_day) if max_day is not None else interval
        test_blocks.append({"name": name, "interval": clipped.as_list(), "days": clipped.days})

    return {
        "label_delay_days": config.label_delay_days,
        "cadence_days": config.cadence_days,
        "eligibility_rule": "available_at < cutoff (estricto)",
        "max_day_observado": max_day,
        "development": {
            "tuning_folds": [dict(fold) for fold in config.tuning_folds],
            "base_fit": config.base_fit.as_list(),
            "calibration": config.calibration.as_list(),
            "policy": config.policy.as_list(),
            "promotion_validation": config.promotion_validation.as_list(),
        },
        "warmup": {"interval": config.warmup.as_list(), "uso": "solo estado X; sin scores ni acciones"},
        "test_blocks": test_blocks,
        "versions": {name: [r.to_dict() for r in rs] for name, rs in roles.items()},
    }


def assert_no_leakage_between_roles(roles: VersionRoles, ids_by_role: dict[str, set[Any]]) -> None:
    """Comprueba sobre IDs REALES que los roles no comparten filas.

    La disjuncion de intervalos es condicion necesaria pero no suficiente: un bug
    de filtrado podria reintroducir filas. Esta verificacion trabaja sobre los
    conjuntos de TransactionID efectivamente usados por cada rol.
    """
    names = list(ids_by_role)
    for i, role_a in enumerate(names):
        for role_b in names[i + 1:]:
            shared = ids_by_role[role_a] & ids_by_role[role_b]
            if shared:
                raise ValueError(
                    "Fuga en %s: %d IDs compartidos entre %s y %s (ej. %s)"
                    % (roles.version_id, len(shared), role_a, role_b, list(shared)[:5])
                )
