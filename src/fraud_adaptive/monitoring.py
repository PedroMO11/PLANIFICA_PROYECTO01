"""Politica de alertas, escalamiento y observabilidad del servicio.

Principio que gobierna el modulo
--------------------------------
NO existe la cadena ``alerta -> deploy``. Una senal genera, como maximo, una
RECOMENDACION dirigida a una persona. El reentrenamiento y la promocion requieren
autorizacion humana registrada, incluso cuando todos los gates tecnicos pasan.

Escalamiento del avance (§6.6), implementado tal cual:

* S1 o S2 -> alerta y abstencion (ampliar revision), no retuning automatico
* + S3    -> recomendacion de reentrenamiento
* S4      -> antecedente historico, no detector operativo

Rollback tecnico frente a deterioro de negocio
----------------------------------------------
Son dos cosas distintas y el modulo las separa. Un error HTTP o una latencia p95
excedida son observables al instante y habilitan un rollback tecnico bajo regla
humana preautorizada. Un deterioro de costo solo se puede afirmar con etiquetas
maduras, es decir L=30 dias despues, y nunca justifica un rollback inmediato.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .tracking import utc_now

LOGGER = logging.getLogger("fraud_adaptive.monitoring")

SEVERIDAD = ("informativa", "advertencia", "critica")


@dataclass
class Alert:
    """Una alerta con su accion recomendada y quien debe decidir."""

    senal: str
    severidad: str
    dia_evento: int
    dia_disponibilidad: int
    mensaje: str
    accion_recomendada: str
    requiere_humano: bool = True
    evidencia: dict[str, Any] = field(default_factory=dict)
    emitida_en: str = field(default_factory=utc_now)

    @property
    def retraso_dias(self) -> int:
        return self.dia_disponibilidad - self.dia_evento

    def to_dict(self) -> dict[str, Any]:
        return {
            "senal": self.senal,
            "severidad": self.severidad,
            "dia_evento": self.dia_evento,
            "dia_disponibilidad": self.dia_disponibilidad,
            "retraso_dias": self.retraso_dias,
            "mensaje": self.mensaje,
            "accion_recomendada": self.accion_recomendada,
            "requiere_humano": self.requiere_humano,
            "evidencia": self.evidencia,
            "emitida_en": self.emitida_en,
        }


class AlertPolicy:
    """Convierte registros de senales en alertas con escalamiento."""

    def __init__(self, *, ap_relative_drop_alert: float = 0.10, gap_alert_pp: float = 2.0):
        self.ap_relative_drop_alert = ap_relative_drop_alert
        self.gap_alert_pp = gap_alert_pp
        self.alerts: list[Alert] = []

    def _emit(self, alert: Alert) -> Alert:
        self.alerts.append(alert)
        LOGGER.log(
            logging.WARNING if alert.severidad != "informativa" else logging.INFO,
            "[%s] %s (dia %d, disponible %d): %s",
            alert.severidad, alert.senal, alert.dia_evento, alert.dia_disponibilidad, alert.mensaje,
        )
        return alert

    def evaluate_drift_log(self, drift_log: pd.DataFrame) -> list[Alert]:
        """Recorre la bitacora de senales y emite las alertas correspondientes."""
        if drift_log.empty:
            return []

        for _, row in drift_log.iterrows():
            if not bool(row.get("alerta", False)):
                continue
            signal = str(row.get("senal", "desconocida"))
            event_day = int(row.get("dia_evento", 0))
            available_day = int(row.get("dia_disponibilidad", event_day))

            if signal == "S_KS_PSI":
                self._emit(Alert(
                    senal=signal, severidad="advertencia",
                    dia_evento=event_day, dia_disponibilidad=available_day,
                    mensaje="Desplazamiento de covariables en %.0f%% del panel"
                            % (float(row.get("fraccion_disparada", 0)) * 100),
                    accion_recomendada="ampliar revision humana; NO retunear automaticamente",
                    evidencia={
                        "psi_max": row.get("psi_max"),
                        "ks_d_max": row.get("ks_d_max"),
                        "variables": row.get("variables_disparadas"),
                    },
                ))
            elif signal == "S1":
                self._emit(Alert(
                    senal=signal, severidad="advertencia",
                    dia_evento=event_day, dia_disponibilidad=available_day,
                    mensaje="Domain classifier separa referencia y reciente (AUC=%.3f)"
                            % float(row.get("auc", float("nan"))),
                    accion_recomendada="revisar integridad de datos antes de atribuir a drift",
                    evidencia={"auc": row.get("auc"), "n_variables": row.get("n_variables")},
                ))
            elif signal == "S2":
                self._emit(Alert(
                    senal=signal, severidad="informativa",
                    dia_evento=event_day, dia_disponibilidad=available_day,
                    mensaje="Distribucion de scores desplazada (PSI=%.3f)"
                            % float(row.get("psi_scores", float("nan"))),
                    accion_recomendada="verificar si coincide con un cambio de version",
                    evidencia={"psi_scores": row.get("psi_scores"),
                               "version_id": row.get("version_id")},
                ))
        return self.alerts

    def evaluate_adwin(self, detections: pd.DataFrame) -> list[Alert]:
        """ADWIN es la unica senal basada en error real; llega con L dias de retraso."""
        if detections.empty:
            return []
        for _, row in detections.iterrows():
            self._emit(Alert(
                senal="ADWIN_brier", severidad="critica",
                dia_evento=int(row["dia_evento"]), dia_disponibilidad=int(row["dia_disponibilidad"]),
                mensaje="Cambio en el error de la version %s tras %d etiquetas maduras"
                        % (row["version_id"], row["n_updates"]),
                accion_recomendada="recomendar reentrenamiento; requiere autorizacion humana",
                evidencia={"version_id": row["version_id"], "ancho_ventana": row.get("ancho_ventana")},
            ))
        return self.alerts

    def evaluate_performance(
        self, ap_reference: float, ap_current: float, *, day: int, available_day: int
    ) -> Alert | None:
        """Caida relativa de AP como senal DIAGNOSTICA, no como prueba de drift."""
        if not (np.isfinite(ap_reference) and np.isfinite(ap_current)) or ap_reference <= 0:
            return None
        relative_drop = (ap_reference - ap_current) / ap_reference
        if relative_drop < self.ap_relative_drop_alert:
            return None
        return self._emit(Alert(
            senal="caida_ap", severidad="advertencia",
            dia_evento=day, dia_disponibilidad=available_day,
            mensaje="AP cayo %.1f%% frente a la referencia" % (relative_drop * 100),
            accion_recomendada="diagnostico; la comparacion temporal no identifica causalmente drift",
            evidencia={"ap_referencia": ap_reference, "ap_actual": ap_current,
                       "caida_relativa": relative_drop},
        ))

    def evaluate_segments(self, segment_table: pd.DataFrame, *, day: int) -> list[Alert]:
        """Brecha social por segmento, solo donde hay soporte suficiente."""
        if segment_table.empty or "alerta_brecha" not in segment_table.columns:
            return []
        for _, row in segment_table[segment_table["alerta_brecha"]].iterrows():
            self._emit(Alert(
                senal="brecha_segmento", severidad="advertencia",
                dia_evento=day, dia_disponibilidad=day,
                mensaje="Bloqueo de legitimas en %s=%s difiere %.2f pp del global"
                        % (row["variable"], row["grupo"], row["brecha_pp"]),
                accion_recomendada="revision humana explicita antes de promover",
                evidencia={"n_legitimas": int(row["n_legitimas"]),
                           "tasa": float(row["tasa_bloqueo"]),
                           "ic": [row.get("ic_low"), row.get("ic_high")]},
            ))
        return self.alerts

    def escalation(self) -> dict[str, Any]:
        """Aplica el escalamiento del avance sobre las alertas acumuladas."""
        signals = {a.senal for a in self.alerts}
        covariate = bool(signals & {"S_KS_PSI", "S1", "S2"})
        analyst = "S3" in signals
        error = "ADWIN_brier" in signals

        if error:
            level, action = "reentrenamiento_recomendado", \
                "solicitar autorizacion humana de reentrenamiento (evidencia de error real)"
        elif covariate and analyst:
            level, action = "recomendacion", "recomendar reentrenamiento con revision humana"
        elif covariate:
            level, action = "alerta_y_abstencion", "ampliar revision humana; sin retuning automatico"
        else:
            level, action = "sin_accion", "monitoreo normal"

        return {
            "nivel": level,
            "accion": action,
            "senales_activas": sorted(signals),
            "n_alertas": len(self.alerts),
            "por_severidad": {
                severity: sum(1 for a in self.alerts if a.severidad == severity)
                for severity in SEVERIDAD
            },
            "autonomia": "Ninguna alerta despliega por si sola. Reentrenar y promover exigen humano.",
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([a.to_dict() for a in self.alerts])


# --------------------------------------------------------------------------- salud del servicio

@dataclass
class ServiceHealthPolicy:
    """Regla de rollback TECNICO, preautorizada por el responsable.

    Solo cubre fallos observables al instante. El deterioro economico necesita
    etiquetas maduras y se revisa aparte, con evidencia y decision humana.
    """

    http_error_rate_max: float = 0.01
    p95_ms_max: float = 300.0
    consecutive_batches: int = 2
    _breaches: int = 0

    def evaluate_batch(self, *, n_requests: int, n_errors: int, p95_ms: float) -> dict[str, Any]:
        error_rate = (n_errors / n_requests) if n_requests else 0.0
        breach_error = error_rate > self.http_error_rate_max
        breach_latency = p95_ms > self.p95_ms_max
        breached = breach_error or breach_latency
        self._breaches = self._breaches + 1 if breached else 0

        rollback = self._breaches >= self.consecutive_batches
        return {
            "n_requests": n_requests,
            "tasa_error": error_rate,
            "p95_ms": p95_ms,
            "supera_error": breach_error,
            "supera_latencia": breach_latency,
            "lotes_consecutivos_en_falla": self._breaches,
            "accion": "rollback_tecnico_al_paquete_anterior" if rollback else
                      ("alerta" if breached else "ok"),
            "rollback_recomendado": rollback,
            "nota": "Rollback tecnico por regla preautorizada. El deterioro de negocio "
                    "requiere etiquetas maduras y decision humana.",
        }


def observability_contract() -> dict[str, Any]:
    """Que se registra y que NO, para que el equipo lo configure en la nube."""
    return {
        "logs": {
            "formato": "JSON por linea en stdout",
            "campos": ["ts", "event_id_hash", "model_version", "policy_version", "action",
                       "latency_ms", "status"],
            "cardinalidad": "baja: sin series etiquetadas por TransactionID",
            "prohibido": ["isFraud", "features crudas", "identificadores de cliente"],
        },
        "metricas": {
            "servicio": ["latencia p50/p95/p99", "throughput", "tasa de error", "cold starts"],
            "modelo": ["distribucion de acciones", "PSI de scores", "version activa"],
            "negocio_tardio": ["costo por transaccion", "revisiones por dia", "bloqueo de legitimas"],
        },
        "retencion": "la define el equipo en su proyecto; aqui solo se emiten los eventos",
        "alertas_remotas": "Cloud Monitoring queda como diseno; no se configura desde este repositorio",
    }


def build_monitoring_report(
    drift_log: pd.DataFrame,
    adwin_detections: pd.DataFrame,
    *,
    ap_relative_drop_alert: float = 0.10,
    gap_alert_pp: float = 2.0,
    segment_table: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Consolida senales en alertas, escalamiento y evidencia trazable."""
    policy = AlertPolicy(ap_relative_drop_alert=ap_relative_drop_alert, gap_alert_pp=gap_alert_pp)
    policy.evaluate_drift_log(drift_log)
    policy.evaluate_adwin(adwin_detections)
    if segment_table is not None and not segment_table.empty:
        policy.evaluate_segments(segment_table, day=int(drift_log["dia_evento"].max()) if not drift_log.empty else 0)

    alerts = policy.to_frame()
    return {
        "escalamiento": policy.escalation(),
        "alertas": alerts,
        "n_alertas": len(alerts),
        "observabilidad": observability_contract(),
        "senales_sin_alerta": (
            "Una senal que no dispara tambien es evidencia: se conserva la bitacora completa."
        ),
    }
