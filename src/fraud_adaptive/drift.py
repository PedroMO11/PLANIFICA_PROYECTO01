"""Senales de drift S1-S4 y el detector ADWIN.

Las cuatro senales del avance, con lo que cada una puede y no puede afirmar:

* **S1 domain classifier** - entrena una logistica para distinguir el periodo de
  referencia del reciente usando solo covariables. Un AUC alto prueba que P(X)
  cambio. NO prueba que P(y|X) haya cambiado.
* **S2 scores** - PSI de la distribucion de scores de una version frente a los de
  su propia calibracion. Se dispara tambien al cambiar de version, asi que cada
  cambio se anota para no atribuirlo al entorno.
* **S3 analista** - conteo de veredictos simulados. Diagnostico TARDIO: solo
  existe cuando la etiqueta madura (C28). No es feedback humano rapido.
* **S4 stale-model gap** - antecedente historico del benchmark. No se recomputa.

* **ADWIN** sobre el Brier individual ``(p_emitida - y)^2`` es la unica senal que
  observa el error real, y por eso llega con L=30 dias de retraso.

Disponibilidad, no solo deteccion
---------------------------------
Cada evento de drift guarda la fecha del EVENTO y la fecha en que el diagnostico
estuvo DISPONIBLE. Esa distancia es el objetivo O4 del avance: una senal sin
etiqueta (S1, KS/PSI) puede anticipar una caida que ADWIN solo confirma un mes
despues.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .data import benjamini_hochberg, ks_test, psi

LOGGER = logging.getLogger("fraud_adaptive.drift")


# --------------------------------------------------------------------------- KS / PSI

@dataclass
class CovariateMonitor:
    """Monitor diario de covariables contra una referencia FIJA de desarrollo.

    La referencia no se actualiza. Si se moviera con el tiempo, el monitor
    compararia el presente contra un pasado reciente ya desplazado y un drift
    lento y sostenido se volveria invisible: cada dia se pareceria al anterior.
    """

    reference: pd.DataFrame
    columns: list[str]
    psi_alert: float = 0.20
    ks_alert: float = 0.10
    bh_q: float = 0.05
    panel_fraction_alert: float = 0.20
    consecutive_closes: int = 2
    epsilon: float = 1e-6
    _reference_bins: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _reference_values: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _streak: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # La referencia es fija, asi que se convierte UNA vez y se cachea. Sin esto,
        # cada cierre diario reconvertiria ~190k filas por variable: con 23
        # variables y 62 dias son mas de 250 millones de conversiones inutiles.
        #
        # Los bordes de bin tambien se congelan aqui: comparar siempre contra la
        # misma particion es lo que hace que el PSI mida desplazamiento y no rebinning.
        for column in self.columns:
            if column not in self.reference.columns:
                continue
            values = pd.to_numeric(self.reference[column], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size < 10:
                continue
            self._reference_values[column] = values
            edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
            if edges.size >= 2:
                self._reference_bins[column] = edges

    def evaluate(self, current: pd.DataFrame, *, day: int) -> dict[str, Any]:
        """Un cierre diario del monitor sobre la ventana reciente de X."""
        rows = []
        for column in self.columns:
            if column not in current.columns or column not in self._reference_bins:
                continue
            reference_values = self._reference_values[column]
            current_values = pd.to_numeric(current[column], errors="coerce").to_numpy(dtype=float)
            d_stat, p_value = ks_test(reference_values, current_values)
            rows.append({
                "variable": column,
                "ks_d": d_stat,
                "ks_p": p_value,
                "psi": psi(reference_values, current_values,
                           bins=self._reference_bins[column], epsilon=self.epsilon),
                "n_cur": int(np.isfinite(current_values).sum()),
            })

        panel = pd.DataFrame(rows)
        if panel.empty:
            result = {"dia": day, "alerta": False, "motivo": "panel_vacio", "n_variables": 0}
            self.history.append(result)
            return result

        panel["ks_significativo"] = benjamini_hochberg(panel["ks_p"].to_numpy(), q=self.bh_q)
        # Efecto Y significacion: con cientos de miles de filas el p-valor solo
        # dice que la diferencia no es casual, no que sea grande.
        panel["dispara"] = (
            (panel["psi"] >= self.psi_alert)
            | ((panel["ks_d"] >= self.ks_alert) & panel["ks_significativo"])
        )

        fraction = float(panel["dispara"].mean())
        breached = fraction >= self.panel_fraction_alert
        self._streak = self._streak + 1 if breached else 0
        alert = self._streak >= self.consecutive_closes

        result = {
            "dia": day,
            "n_variables": int(len(panel)),
            "fraccion_disparada": fraction,
            "psi_max": float(panel["psi"].max()),
            "ks_d_max": float(panel["ks_d"].max()),
            "variables_disparadas": panel.loc[panel["dispara"], "variable"].tolist()[:10],
            "cierres_consecutivos": self._streak,
            "alerta": bool(alert),
            "senal": "S_KS_PSI",
            "limitacion": "Cambio en P(X). No demuestra cambio en P(y|X).",
        }
        self.history.append(result)
        if alert:
            LOGGER.warning("Alerta KS/PSI en dia %d: %.0f%% del panel disparado", day, fraction * 100)
        return result

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


# --------------------------------------------------------------------------- S1

def domain_classifier(
    reference: pd.DataFrame,
    recent: pd.DataFrame,
    columns: Sequence[str],
    *,
    cap_per_domain: int = 2000,
    inner_split: float = 0.70,
    auc_alert: float = 0.75,
    seed: int = 42,
) -> dict[str, Any]:
    """S1: distingue referencia de reciente usando solo covariables.

    Tres precauciones que hacen el resultado interpretable:

    1. El cap por dominio equilibra las clases; sin el, el AUC reflejaria el
       desbalance entre periodos en vez del cambio de distribucion.
    2. La division interna es CRONOLOGICA, no aleatoria, dentro de cada dominio.
    3. La imputacion y el escalado se ajustan SOLO en el 70% de entrenamiento. El
       benchmark historico los ajustaba antes de dividir, y esa contaminacion es
       una de las limitaciones que el plan arrastra como antecedente.
    """
    usable = [c for c in columns if c in reference.columns and c in recent.columns]
    if not usable:
        return {"senal": "S1", "auc": float("nan"), "motivo": "sin_columnas_comunes"}

    rng = np.random.default_rng(seed)

    def take_tail(frame: pd.DataFrame) -> pd.DataFrame:
        # Se conserva la COLA temporal, no una muestra aleatoria: el dominio
        # "reciente" debe ser el periodo mas cercano al corte.
        if len(frame) <= cap_per_domain:
            return frame
        return frame.iloc[-cap_per_domain:]

    reference_sample = take_tail(reference)
    recent_sample = take_tail(recent)

    if len(reference_sample) < 50 or len(recent_sample) < 50:
        return {"senal": "S1", "auc": float("nan"), "motivo": "soporte_insuficiente",
                "n_ref": len(reference_sample), "n_cur": len(recent_sample)}

    def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        cut = int(len(frame) * inner_split)
        return frame.iloc[:cut], frame.iloc[cut:]

    reference_train, reference_test = split(reference_sample)
    recent_train, recent_test = split(recent_sample)

    train = pd.concat([reference_train[usable], recent_train[usable]], axis=0)
    test = pd.concat([reference_test[usable], recent_test[usable]], axis=0)
    y_train = np.concatenate([np.zeros(len(reference_train)), np.ones(len(recent_train))])
    y_test = np.concatenate([np.zeros(len(reference_test)), np.ones(len(recent_test))])

    if len(np.unique(y_test)) < 2:
        return {"senal": "S1", "auc": float("nan"), "motivo": "test_interno_con_una_clase"}

    # Preprocessing ajustado SOLO con el tramo entrenable.
    train_numeric = train.apply(pd.to_numeric, errors="coerce")
    test_numeric = test.apply(pd.to_numeric, errors="coerce")
    medians = train_numeric.median()
    train_filled = train_numeric.fillna(medians)
    test_filled = test_numeric.fillna(medians)
    means = train_filled.mean()
    stds = train_filled.std().replace(0, 1.0).fillna(1.0)
    x_train = ((train_filled - means) / stds).fillna(0.0).to_numpy(dtype=float)
    x_test = ((test_filled - means) / stds).fillna(0.0).to_numpy(dtype=float)

    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(x_train, y_train)
    auc = float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))

    return {
        "senal": "S1",
        "auc": auc,
        "alerta": bool(auc >= auc_alert),
        "umbral_auc": auc_alert,
        "n_ref": int(len(reference_sample)),
        "n_cur": int(len(recent_sample)),
        "n_variables": len(usable),
        "limitacion": "AUC alto prueba cambio en P(X); no identifica cambio en P(y|X).",
    }


# --------------------------------------------------------------------------- S2

def score_drift(
    calibration_scores: np.ndarray, recent_scores: np.ndarray, *, psi_alert: float = 0.20
) -> dict[str, Any]:
    """S2: PSI de los scores frente a los de la propia calibracion de esa version."""
    value = psi(np.asarray(calibration_scores, dtype=float), np.asarray(recent_scores, dtype=float))
    return {
        "senal": "S2",
        "psi_scores": value,
        "alerta": bool(np.isfinite(value) and value >= psi_alert),
        "umbral": psi_alert,
        "n_ref": int(np.size(calibration_scores)),
        "n_cur": int(np.size(recent_scores)),
        "limitacion": "Un cambio de version desplaza los scores por si solo; anotar el cambio.",
    }


# --------------------------------------------------------------------------- S3

def analyst_signal(matured_outcomes: pd.DataFrame) -> dict[str, Any]:
    """S3: veredictos simulados, disponibles solo tras la madurez de la etiqueta."""
    reviewed = matured_outcomes[matured_outcomes["revision_admitida"]]
    if reviewed.empty:
        return {"senal": "S3", "n_revisiones": 0, "motivo": "sin_revisiones_maduras"}
    confirmed = int(reviewed["analista_detecto_fraude"].sum())
    return {
        "senal": "S3",
        "n_revisiones": int(len(reviewed)),
        "n_fraudes_confirmados": confirmed,
        "tasa_confirmacion": confirmed / len(reviewed),
        "disponibilidad": "solo tras madurez de y (C28)",
        "limitacion": "Diagnostico tardio simulado; no es ground truth ni feedback rapido.",
    }


# --------------------------------------------------------------------------- ADWIN

@dataclass
class BrierAdwin:
    """Un detector ADWIN por version, alimentado con el Brier individual.

    Dos reglas que evitan corromper el detector:

    * Cada etiqueta madura actualiza el detector EXACTAMENTE una vez. Una
      reanudacion que reprocese un dia no debe volver a alimentarlo.
    * El detector de una version retirada se conserva hasta que maduren todas sus
      predicciones. Retirarlo antes perderia la evidencia de por que fallo.
    """

    version_id: str
    delta: float = 0.002
    clock: int = 32
    _detector: Any = field(default=None, repr=False)
    _seen: set[Any] = field(default_factory=set, repr=False)
    n_updates: int = 0
    detections: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        from river.drift import ADWIN

        self._detector = ADWIN(delta=self.delta, clock=self.clock)

    def update(
        self, event_id: Any, probability: float, label: int, *, event_day: int, available_day: int
    ) -> bool:
        if event_id in self._seen:
            return False
        self._seen.add(event_id)
        loss = float((float(probability) - float(label)) ** 2)
        self._detector.update(loss)
        self.n_updates += 1
        if self._detector.drift_detected:
            self.detections.append({
                "version_id": self.version_id,
                "n_updates": self.n_updates,
                "event_id": event_id,
                # Los dos relojes: cuando ocurrio y cuando se pudo saber.
                "dia_evento": int(event_day),
                "dia_disponibilidad": int(available_day),
                "retraso_dias": int(available_day - event_day),
                "ancho_ventana": float(getattr(self._detector, "width", float("nan"))),
                "senal": "ADWIN_brier",
            })
            LOGGER.warning(
                "ADWIN detecto cambio en %s tras %d etiquetas (evento dia %d, disponible dia %d)",
                self.version_id, self.n_updates, event_day, available_day,
            )
            return True
        return False

    @property
    def n_detections(self) -> int:
        return len(self.detections)


def adwin_selftest(
    streams: dict[str, np.ndarray], *, delta: float = 0.002, clock: int = 32, jump_at: int = 1000
) -> pd.DataFrame:
    """Prueba del detector contra verdad conocida.

    Mide retardo y falsas alarmas sobre un stream estacionario y otro con un salto
    en una posicion conocida. Es validacion del INSTRUMENTO: sus numeros no son
    resultados sobre fraude y los parametros no se retocan despues de mirar los
    datos reales.
    """
    from river.drift import ADWIN

    rows = []
    for name, values in streams.items():
        if name == "jump_at":
            continue
        detector = ADWIN(delta=delta, clock=clock)
        detections: list[int] = []
        for i, value in enumerate(np.asarray(values, dtype=float)):
            detector.update(float(value))
            if detector.drift_detected:
                detections.append(i)

        has_true_change = name != "stationary"
        after_jump = [d for d in detections if d >= jump_at] if has_true_change else []
        false_alarms = len(detections) - len(after_jump) if has_true_change else len(detections)

        rows.append({
            "stream": name,
            "n_obs": int(np.size(values)),
            "cambio_real": "si (obs %d)" % jump_at if has_true_change else "no",
            "n_detecciones": len(detections),
            "primera_deteccion": detections[0] if detections else None,
            "retardo_obs": (after_jump[0] - jump_at) if after_jump else None,
            "falsas_alarmas": false_alarms,
            "delta": delta,
            "clock": clock,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- consolidacion

@dataclass
class DriftLog:
    """Bitacora unica de todas las senales, con sus dos relojes."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def add(self, signal: str, *, event_day: int, available_day: int | None = None, **fields: Any) -> None:
        self.records.append({
            "senal": signal,
            "dia_evento": int(event_day),
            "dia_disponibilidad": int(available_day if available_day is not None else event_day),
            "retraso_dias": int((available_day if available_day is not None else event_day) - event_day),
            **fields,
        })

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def alerts(self) -> pd.DataFrame:
        frame = self.to_frame()
        if frame.empty or "alerta" not in frame.columns:
            return pd.DataFrame()
        return frame[frame["alerta"].fillna(False).astype(bool)]

    def summary(self) -> dict[str, Any]:
        frame = self.to_frame()
        if frame.empty:
            return {"n_registros": 0, "n_alertas": 0}
        alerts = self.alerts()
        return {
            "n_registros": int(len(frame)),
            "n_alertas": int(len(alerts)),
            "senales_con_alerta": sorted(alerts["senal"].unique().tolist()) if not alerts.empty else [],
            "retraso_medio_dias": float(frame["retraso_dias"].mean()),
            "retraso_max_dias": int(frame["retraso_dias"].max()),
        }
