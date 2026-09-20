"""Adquisicion, validacion e integracion de las dos fuentes tabulares.

Cubre F2 del plan: join auditado transacciones-identidad, derivacion del reloj de
evento y EDA temporal con intervalos de Wilson, KS y PSI.

Dos advertencias que el codigo hace explicitas porque el dataset invita al error:

* ``TransactionDT`` es un DELTA en segundos desde un origen desconocido, no una
  fecha. La hora derivada es relativa; no identifica hora local ni dia laboral.
* La ausencia de identidad NO es un dato faltante a imputar: es informacion
  (``has_identity``). Eliminar esas filas sesgaria el panel completo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .tracking import sha256_file, utc_now, write_json

LOGGER = logging.getLogger("fraud_adaptive.data")

SECONDS_PER_DAY = 86_400


# --------------------------------------------------------------------------- fuentes

@dataclass
class SourceSpec:
    """Ubicacion y rol de un CSV crudo."""

    name: str
    path: Path
    role: str

    @property
    def exists(self) -> bool:
        return self.path.exists()


def resolve_sources(data_root: str | Path, transactions_file: str, identity_file: str) -> dict[str, SourceSpec]:
    root = Path(data_root)
    return {
        "transactions": SourceSpec("transactions", root / transactions_file, "tabla base, una fila por evento"),
        "identity": SourceSpec("identity", root / identity_file, "tabla complementaria, cobertura parcial"),
    }


def build_sources_manifest(sources: dict[str, SourceSpec], *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Manifest con hash y esquema de cada fuente.

    El hash es lo que permite afirmar mas tarde que un resultado corresponde a
    estos archivos y no a una version distinta descargada despues.
    """
    entries: dict[str, Any] = {}
    for key, spec in sources.items():
        entry: dict[str, Any] = {
            "path": spec.path.as_posix(),
            "role": spec.role,
            "exists": spec.exists,
        }
        if spec.exists:
            entry["bytes"] = spec.path.stat().st_size
            entry["sha256"] = sha256_file(spec.path)
            header = pd.read_csv(spec.path, nrows=0)
            entry["n_columns"] = int(header.shape[1])
            entry["columns_preview"] = list(header.columns[:15])
        entries[key] = entry
    manifest = {"generated_at": utc_now(), "sources": entries}
    if extra:
        manifest.update(extra)
    return manifest


def assert_no_forbidden_files(data_root: str | Path, forbidden: Sequence[str]) -> None:
    """Falla si los CSV de test de Kaggle estan al alcance del pipeline.

    No tienen etiqueta: usarlos no produciria fuga de etiquetas, pero si una
    evaluacion sin ground truth que el informe no podria sustentar. El plan los
    prohibe de forma explicita, asi que el codigo lo verifica en vez de confiar.
    """
    root = Path(data_root)
    present = [name for name in forbidden if (root / name).exists()]
    if present:
        raise RuntimeError(
            "Archivos prohibidos presentes en DATA_ROOT: %s. El backtest usa solo train_*." % present
        )


def kaggle_download_instructions(data_root: str | Path) -> str:
    """Instrucciones de reconstruccion. No ejecuta nada ni maneja credenciales."""
    return (
        "Descarga reproducible de IEEE-CIS Fraud Detection\n"
        "-------------------------------------------------\n"
        "1. Acepta las reglas de la competencia en:\n"
        "   https://www.kaggle.com/competitions/ieee-fraud-detection/rules\n"
        "2. Crea un token en Kaggle > Settings > API > Create New Token.\n"
        "   Guarda kaggle.json en %USERPROFILE%\\.kaggle\\kaggle.json (fuera del repositorio).\n"
        "3. Instala el cliente y descarga solo los dos archivos etiquetados:\n"
        "     uv pip install kaggle\n"
        "     kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p " + str(data_root) + "\n"
        "     kaggle competitions download -c ieee-fraud-detection -f train_identity.csv -p " + str(data_root) + "\n"
        "4. Descomprime los .zip en el mismo directorio.\n"
        "5. Verifica con: fraud-adaptive data inspect\n"
        "\n"
        "Los archivos test_* NO se descargan: no tienen etiqueta y el backtest es temporal interno.\n"
        "El token es individual y nunca se versiona.\n"
        "\n"
        "Sin acceso a Kaggle\n"
        "-------------------\n"
        "     fraud-adaptive data surrogate --scale 1.0\n"
        "\n"
        "Genera un dataset SUSTITUTO sintetico con el mismo esquema, eje temporal y\n"
        "prevalencia, con drift de parametros conocidos. Sirve para ejecutar y verificar\n"
        "el pipeline completo, pero NINGUNA de sus cifras describe el fraude real: cada\n"
        "artefacto queda marcado con data_source=sintetico_sustituto.\n"
    )


# --------------------------------------------------------------------------- carga y join

def load_raw(
    sources: dict[str, SourceSpec], *, nrows: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for key in ("transactions", "identity"):
        if not sources[key].exists:
            raise FileNotFoundError(
                "Falta %s en %s.\n%s" % (key, sources[key].path, kaggle_download_instructions(sources[key].path.parent))
            )
    transactions = pd.read_csv(sources["transactions"].path, nrows=nrows)
    identity = pd.read_csv(sources["identity"].path)
    return transactions, identity


def validate_schema(transactions: pd.DataFrame, identity: pd.DataFrame, *, join_key: str, target: str,
                    time_col: str, amount_col: str) -> dict[str, Any]:
    """Comprobaciones de integridad previas a cualquier fit.

    Se ejecutan antes del join porque un duplicado en ``identity`` convertiria un
    left join en many-to-many y multiplicaria filas silenciosamente.
    """
    problems: list[str] = []

    for col in (join_key, target, time_col, amount_col):
        if col not in transactions.columns:
            problems.append("Falta la columna %s en transacciones" % col)
    if join_key not in identity.columns:
        problems.append("Falta la columna %s en identidad" % join_key)
    if problems:
        raise ValueError("Esquema invalido: " + "; ".join(problems))

    if transactions[join_key].duplicated().any():
        problems.append("TransactionID duplicado en transacciones")
    if identity[join_key].duplicated().any():
        problems.append("TransactionID duplicado en identidad: el join dejaria de ser uno-a-uno")

    target_values = set(pd.unique(transactions[target].dropna()))
    if not target_values.issubset({0, 1}):
        problems.append("isFraud no es binaria: %s" % sorted(target_values)[:10])

    amounts = transactions[amount_col]
    n_negative = int((amounts < 0).sum())
    n_nonfinite = int((~np.isfinite(amounts)).sum())
    if n_negative:
        problems.append("%d montos negativos" % n_negative)

    if problems:
        raise ValueError("Validacion de esquema fallida: " + "; ".join(problems))

    return {
        "n_transactions": int(len(transactions)),
        "n_identity": int(len(identity)),
        "n_columns_transactions": int(transactions.shape[1]),
        "n_columns_identity": int(identity.shape[1]),
        "prevalencia": float(transactions[target].mean()),
        "n_fraudes": int(transactions[target].sum()),
        "monto_min": float(amounts.min()),
        "monto_max": float(amounts.max()),
        "monto_mediana": float(amounts.median()),
        "montos_no_finitos": n_nonfinite,
        "dt_min": int(transactions[time_col].min()),
        "dt_max": int(transactions[time_col].max()),
    }


def join_sources(
    transactions: pd.DataFrame, identity: pd.DataFrame, *, join_key: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Left join uno-a-uno que preserva todas las transacciones.

    Devuelve tambien la auditoria del join: cobertura, huerfanos y el conteo
    invariante que prueba que no se multiplicaron filas.
    """
    n_before = len(transactions)
    identity_ids = set(identity[join_key])
    transaction_ids = set(transactions[join_key])

    merged = transactions.merge(identity, on=join_key, how="left", validate="one_to_one")

    if len(merged) != n_before:
        raise RuntimeError(
            "El join altero el numero de filas: %d -> %d. Debe ser uno-a-uno." % (n_before, len(merged))
        )

    identity_cols = [c for c in identity.columns if c != join_key]
    merged["has_identity"] = merged[identity_cols].notna().any(axis=1).astype("int8") if identity_cols else 0

    audit = {
        "filas_antes": n_before,
        "filas_despues": int(len(merged)),
        "filas_invariantes": bool(len(merged) == n_before),
        "ids_identidad": len(identity_ids),
        "ids_identidad_huerfanos": len(identity_ids - transaction_ids),
        "cobertura_identidad": float(merged["has_identity"].mean()),
        "columnas_identidad": len(identity_cols),
        "nota": "La ausencia de identidad es informativa (has_identity), no se imputa ni se descarta la fila.",
    }
    return merged, audit


def derive_time_columns(frame: pd.DataFrame, *, time_col: str) -> pd.DataFrame:
    """Deriva el reloj de evento a partir del delta en segundos.

    ``dia`` es el eje de todo el protocolo temporal. ``hora`` es relativa al origen
    desconocido del dataset: sirve como ciclo de 24 h, no como hora local.
    """
    out = frame.copy()
    origin = int(out[time_col].min())
    seconds = out[time_col].astype("int64") - origin
    out["s_rel"] = seconds
    out["dia"] = (seconds // SECONDS_PER_DAY).astype("int32")
    out["semana"] = (out["dia"] // 7).astype("int32")
    out["hora"] = ((seconds // 3600) % 24).astype("int16")
    # Codificacion ciclica: evita el salto artificial entre la hora 23 y la 0.
    angle = 2.0 * np.pi * out["hora"].to_numpy() / 24.0
    out["hora_sin"] = np.sin(angle)
    out["hora_cos"] = np.cos(angle)
    return out


def order_events(frame: pd.DataFrame, *, time_col: str, join_key: str) -> pd.DataFrame:
    """Orden canonico (tiempo, id).

    El id solo desempata para que la corrida sea reproducible; no introduce
    informacion, porque las features de eventos empatados se calculan contra el
    mismo estado previo (ver features.py).
    """
    return frame.sort_values([time_col, join_key], kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- EDA temporal

def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Intervalo de Wilson para una proporcion.

    Se prefiere a la aproximacion normal porque con prevalencias de ~3.5% y
    semanas de pocos miles de filas el intervalo de Wald puede salirse de [0,1].
    """
    if total == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    margin = z * np.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def weekly_profile(frame: pd.DataFrame, *, target: str, amount_col: str) -> pd.DataFrame:
    """Perfil semanal: volumen, prevalencia con Wilson, montos y cobertura de identidad."""
    rows = []
    for week, group in frame.groupby("semana", sort=True):
        n = len(group)
        fraudes = int(group[target].sum())
        low, high = wilson_interval(fraudes, n)
        rows.append({
            "semana": int(week),
            "dia_min": int(group["dia"].min()),
            "dia_max": int(group["dia"].max()),
            "n": n,
            "fraudes": fraudes,
            "prevalencia": fraudes / n if n else float("nan"),
            "wilson_low": low,
            "wilson_high": high,
            "monto_mediana": float(group[amount_col].median()),
            "monto_p95": float(group[amount_col].quantile(0.95)),
            "monto_total": float(group[amount_col].sum()),
            "cobertura_identidad": float(group["has_identity"].mean()) if "has_identity" in group else float("nan"),
        })
    return pd.DataFrame(rows)


def daily_profile(frame: pd.DataFrame, *, target: str, amount_col: str) -> pd.DataFrame:
    rows = []
    for day, group in frame.groupby("dia", sort=True):
        n = len(group)
        fraudes = int(group[target].sum())
        rows.append({
            "dia": int(day),
            "n": n,
            "fraudes": fraudes,
            "prevalencia": fraudes / n if n else float("nan"),
            "monto_total": float(group[amount_col].sum()),
        })
    return pd.DataFrame(rows)


def missingness_by_family(frame: pd.DataFrame) -> pd.DataFrame:
    """Faltantes agregados por familia de columnas (V, C, D, M, id_, card, addr).

    IEEE-CIS tiene cientos de columnas anonimas; reportarlas una a una no es
    interpretable. La familia es la unidad util para hablar de patrones de captura.
    """
    families: dict[str, list[str]] = {}
    for col in frame.columns:
        if col.startswith("id_"):
            family = "id_"
        elif col.startswith("card"):
            family = "card"
        elif col.startswith("addr"):
            family = "addr"
        elif len(col) > 1 and col[0] in "VCDM" and col[1:].isdigit():
            family = col[0]
        else:
            family = "otras"
        families.setdefault(family, []).append(col)

    rows = []
    for family, cols in sorted(families.items()):
        missing = frame[cols].isna().mean()
        rows.append({
            "familia": family,
            "n_columnas": len(cols),
            "missing_medio": float(missing.mean()),
            "missing_min": float(missing.min()),
            "missing_max": float(missing.max()),
            "columnas_sobre_95pct": int((missing > 0.95).sum()),
        })
    return pd.DataFrame(rows)


def psi(reference: np.ndarray, current: np.ndarray, *, bins: np.ndarray | None = None,
        n_bins: int = 10, epsilon: float = 1e-6) -> float:
    """Population Stability Index.

    Los bordes se toman de la REFERENCIA, nunca del periodo actual: si los bins se
    recalcularan en cada ventana, el indice mediria el rebinning en lugar del
    desplazamiento de la distribucion.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size == 0 or current.size == 0:
        return float("nan")

    if bins is None:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.unique(np.quantile(reference, quantiles))
        if bins.size < 2:
            return 0.0
    bins = np.concatenate(([-np.inf], bins[1:-1], [np.inf]))

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)
    ref_share = ref_counts / max(1, ref_counts.sum())
    cur_share = cur_counts / max(1, cur_counts.sum())
    ref_share = np.clip(ref_share, epsilon, None)
    cur_share = np.clip(cur_share, epsilon, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov de dos muestras.

    Solo para variables continuas. Aplicarlo a codigos nominales (card1, addr1)
    produciria un estadistico que depende del orden arbitrario de los codigos.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size < 2 or current.size < 2:
        return (float("nan"), float("nan"))
    result = stats.ks_2samp(reference, current)
    return (float(result.statistic), float(result.pvalue))


def benjamini_hochberg(pvalues: Sequence[float], q: float = 0.05) -> np.ndarray:
    """Correccion BH.

    Con paneles de decenas de variables comparadas cada semana, sin correccion el
    numero esperado de falsos positivos convierte cualquier alerta en ruido.
    """
    values = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(values)
    rejected = np.zeros(values.shape, dtype=bool)
    if not finite.any():
        return rejected
    subset = values[finite]
    order = np.argsort(subset)
    ranked = subset[order]
    m = ranked.size
    thresholds = q * np.arange(1, m + 1) / m
    passing = ranked <= thresholds
    if passing.any():
        cut = np.max(np.where(passing)[0])
        keep = np.zeros(m, dtype=bool)
        keep[order[: cut + 1]] = True
        rejected[np.where(finite)[0]] = keep
    return rejected


def drift_panel(
    frame: pd.DataFrame,
    reference_mask: np.ndarray,
    current_mask: np.ndarray,
    columns: Sequence[str],
    *,
    q: float = 0.05,
) -> pd.DataFrame:
    """KS y PSI por variable entre una referencia fija y un periodo actual.

    Reporta efecto (D, PSI) junto a significacion: con cientos de miles de filas
    un p-valor diminuto acompana diferencias irrelevantes, asi que el tamano del
    efecto es lo que decide la alerta.
    """
    rows = []
    for col in columns:
        if col not in frame.columns:
            continue
        ref = frame.loc[reference_mask, col].to_numpy(dtype=float, na_value=np.nan)
        cur = frame.loc[current_mask, col].to_numpy(dtype=float, na_value=np.nan)
        d_stat, p_value = ks_test(ref, cur)
        rows.append({
            "variable": col,
            "ks_d": d_stat,
            "ks_p": p_value,
            "psi": psi(ref, cur),
            "n_ref": int(np.isfinite(ref).sum()),
            "n_cur": int(np.isfinite(cur).sum()),
            "missing_ref": float(np.mean(~np.isfinite(ref))) if ref.size else float("nan"),
            "missing_cur": float(np.mean(~np.isfinite(cur))) if cur.size else float("nan"),
        })
    panel = pd.DataFrame(rows)
    if not panel.empty:
        panel["ks_significativo_bh"] = benjamini_hochberg(panel["ks_p"].to_numpy(), q=q)
    return panel


def select_monitor_panel(frame: pd.DataFrame, candidate_columns: Sequence[str], *, max_columns: int = 20) -> list[str]:
    """Elige el panel de monitoreo por completitud y varianza, usando solo desarrollo.

    Fijarlo en desarrollo evita que el panel cambie cuando cambia el periodo
    observado, que haria incomparables las alertas entre bloques.
    """
    scored: list[tuple[float, str]] = []
    for col in candidate_columns:
        if col not in frame.columns:
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        completeness = float(series.notna().mean())
        if completeness < 0.5:
            continue
        variance = float(series.std(skipna=True) or 0.0)
        if not np.isfinite(variance) or variance <= 0:
            continue
        scored.append((completeness, col))
    scored.sort(reverse=True)
    return [col for _, col in scored[:max_columns]]


def detect_anomalies(frame: pd.DataFrame, *, amount_col: str, target: str) -> dict[str, Any]:
    """Anomalias temporales y de monto que deben verse antes de culpar al drift.

    Un hueco de captura o un pico de duplicados explican una alerta mejor que un
    cambio de comportamiento; el plan exige descartarlos primero.
    """
    daily = frame.groupby("dia").size()
    expected_days = set(range(int(frame["dia"].min()), int(frame["dia"].max()) + 1))
    missing_days = sorted(expected_days - set(daily.index.astype(int)))
    median_volume = float(daily.median())

    amounts = frame[amount_col]
    q99 = float(amounts.quantile(0.99))

    duplicated_cols = [c for c in (amount_col, "dia", "card1") if c in frame.columns]

    return {
        "dias_cubiertos": int(daily.size),
        "dias_faltantes": missing_days[:20],
        "n_dias_faltantes": len(missing_days),
        "volumen_diario_mediano": median_volume,
        "dias_volumen_bajo": [int(d) for d in daily[daily < 0.3 * median_volume].index.tolist()][:20],
        "dias_volumen_alto": [int(d) for d in daily[daily > 3.0 * median_volume].index.tolist()][:20],
        "monto_p99": q99,
        "n_montos_extremos": int((amounts > 10 * q99).sum()),
        "n_duplicados_exactos": int(frame.duplicated(subset=duplicated_cols).sum()) if duplicated_cols else 0,
        "prevalencia_global": float(frame[target].mean()),
    }


def write_sources_manifest(path: str | Path, sources: dict[str, SourceSpec], **extra: Any) -> Path:
    return write_json(path, build_sources_manifest(sources, extra=extra))
