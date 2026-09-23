"""Generador de un dataset sustituto con la forma de IEEE-CIS.

IEEE-CIS requiere aceptar las reglas de la competencia y un token de Kaggle, y no
se versiona en el repositorio. Este generador produce tablas con el mismo esquema,
patrones de faltantes parecidos y el mismo eje temporal, con drift inyectado de
parametros conocidos. Sirve para:

1. verificar que el pipeline corre de punta a punta y que los controles de fuga
   funcionan;
2. comprobar que los detectores encuentran un drift conocido.

Las cifras obtenidas sobre estos datos no describen IEEE-CIS. Cada artefacto
derivado queda marcado con ``data_source = "sintetico_sustituto"`` y las tablas y
figuras llevan el rotulo.

El drift inyectado separa dos fenomenos:

* covariate shift: la mezcla de productos y dominios de correo cambia de forma
  gradual entre ``drift_start_day`` y ``drift_end_day``;
* concept drift: el vector de coeficientes que genera la etiqueta rota a velocidad
  constante, de modo que P(y|X) cambia con el tiempo.

Asi se puede comprobar que S1 reacciona al primero y ADWIN sobre el Brier al segundo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("fraud_adaptive.synthetic")

SECONDS_PER_DAY = 86_400

DATA_SOURCE_TAG = "sintetico_sustituto"

EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com",
    "outlook.com", "comcast.net", "icloud.com", "live.com", "msn.com",
]
PRODUCT_CODES = ["W", "C", "R", "H", "S"]
DEVICE_TYPES = ["desktop", "mobile"]
DEVICE_INFO = ["Windows", "iOS Device", "MacOS", "Trident/7.0", "Android", "SAMSUNG", "rv:11.0"]
M_LEVELS = ["T", "F"]


@dataclass
class SyntheticConfig:
    """Parametros del sustituto. Los valores por defecto imitan IEEE-CIS."""

    n_days: int = 182
    transactions_per_day: int = 3_200
    base_prevalence: float = 0.035
    identity_coverage: float = 0.24
    n_v_columns: int = 339
    n_c_columns: int = 14
    n_d_columns: int = 15
    n_m_columns: int = 9
    n_id_columns: int = 38
    n_cards: int = 14_000
    n_devices: int = 6_000
    seed: int = 42
    # Concept drift: rotacion de angulo constante del vector generador.
    # 0.0105 rad/dia => correlacion cos(w*d) de 0.95 a 30 dias, 0.82 a 60,
    # 0.61 a 90 y 0.30 a 120. Un modelo con 30 dias de antiguedad conserva casi
    # toda la senal; uno de 120 dias ha perdido dos tercios.
    drift_rate_rad_per_day: float = 0.0105
    signal_norm: float = 3.4
    # Covariate shift (P(X)): independiente del anterior, para poder comprobar que
    # S1 reacciona al cambio de covariables y ADWIN al cambio de P(y|X).
    drift_start_day: int = 60
    drift_end_day: int = 170
    scale: float = 1.0
    # Episodios de tarjeta comprometida (card testing): dan senal a las features
    # de velocidad por entidad.
    compromised_card_fraction: float = 0.06
    compromise_window_days: float = 3.0
    compromise_logit_boost: float = 2.6
    velocity_logit_weight: float = 1.5
    signal_columns: int = 12
    weekly_seasonality: float = 0.18
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_per_day(self) -> int:
        return max(50, int(self.transactions_per_day * self.scale))


def _drift_weight(day: np.ndarray, config: SyntheticConfig) -> np.ndarray:
    """Peso del segundo regimen en [0,1]: 0 antes del drift, 1 despues.

    La transicion es suave (smoothstep), mas parecida a un cambio de comportamiento
    que un escalon.
    """
    span = max(1, config.drift_end_day - config.drift_start_day)
    t = np.clip((day - config.drift_start_day) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def generate(config: SyntheticConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera ``(train_transaction, train_identity)`` sustitutos.

    Devuelve dos DataFrames con el esquema de Kaggle, listos para escribirse como
    CSV y consumirse por el mismo codigo que consume los datos reales.
    """
    config = config or SyntheticConfig()
    rng = np.random.default_rng(config.seed)

    per_day = config.effective_per_day
    n_days = config.n_days

    # --- eje temporal -------------------------------------------------------
    # El volumen diario tiene estacionalidad semanal y ruido, como el dataset real.
    day_index = np.arange(n_days)
    weekly = 1.0 + config.weekly_seasonality * np.sin(2 * np.pi * day_index / 7.0)
    counts = rng.poisson(per_day * weekly).clip(min=10)
    total = int(counts.sum())

    day = np.repeat(day_index, counts)
    # Hora del dia con dos modas (actividad diurna), no uniforme.
    hour_component = rng.beta(2.2, 2.0, size=total) * 24.0
    seconds_in_day = np.clip(hour_component, 0, 23.999) * 3600.0 + rng.uniform(0, 3600, size=total)
    transaction_dt = (day * SECONDS_PER_DAY + seconds_in_day).astype(np.int64)
    # Origen arbitrario distinto de cero, como en el dataset real.
    transaction_dt = transaction_dt + 86_400
    order = np.argsort(transaction_dt, kind="mergesort")
    transaction_dt = transaction_dt[order]
    day = day[order]

    transaction_id = np.arange(2_987_000, 2_987_000 + total, dtype=np.int64)

    # --- entidades ----------------------------------------------------------
    #
    # Primero se crea un conjunto de tarjetas con atributos fijos y despues se
    # sortea que tarjeta hace cada transaccion. Si card2, card5 y addr1 se sortearan
    # por fila, casi todas las claves de tarjeta serian unicas y el historial
    # quedaria vacio; en IEEE-CIS esas columnas describen la tarjeta.
    n_cards = config.n_cards
    card_pool = {
        "card1": (rng.integers(1000, 19000, size=n_cards)).astype(np.int32),
        "card2": rng.integers(100, 600, size=n_cards).astype(float),
        "card3": np.where(rng.random(n_cards) < 0.92, 150.0, 185.0),
        "card4": rng.choice(["visa", "mastercard", "american express", "discover"],
                            size=n_cards, p=[0.62, 0.32, 0.04, 0.02]),
        "card5": rng.integers(100, 240, size=n_cards).astype(float),
        "card6": rng.choice(["debit", "credit"], size=n_cards, p=[0.74, 0.26]),
        "addr1": rng.integers(100, 540, size=n_cards).astype(float),
        "addr2": np.where(rng.random(n_cards) < 0.95, 87.0, 60.0),
    }

    # Popularidad desigual: unas pocas tarjetas concentran muchas transacciones y
    # hay una cola larga de tarjetas con uno o dos eventos, como en el real.
    popularity = rng.zipf(1.25, size=n_cards).astype(float)
    popularity = np.clip(popularity, 1, 500)
    popularity = popularity / popularity.sum()
    card_index = rng.choice(n_cards, size=total, p=popularity)

    card1 = card_pool["card1"][card_index]
    card2 = card_pool["card2"][card_index]
    card3 = card_pool["card3"][card_index]
    card4 = card_pool["card4"][card_index]
    card5 = card_pool["card5"][card_index]
    card6 = card_pool["card6"][card_index]
    addr1 = card_pool["addr1"][card_index]
    addr2 = card_pool["addr2"][card_index]

    device_id = (rng.zipf(1.5, size=total) % config.n_devices).astype(np.int32)

    # --- covariate shift ----------------------------------------------------
    # La mezcla de productos y dominios de correo deriva con el tiempo (cambio en
    # P(X)), que S1 y KS/PSI deben detectar.
    drift_w = _drift_weight(day, config)
    product_probs_early = np.array([0.74, 0.12, 0.06, 0.05, 0.03])
    product_probs_late = np.array([0.52, 0.26, 0.10, 0.08, 0.04])
    product_cd = np.empty(total, dtype=object)
    uniform_draw = rng.random(total)
    for i_prod in range(len(PRODUCT_CODES)):
        probs = (1 - drift_w) * product_probs_early[i_prod] + drift_w * product_probs_late[i_prod]
        lower = np.zeros(total)
        for j in range(i_prod):
            lower += (1 - drift_w) * product_probs_early[j] + drift_w * product_probs_late[j]
        mask = (uniform_draw >= lower) & (uniform_draw < lower + probs)
        product_cd[mask] = PRODUCT_CODES[i_prod]
    product_cd[pd.isna(product_cd)] = PRODUCT_CODES[0]

    email_probs_early = np.array([0.34, 0.18, 0.14, 0.12, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02])
    email_probs_late = np.array([0.28, 0.14, 0.10, 0.20, 0.05, 0.09, 0.04, 0.05, 0.03, 0.02])
    p_email = np.empty(total, dtype=object)
    draw_email = rng.random(total)
    cumulative = np.zeros(total)
    for idx, domain in enumerate(EMAIL_DOMAINS):
        probs = (1 - drift_w) * email_probs_early[idx] + drift_w * email_probs_late[idx]
        mask = (draw_email >= cumulative) & (draw_email < cumulative + probs)
        p_email[mask] = domain
        cumulative = cumulative + probs
    p_email[pd.isna(p_email)] = EMAIL_DOMAINS[0]
    r_email = np.where(rng.random(total) < 0.24, p_email, None)

    # --- monto --------------------------------------------------------------
    # Lognormal con media que deriva ligeramente: captura inflacion/mix de canal.
    amount_mu = 3.6 + 0.22 * drift_w
    amount = np.round(rng.lognormal(amount_mu, 1.05, size=total), 3)
    amount = np.clip(amount, 0.251, 32_000.0)

    # --- panel numerico -----------------------------------------------------
    # C: conteos (enteros no negativos, muy sesgados)
    c_cols = {}
    for i in range(1, config.n_c_columns + 1):
        base = rng.gamma(shape=1.1 + 0.1 * i, scale=1.4 + 0.3 * drift_w, size=total)
        c_cols["C%d" % i] = np.floor(base).astype(float)

    # D: deltas temporales, con faltantes crecientes por indice (como el real)
    d_cols = {}
    for i in range(1, config.n_d_columns + 1):
        values = rng.exponential(scale=30.0 + 4.0 * i, size=total)
        missing_rate = min(0.90, 0.05 + 0.055 * i)
        values[rng.random(total) < missing_rate] = np.nan
        d_cols["D%d" % i] = np.round(values, 1)

    # M: banderas categoricas T/F con faltantes
    m_cols = {}
    for i in range(1, config.n_m_columns + 1):
        values = rng.choice(M_LEVELS, size=total, p=[0.62, 0.38]).astype(object)
        values[rng.random(total) < 0.15 + 0.03 * i] = None
        m_cols["M%d" % i] = values

    # V: panel anonimo ancho y muy incompleto; unas pocas columnas llevan senal
    v_cols = {}
    signal_idx = set(range(1, config.signal_columns + 1))
    for i in range(1, config.n_v_columns + 1):
        if i in signal_idx:
            values = rng.normal(0.0, 1.0, size=total)
            missing_rate = 0.02
        else:
            values = rng.normal(0.0, 1.0, size=total)
            # Bloques de V con faltantes correlacionados, igual que en IEEE-CIS.
            missing_rate = 0.12 if i < 120 else (0.47 if i < 240 else 0.76)
        values[rng.random(total) < missing_rate] = np.nan
        v_cols["V%d" % i] = np.round(values, 4)

    # --- etiqueta con concept drift ----------------------------------------
    #
    # El vector generador rota a velocidad angular constante en el plano de dos
    # vectores ortonormales:
    #
    #     beta(t) = norma * [cos(w*t) * b1 + sin(w*t) * b2]
    #
    #   * ``signal_norm`` fija cuanta senal hay y no cambia con el tiempo;
    #   * ``drift_rate`` fija la velocidad del cambio: la correlacion entre beta(t)
    #     y beta(t+d) es cos(w*d), independiente del instante.
    #
    # Asi hay un compromiso entre ventanas: W30 gana en frescura y W90 en volumen.
    signal_names = ["V%d" % i for i in sorted(signal_idx)]
    signal_matrix = np.column_stack([np.nan_to_num(v_cols[name], nan=0.0) for name in signal_names])
    n_signal = len(signal_names)

    raw_a = rng.normal(0.0, 1.0, size=n_signal)
    raw_b = rng.normal(0.0, 1.0, size=n_signal)
    basis_a = raw_a / np.linalg.norm(raw_a)
    # Gram-Schmidt: b2 ortogonal a b1, para que la rotacion sea pura y la
    # correlacion decaiga exactamente como el coseno del angulo.
    proj = raw_b - np.dot(raw_b, basis_a) * basis_a
    basis_b = proj / np.linalg.norm(proj)

    angle = config.drift_rate_rad_per_day * day.astype(float)
    beta_mix = config.signal_norm * (
        np.cos(angle)[:, None] * basis_a[None, :] + np.sin(angle)[:, None] * basis_b[None, :]
    )
    logit = np.einsum("ij,ij->i", signal_matrix, beta_mix)

    # Contribuciones estables: una parte de la senal no caduca y el modelo
    # estatico se degrada sin caer a cero.
    logit = logit + 0.45 * (np.log1p(amount) - float(np.mean(np.log1p(amount))))
    logit = logit + np.where(product_cd == "C", 0.90, 0.0)
    logit = logit + np.where(p_email == "anonymous.com", 1.05, 0.0)
    logit = logit + 0.30 * np.sin(2 * np.pi * (transaction_dt % SECONDS_PER_DAY) / SECONDS_PER_DAY)

    has_identity_flag = rng.random(total) < config.identity_coverage
    logit = logit + np.where(has_identity_flag, -0.25, 0.20)

    # --- episodios de tarjeta comprometida ---------------------------------
    # Un subconjunto de tarjetas concentra fraude durante una ventana corta (card
    # testing), lo que da senal a las features de velocidad por entidad.
    n_compromised = max(1, int(n_cards * config.compromised_card_fraction))
    compromised = rng.choice(n_cards, size=n_compromised, replace=False)
    episode_start = rng.uniform(0, max(1, n_days - config.compromise_window_days), size=n_compromised)
    start_by_card = np.full(n_cards, np.inf)
    start_by_card[compromised] = episode_start
    card_start = start_by_card[card_index]
    in_episode = (day >= card_start) & (day < card_start + config.compromise_window_days)
    logit = logit + np.where(in_episode, config.compromise_logit_boost, 0.0)

    # Velocidad reciente por tarjeta con el pasado estricto, con la misma semantica
    # que features.py: ventana semiabierta [t-w, t).
    velocity_24h = np.zeros(total, dtype=float)
    order_by_card = np.argsort(card_index, kind="mergesort")
    sorted_cards = card_index[order_by_card]
    boundaries = np.searchsorted(sorted_cards, np.arange(n_cards + 1))
    for card in range(n_cards):
        lo, hi = boundaries[card], boundaries[card + 1]
        if hi - lo < 2:
            continue
        rows = order_by_card[lo:hi]
        card_times = transaction_dt[rows]  # ya ordenados por tiempo global
        # Eventos estrictamente anteriores menos los anteriores a t-24h.
        before = np.searchsorted(card_times, card_times, side="left")
        window_start = np.searchsorted(card_times, card_times - SECONDS_PER_DAY, side="left")
        velocity_24h[rows] = before - window_start
    logit = logit + config.velocity_logit_weight * np.log1p(velocity_24h)

    # El intercepto se calibra por dia para mantener la prevalencia casi constante.
    # Con un intercepto global, la rotacion arrastraba P(y) de 4.7% a 0.8% entre
    # bloques. Con la prevalencia fija, lo que cambia es que combinaciones de X
    # predicen y, y un modelo estatico se degrada con una tasa de fraude estable.
    target = config.base_prevalence
    probability = np.empty(total, dtype=float)
    # Estacionalidad suave de la prevalencia, para no fabricar una serie plana.
    seasonal = 1.0 + 0.10 * np.sin(2 * np.pi * day_index / 28.0)
    for current_day in range(n_days):
        mask = day == current_day
        if not mask.any():
            continue
        day_logit = logit[mask]
        day_target = float(np.clip(target * seasonal[current_day], 1e-4, 0.5))
        lo, hi = -30.0, 30.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            rate = float(np.mean(1.0 / (1.0 + np.exp(-(day_logit + mid)))))
            if rate > day_target:
                hi = mid
            else:
                lo = mid
        probability[mask] = 1.0 / (1.0 + np.exp(-(day_logit + 0.5 * (lo + hi))))
    is_fraud = (rng.random(total) < probability).astype(np.int8)

    # --- ensamblado de transacciones ---------------------------------------
    transactions = pd.DataFrame({
        "TransactionID": transaction_id,
        "isFraud": is_fraud,
        "TransactionDT": transaction_dt,
        "TransactionAmt": amount,
        "ProductCD": product_cd,
        "card1": card1,
        "card2": card2,
        "card3": card3,
        "card4": card4,
        "card5": card5,
        "card6": card6,
        "addr1": addr1,
        "addr2": addr2,
        "dist1": np.where(rng.random(total) < 0.40, rng.exponential(60, total).round(1), np.nan),
        "dist2": np.where(rng.random(total) < 0.07, rng.exponential(230, total).round(1), np.nan),
        "P_emaildomain": p_email,
        "R_emaildomain": r_email,
    })
    # Un solo concat evita fragmentar el frame con cientos de asignaciones.
    panel = {}
    panel.update(c_cols)
    panel.update(d_cols)
    panel.update(m_cols)
    panel.update(v_cols)
    transactions = pd.concat([transactions, pd.DataFrame(panel, index=transactions.index)], axis=1)

    # --- identidad ----------------------------------------------------------
    # Cobertura parcial: solo un subconjunto de TransactionID tiene identidad.
    identity_mask = has_identity_flag
    n_identity = int(identity_mask.sum())
    identity = pd.DataFrame({"TransactionID": transaction_id[identity_mask]})
    for i in range(1, config.n_id_columns + 1):
        name = "id_%02d" % i
        if i <= 11:
            values = rng.normal(0, 1, size=n_identity).round(3)
            values[rng.random(n_identity) < 0.08] = np.nan
            identity[name] = values
        else:
            levels = ["Found", "NotFound", "New", "Unknown"]
            values = rng.choice(levels, size=n_identity, p=[0.42, 0.34, 0.16, 0.08]).astype(object)
            values[rng.random(n_identity) < 0.22] = None
            identity[name] = values
    # DeviceInfo lleva familia y build, como en el real ("SM-G950F Build/NRD90M");
    # con solo siete valores el proxy de dispositivo agruparia demasiados eventos.
    device_family = np.array(DEVICE_INFO, dtype=object)[device_id[identity_mask] % len(DEVICE_INFO)]
    device_build = (device_id[identity_mask] % 400).astype(str)
    identity["DeviceType"] = np.where(
        np.isin(device_family, ["iOS Device", "Android", "SAMSUNG"]), "mobile", "desktop"
    )
    identity["DeviceInfo"] = pd.Series(device_family).str.cat(
        pd.Series(device_build), sep=" Build/"
    ).to_numpy()

    LOGGER.info(
        "Sustituto generado: %d filas, %d dias, prevalencia %.4f, identidad %.3f",
        total, n_days, float(is_fraud.mean()), float(identity_mask.mean()),
    )
    return transactions, identity


def write_surrogate(
    output_dir: str | Path, config: SyntheticConfig | None = None
) -> dict[str, Any]:
    """Escribe los CSV sustitutos y un aviso permanente junto a ellos."""
    config = config or SyntheticConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transactions, identity = generate(config)
    tx_path = output_dir / "train_transaction.csv"
    id_path = output_dir / "train_identity.csv"
    transactions.to_csv(tx_path, index=False)
    identity.to_csv(id_path, index=False)

    notice = (
        "DATOS SUSTITUTOS, NO SON IEEE-CIS\n"
        "=================================\n\n"
        "Estos CSV fueron generados por src/fraud_adaptive/synthetic.py para poder\n"
        "ejecutar y verificar el pipeline completo en una maquina sin acceso al\n"
        "dataset real de Kaggle.\n\n"
        "Ninguna metrica calculada sobre estos archivos describe el fraude real ni\n"
        "puede citarse como resultado de IEEE-CIS Fraud Detection.\n\n"
        "Parametros del generador:\n"
        "  seed                = %d\n"
        "  n_days              = %d\n"
        "  transactions_per_day= %d (scale=%.3f)\n"
        "  prevalencia objetivo= %.4f\n"
        "  drift P(y|X)        = rotacion de %.4f rad/dia, norma de senal %.2f\n"
        "  drift P(X)          = dias %d a %d (transicion suave)\n\n"
        "Para reemplazarlos por los datos reales, sigue las instrucciones de\n"
        "  fraud-adaptive data instructions\n"
        "y vuelve a ejecutar el pipeline: los comandos son identicos.\n"
    ) % (
        config.seed, config.n_days, config.transactions_per_day, config.scale,
        config.base_prevalence, config.drift_rate_rad_per_day, config.signal_norm,
        config.drift_start_day, config.drift_end_day,
    )
    (output_dir / "LEEME_DATOS_SUSTITUTOS.txt").write_text(notice, encoding="utf-8")

    return {
        "data_source": DATA_SOURCE_TAG,
        "transactions_path": tx_path.as_posix(),
        "identity_path": id_path.as_posix(),
        "n_transactions": int(len(transactions)),
        "n_identity": int(len(identity)),
        "n_days": config.n_days,
        "prevalencia": float(transactions["isFraud"].mean()),
        "cobertura_identidad": float(len(identity) / len(transactions)),
        "drift_rate_rad_per_day": config.drift_rate_rad_per_day,
        "signal_norm": config.signal_norm,
        "drift_start_day": config.drift_start_day,
        "drift_end_day": config.drift_end_day,
        "seed": config.seed,
        "advertencia": "Resultados sobre estos datos no representan IEEE-CIS.",
    }


def make_adwin_streams(n_obs: int = 2000, seed: int = 42, jump_at: int = 1000) -> dict[str, np.ndarray]:
    """Dos streams de perdidas acotadas para probar ADWIN contra verdad conocida.

    Miden el retardo de deteccion y las falsas alarmas del detector.
    """
    rng = np.random.default_rng(seed)
    stationary = rng.beta(2.0, 8.0, size=n_obs)
    jump = np.concatenate([
        rng.beta(2.0, 8.0, size=jump_at),
        rng.beta(6.0, 4.0, size=n_obs - jump_at),
    ])
    return {"stationary": stationary, "jump": jump, "jump_at": np.array([jump_at])}
