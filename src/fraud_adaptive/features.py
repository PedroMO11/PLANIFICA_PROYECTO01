"""Features causales de historial por entidad, con correccion point-in-time.

La regla que gobierna este modulo
---------------------------------
Las features de un evento se calculan con el estado ANTERIOR a ese evento y se
emiten ANTES de incorporarlo al historial. Nunca se usa la fila actual ni ningun
evento posterior.

El caso que suele romper una implementacion ingenua son los EMPATES de tiempo. Si
dos transacciones comparten ``TransactionDT``, procesarlas en secuencia haria que
la segunda viera a la primera, y su feature dependeria del orden arbitrario de
desempate. Aqui se procesan por grupos de tiempo identico: todo el grupo lee el
mismo pasado y solo despues se actualiza el estado. Eso hace que permutar los IDs
empatados deje las features bit a bit iguales, que es lo que verifica
``tests/test_point_in_time_features.py``.

Entidades
---------
IEEE-CIS esta anonimizado y no trae un identificador de cliente. Se construyen tres
PROXIES y se mide su calidad en lugar de asumirla:

* ``card_proxy``: card1..card6 + addr1. Clave gruesa, por emisor.
* ``cliente_proxy``: card1 + addr1 + D1n, con ``D1n`` el dia de alta de la
  tarjeta. Clave fina, por tarjeta individual.
* ``device_proxy``: DeviceType + DeviceInfo (solo con identidad presente).

Las dos claves de tarjeta conviven porque resuelven la misma entidad a granularidad
distinta y los historiales resultantes son complementarios, no redundantes.

Un proxy no es una persona. Dos clientes pueden colisionar en la misma clave y un
cliente puede aparecer con varias. Por eso ``proxy_quality`` reporta cardinalidad
y tasa de colision, y el informe habla de proxies, no de clientes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("fraud_adaptive.features")

SECONDS_PER_DAY = 86_400

# Ventanas de historial hacia atras, en segundos. Todas semiabiertas [t-w, t).
HISTORY_WINDOWS: dict[str, int] = {
    "1h": 3_600,
    "24h": 86_400,
    "7d": 7 * 86_400,
}

# Columnas que el predictor NUNCA puede ver.
#
# TransactionID memorizaria filas concretas. TransactionDT, s_rel, dia y semana son
# tiempo absoluto: un arbol podria partir en "dia > 120" y aprender el bloque de
# test en lugar del fenomeno. La hora ciclica si se permite porque es estacional y
# se repite, no identifica un instante del calendario.
FORBIDDEN_PREDICTOR_COLUMNS: frozenset[str] = frozenset({
    "TransactionID", "TransactionDT", "s_rel", "dia", "semana", "hora",
    "isFraud", "available_at", "card_proxy", "cliente_proxy", "device_proxy",
})

# Sufijos prohibidos para el predictor.
#
# Un conteo expansivo acumula desde el primer evento, de modo que crece de forma
# monotona con el calendario y funciona como sustituto del dia absoluto. Medido
# sobre IEEE-CIS, la media de `card_cnt_expansivo` pasa de 138,8 en [0,60) a 707,4
# en [150,182), con una correlacion de +0,23 con el dia. Una validacion adversarial
# por feature le asigna AUC 0,71, el valor mas alto de las 425 columnas del panel.
#
# Dejarlo dentro del predictor contradice la prohibicion de tiempo absoluto y hace
# que la comparacion entre ventanas dependa de cuan lejos quede el rango de
# entrenamiento del rango de test. Las ventanas acotadas de 1 h, 24 h y 7 d no
# tienen el problema: la correlacion con el dia es de -0,009 para `card_cnt_7d`.
#
# Las columnas se siguen calculando porque son utiles como diagnostico.
FORBIDDEN_PREDICTOR_SUFFIXES: tuple[str, ...] = ("_expansivo",)


# --------------------------------------------------------------------------- proxies

def build_card_proxy(frame: pd.DataFrame) -> pd.Series:
    """Clave de emisor, encadenando ``card1`` a ``card6`` mas ``addr1``.

    Agrupa por atributos del medio de pago: numero de emisor, tipo, red y region.
    Es una clave GRUESA. Sobre IEEE-CIS produce 43 018 entidades con 13,7 eventos
    de media, de modo que sus ventanas de 24 h y 7 d casi siempre tienen soporte.
    A cambio, una misma clave mezcla tarjetas distintas del mismo emisor.
    """
    columns = [c for c in ("card1", "card2", "card3", "card4", "card5", "card6", "addr1")
               if c in frame.columns]
    if not columns:
        return pd.Series(["desconocida"] * len(frame), index=frame.index, dtype="object")
    key = frame[columns[0]].astype("string").fillna("na")
    for column in columns[1:]:
        key = key + "|" + frame[column].astype("string").fillna("na")
    return key.astype("object")


def build_client_proxy(frame: pd.DataFrame, *, time_col: str = "TransactionDT") -> pd.Series:
    """Clave de cliente, construida como ``card1 + addr1 + D1n``.

    ``D1`` mide los dias transcurridos desde que la tarjeta empezo a usarse, de modo
    que ``D1n = dia - D1`` identifica el dia de alta. Dos transacciones con el mismo
    ``card1``, el mismo ``addr1`` y la misma fecha de alta pertenecen casi siempre a
    la misma tarjeta. Es la clave FINA: 217 850 entidades con 2,71 eventos de media.

    Las dos claves no compiten, se complementan. Por separado rinden igual, con AP
    de 0,1668 y 0,1671 sobre un test temporal usando solo features de historial.
    Juntas rinden 0,1829, porque la gruesa aporta soporte estadistico y la fina
    aporta resolucion por tarjeta. Por eso el panel emite ambos historiales.

    ``D1n`` no introduce tiempo absoluto en el predictor. Es una constante por
    tarjeta, no una funcion creciente del calendario, y ademas solo se usa para
    agrupar: la clave nunca entra al modelo.

    El dia se deriva aqui de ``TransactionDT`` y no de la columna ``dia``. Esta
    ultima se mide desde el primer evento del frame, de modo que una misma tarjeta
    recibiria claves distintas segun el rango que se este procesando. El dia
    absoluto es estable entre llamadas y produce exactamente la misma agrupacion.
    """
    if "card1" not in frame.columns:
        return pd.Series(["desconocida"] * len(frame), index=frame.index, dtype="object")

    partes = [frame["card1"].astype("string").fillna("na")]
    if "addr1" in frame.columns:
        partes.append(frame["addr1"].astype("string").fillna("na"))

    if "D1" in frame.columns and time_col in frame.columns:
        dia_absoluto = np.floor(
            pd.to_numeric(frame[time_col], errors="coerce") / SECONDS_PER_DAY)
        d1n = dia_absoluto - pd.to_numeric(frame["D1"], errors="coerce")
        partes.append(d1n.round().astype("Int64").astype("string").fillna("na"))

    key = partes[0]
    for parte in partes[1:]:
        key = key + "|" + parte
    return key.astype("object")


def build_device_proxy(frame: pd.DataFrame) -> pd.Series:
    """Clave de dispositivo. Sin identidad no hay entidad: queda ``None``.

    Marcarla como ausente y no como una categoria propia evita que todas las
    transacciones sin identidad compartan un historial comun inexistente.
    """
    columns = [c for c in ("DeviceType", "DeviceInfo") if c in frame.columns]
    if not columns:
        return pd.Series([None] * len(frame), index=frame.index, dtype="object")
    parts = [frame[col].astype("string") for col in columns]
    key = parts[0].fillna("na")
    for part in parts[1:]:
        key = key + "|" + part.fillna("na")
    key = key.astype("object")
    if "has_identity" in frame.columns:
        key = key.where(frame["has_identity"].astype(bool), other=None)
    return key


def proxy_quality(frame: pd.DataFrame, proxy_column: str) -> dict[str, Any]:
    """Cardinalidad y concentracion del proxy.

    Se reporta para poder decir en el informe cuanto se parece el proxy a una
    entidad real, en vez de afirmarlo.
    """
    series = frame[proxy_column].dropna()
    if series.empty:
        return {"proxy": proxy_column, "n_entidades": 0, "cobertura": 0.0}
    counts = series.value_counts()
    return {
        "proxy": proxy_column,
        "n_entidades": int(counts.size),
        "cobertura": float(len(series) / len(frame)),
        "eventos_por_entidad_medio": float(counts.mean()),
        "eventos_por_entidad_p95": float(counts.quantile(0.95)),
        "entidades_con_un_evento": int((counts == 1).sum()),
        "fraccion_singleton": float((counts == 1).mean()),
        "top1_share": float(counts.iloc[0] / len(series)),
        "nota": "Un proxy agrupa comportamientos, no identifica personas; puede colisionar.",
    }


# --------------------------------------------------------------------------- estado causal

@dataclass
class _EntityState:
    """Historial append-only de una entidad, con punteros por ventana.

    Los eventos llegan en orden temporal, asi que cada puntero solo avanza: el
    costo amortizado por evento es O(1) y no O(n) como lo seria re-escanear la
    ventana en cada fila.
    """

    times: list[float] = field(default_factory=list)
    prefix_amount: list[float] = field(default_factory=lambda: [0.0])
    pointers: dict[str, int] = field(default_factory=dict)

    def snapshot(self, now: float) -> dict[str, float]:
        """Agregados del pasado estricto de ``now``. No incluye el evento actual."""
        total = len(self.times)
        out: dict[str, float] = {}

        for name, span in HISTORY_WINDOWS.items():
            pointer = self.pointers.get(name, 0)
            limit = now - span
            # Avanza mientras el evento sea mas antiguo que el borde de la ventana.
            while pointer < total and self.times[pointer] < limit:
                pointer += 1
            self.pointers[name] = pointer
            count = total - pointer
            amount = self.prefix_amount[total] - self.prefix_amount[pointer]
            out["cnt_" + name] = float(count)
            out["amt_" + name] = float(amount)
            out["amt_medio_" + name] = float(amount / count) if count else np.nan

        out["cnt_expansivo"] = float(total)
        out["amt_expansivo"] = float(self.prefix_amount[total])
        if total:
            out["amt_rezago"] = float(self.prefix_amount[total] - self.prefix_amount[total - 1])
            out["segundos_desde_ultimo"] = float(now - self.times[-1])
        else:
            out["amt_rezago"] = np.nan
            out["segundos_desde_ultimo"] = np.nan
        return out

    def push(self, now: float, amount: float) -> None:
        self.times.append(now)
        self.prefix_amount.append(self.prefix_amount[-1] + float(amount))


_FEATURE_SUFFIXES = (
    [("cnt_" + w) for w in HISTORY_WINDOWS]
    + [("amt_" + w) for w in HISTORY_WINDOWS]
    + [("amt_medio_" + w) for w in HISTORY_WINDOWS]
    + ["cnt_expansivo", "amt_expansivo", "amt_rezago", "segundos_desde_ultimo"]
)


def history_feature_names(prefixes: Iterable[str]) -> list[str]:
    return [prefix + "_" + suffix for prefix in prefixes for suffix in _FEATURE_SUFFIXES]


def compute_history_features(
    frame: pd.DataFrame,
    *,
    time_col: str = "TransactionDT",
    amount_col: str = "TransactionAmt",
    entity_columns: dict[str, str] | None = None,
    id_col: str = "TransactionID",
) -> pd.DataFrame:
    """Calcula las features de historial en una sola pasada causal.

    ``entity_columns`` mapea prefijo de feature -> columna de proxy, por ejemplo
    ``{"card": "card_proxy", "cliente": "cliente_proxy"}``.

    El frame debe venir ordenado por tiempo. El recorrido agrupa eventos con el
    mismo timestamp, emite las features de todo el grupo contra el estado previo y
    solo entonces incorpora el grupo al historial.

    Dentro de un grupo empatado, la INCORPORACION al historial sigue el orden del
    identificador, no el orden de las filas. Los agregados (conteo, suma) son
    invariantes a ese orden, pero el rezago y el tiempo desde el ultimo evento no:
    dependen de cual fue "el ultimo" entre eventos simultaneos, que es ambiguo. Al
    canonicalizar por ID, la invariancia frente a permutaciones de empatados pasa a
    ser una propiedad de esta funcion y no algo que el llamador deba recordar.
    """
    entity_columns = entity_columns or {
        "card": "card_proxy", "cliente": "cliente_proxy", "device": "device_proxy",
    }
    n_rows = len(frame)

    times = frame[time_col].to_numpy(dtype="float64")
    amounts = frame[amount_col].to_numpy(dtype="float64")

    if n_rows and not np.all(np.diff(times) >= 0):
        raise ValueError("El frame debe estar ordenado por tiempo antes de calcular historial")

    entity_values: dict[str, np.ndarray] = {}
    for prefix, column in entity_columns.items():
        if column not in frame.columns:
            raise KeyError("Falta la columna de proxy %s" % column)
        entity_values[prefix] = frame[column].to_numpy(dtype=object)

    identifiers = frame[id_col].to_numpy() if id_col in frame.columns else None

    columns = history_feature_names(entity_columns.keys())
    output = np.full((n_rows, len(columns)), np.nan, dtype="float32")
    column_index = {name: i for i, name in enumerate(columns)}

    states: dict[str, dict[Any, _EntityState]] = {prefix: {} for prefix in entity_columns}

    start = 0
    while start < n_rows:
        # Delimita el grupo de eventos con timestamp identico.
        end = start + 1
        current_time = times[start]
        while end < n_rows and times[end] == current_time:
            end += 1

        # 1) EMITIR: todo el grupo lee el mismo estado previo.
        for row in range(start, end):
            for prefix, values in entity_values.items():
                key = values[row]
                if key is None or key is np.nan or (isinstance(key, float) and np.isnan(key)):
                    continue
                state = states[prefix].get(key)
                if state is None:
                    state = _EntityState()
                    states[prefix][key] = state
                snapshot = state.snapshot(current_time)
                for suffix, value in snapshot.items():
                    output[row, column_index[prefix + "_" + suffix]] = value

        # 2) ACTUALIZAR: solo despues de emitir el grupo completo, y en orden
        #    canonico de identificador para que el rezago no dependa de como
        #    llegaron ordenadas las filas empatadas.
        push_order = range(start, end)
        if end - start > 1 and identifiers is not None:
            push_order = sorted(push_order, key=lambda r: identifiers[r])
        for row in push_order:
            for prefix, values in entity_values.items():
                key = values[row]
                if key is None or key is np.nan or (isinstance(key, float) and np.isnan(key)):
                    continue
                states[prefix][key].push(current_time, amounts[row])

        start = end

    return pd.DataFrame(output, columns=columns, index=frame.index)


def add_derived_features(frame: pd.DataFrame, *, amount_col: str = "TransactionAmt") -> pd.DataFrame:
    """Transformaciones puntuales que no dependen de otras filas.

    Son seguras respecto de la causalidad porque cada una se calcula con la propia
    fila: no hay agregado, ni estadistico global, ni ajuste que mire el futuro.
    """
    out = frame.copy()
    out["monto_log"] = np.log1p(out[amount_col].clip(lower=0))
    # Cuanto se desvia el monto de lo habitual en esa entidad, en escala log.
    for prefijo, sufijo in (("card", "tarjeta"), ("cliente", "cliente")):
        columna = "%s_amt_medio_7d" % prefijo
        if columna in out.columns:
            ratio = out[amount_col] / out[columna].replace(0, np.nan)
            out["monto_vs_media_%s_7d" % sufijo] = np.log1p(ratio.clip(lower=0)).astype("float32")
    for prefijo in ("card", "cliente", "device"):
        columna = "%s_segundos_desde_ultimo" % prefijo
        if columna in out.columns:
            out["horas_desde_ultimo_%s" % prefijo] = (out[columna] / 3600.0).astype("float32")
    return out


# --------------------------------------------------------------------------- panel

def build_feature_panel(
    frame: pd.DataFrame,
    *,
    target: str = "isFraud",
    extra_forbidden: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    """Separa las columnas utilizables en numericas y categoricas.

    Devuelve ``(numericas, categoricas)``. Aplica la lista negra de columnas
    prohibidas para el predictor, de modo que la exclusion sea una propiedad del
    codigo y no un recordatorio en la documentacion.
    """
    forbidden = set(FORBIDDEN_PREDICTOR_COLUMNS) | set(extra_forbidden) | {target}
    numeric: list[str] = []
    categorical: list[str] = []
    for column in frame.columns:
        if column in forbidden or column.endswith(FORBIDDEN_PREDICTOR_SUFFIXES):
            continue
        dtype = frame[column].dtype
        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def assert_no_forbidden_features(columns: Sequence[str], *, extra_forbidden: Iterable[str] = ()) -> None:
    """Falla si una columna prohibida llego al conjunto de entrada del modelo."""
    forbidden = set(FORBIDDEN_PREDICTOR_COLUMNS) | set(extra_forbidden)
    present = sorted(
        [c for c in columns if c in forbidden]
        + [c for c in columns if c.endswith(FORBIDDEN_PREDICTOR_SUFFIXES)]
    )
    if present:
        raise ValueError(
            "Columnas prohibidas en el panel del predictor: %s. "
            "Permitirlas dejaria al modelo memorizar identificadores o tiempo absoluto." % present
        )


def prepare_features(
    frame: pd.DataFrame,
    *,
    time_col: str = "TransactionDT",
    amount_col: str = "TransactionAmt",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pipeline completo de features: proxies, historial causal y derivadas.

    Es la unica puerta de entrada usada por el resto del sistema, para que no
    existan dos caminos con reglas de causalidad distintas.
    """
    out = frame.copy()
    out["card_proxy"] = build_card_proxy(out)
    out["cliente_proxy"] = build_client_proxy(out, time_col=time_col)
    out["device_proxy"] = build_device_proxy(out)

    quality = {
        "card": proxy_quality(out, "card_proxy"),
        "cliente": proxy_quality(out, "cliente_proxy"),
        "device": proxy_quality(out, "device_proxy"),
    }

    LOGGER.info("Calculando historial causal sobre %d eventos", len(out))
    history = compute_history_features(out, time_col=time_col, amount_col=amount_col)
    out = pd.concat([out, history], axis=1)
    out = add_derived_features(out, amount_col=amount_col)

    n_ties = int(out.groupby(time_col).size().gt(1).sum())
    metadata = {
        "proxies": quality,
        "n_features_historial": history.shape[1],
        "ventanas": dict(HISTORY_WINDOWS),
        "grupos_de_empate_temporal": n_ties,
        "regla_causal": "features emitidas antes de actualizar estado; empates leen el mismo pasado",
    }
    return out, metadata
