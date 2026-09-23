"""Generacion de los tres notebooks del proyecto.

Cada celda llama a los modulos de ``fraud_adaptive`` o lee artefactos de la corrida,
sin reentrenar modelos. Los notebooks se generan por codigo para que sigan la API
del paquete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nbformat as nbf


def _markdown(text: str) -> Any:
    return nbf.v4.new_markdown_cell(text.strip())


def _code(source: str) -> Any:
    return nbf.v4.new_code_cell(source.strip())


_PREAMBLE = """
import sys, json
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "configs").exists() and (ROOT.parent / "configs").exists():
    ROOT = ROOT.parent          # el notebook puede abrirse desde notebooks/
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fraud_adaptive.pipeline import load_configs
from fraud_adaptive.tracking import read_json

CONFIGS = load_configs(ROOT / "configs")
# Corrida de la que provienen las cifras del informe.
RUN_ID = "__RUN_ID__"
RUN_DIR = ROOT / CONFIGS["base"]["paths"]["runs"] / RUN_ID
REPORTS = ROOT / CONFIGS["base"]["paths"]["reports"]

MANIFEST = read_json(ROOT / CONFIGS["base"]["paths"]["manifests"] / "sources.json")
DATA_SOURCE = MANIFEST.get("data_source", "desconocido")

if DATA_SOURCE == "sintetico_sustituto":
    print("Aviso: los datos son un sustituto sintetico y las cifras no describen IEEE-CIS.")
else:
    print("Datos: IEEE-CIS Fraud Detection (particion train).")
print("Run:", RUN_DIR)
"""


def _con_run(texto: str, run_id: str) -> str:
    return texto.replace("__RUN_ID__", run_id)


def build_eda_notebook(run_id: str = "principal") -> Any:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _markdown("""
# 01 · EDA temporal e integridad de los datos

Qué se responde aquí:

1. ¿Las dos fuentes se integran sin perder ni duplicar filas?
2. ¿Cómo cambian el fraude, el volumen y los faltantes a lo largo de 182 días?
3. ¿Las features de historial usan solo el pasado?

El notebook lee el Parquet y los manifests que produce `fraud-adaptive data prepare`.
"""),
        _code(_PREAMBLE),
        _markdown("## 1. Auditoría del join\n\nEl número de filas debe mantenerse; un join "
                  "de muchos a muchos las multiplicaría sin aviso."),
        _code("""
join = MANIFEST["join"]
esquema = MANIFEST["esquema"]

print(f"Transacciones : {esquema['n_transactions']:,} filas x {esquema['n_columns_transactions']} columnas")
print(f"Identidad     : {esquema['n_identity']:,} filas x {esquema['n_columns_identity']} columnas")
print(f"Prevalencia   : {esquema['prevalencia']:.4f}")
print()
for clave, valor in join.items():
    print(f"  {clave:<28} {valor}")

assert join["filas_invariantes"], "el join alteró el número de filas"
print("\\nOK: el join es uno-a-uno y preserva todas las transacciones.")
"""),
        _markdown("## 2. Reloj de los eventos\n\n`TransactionDT` es un desfase en segundos desde "
                  "un origen desconocido. La hora derivada sirve como ciclo de 24 horas y no "
                  "permite inferir la hora local ni el día laboral."),
        _code("""
eventos = pd.read_parquet(ROOT / CONFIGS["base"]["paths"]["processed"] / "eventos.parquet")

print(f"Eventos: {len(eventos):,}   columnas: {eventos.shape[1]}")
print(f"Días cubiertos: {eventos['dia'].min()} a {eventos['dia'].max()}")
print(f"Transacciones por día (mediana): {int(eventos.groupby('dia').size().median()):,}")
eventos[["TransactionDT", "s_rel", "dia", "semana", "hora"]].head()
"""),
        _markdown("## 3. Perfil semanal con intervalos de Wilson\n\nCon prevalencias cercanas a "
                  "3,5 %, el intervalo de Wilson se comporta mejor que el de Wald cerca de cero."),
        _code("""
from fraud_adaptive.data import weekly_profile

semanal = weekly_profile(eventos, target="isFraud", amount_col="TransactionAmt")
semanal[["semana", "n", "fraudes", "prevalencia", "wilson_low", "wilson_high",
         "cobertura_identidad"]].round(4).head(12)
"""),
        _markdown("## 4. Faltantes por familia\n\nIEEE-CIS tiene cientos de columnas anónimas. "
                  "La familia (V, C, D, M, id_, card, addr) es la unidad interpretable."),
        _code("""
from fraud_adaptive.data import missingness_by_family

missingness_by_family(eventos).round(3)
"""),
        _markdown("## 5. Anomalías temporales\n\nUn hueco de captura o un pico de duplicados "
                  "pueden disparar una alerta de drift, así que se revisan antes."),
        _code("""
from fraud_adaptive.data import detect_anomalies

print(json.dumps(detect_anomalies(eventos, amount_col="TransactionAmt", target="isFraud"),
                 indent=2, ensure_ascii=False))
"""),
        _markdown("## 6. Calidad de los proxies de entidad\n\nUn proxy agrupa comportamiento y "
                  "puede reunir a varias personas bajo una misma clave, por eso se reporta su "
                  "calidad."),
        _code("""
for nombre, calidad in MANIFEST["features"]["proxies"].items():
    print(f"{nombre}:")
    for clave, valor in calidad.items():
        if clave != "nota":
            print(f"    {clave:<30} {valor}")
    print()
"""),
        _markdown("""
## 7. Causalidad de las features

Cada feature se calcula antes de incorporar el evento al estado. Al agregar un
evento futuro, ninguna fila anterior debe cambiar.
"""),
        _code("""
from fraud_adaptive.features import prepare_features

base = pd.DataFrame({
    "TransactionID": [1, 2, 3], "TransactionDT": [100, 200, 300],
    "TransactionAmt": [10.0, 20.0, 30.0],
    "card1": ["A"]*3, "card2": [1]*3, "card3": [1]*3, "card4": ["visa"]*3,
    "card5": [1]*3, "card6": ["debit"]*3, "addr1": [1]*3,
    "DeviceType": ["desktop"]*3, "DeviceInfo": ["Windows"]*3,
    "has_identity": [1]*3, "isFraud": [0]*3,
})
con_futuro = pd.concat([base, base.iloc[[0]].assign(
    TransactionID=4, TransactionDT=400, TransactionAmt=999999.0)], ignore_index=True)

a, _ = prepare_features(base)
b, _ = prepare_features(con_futuro)
# Solo las features numéricas de historial: los `*_proxy` son claves de entidad, texto.
cols = [c for c in a.columns
        if c.startswith(("card_", "cliente_")) and not c.endswith("_proxy")
        and pd.api.types.is_numeric_dtype(a[c])]

iguales = np.allclose(a[cols].fillna(-1).to_numpy(), b[cols].head(3).fillna(-1).to_numpy())
print("Un evento futuro de monto extremo no altera las filas anteriores:", iguales)
assert iguales
a[["TransactionID", "card_cnt_expansivo", "card_cnt_24h", "card_amt_rezago"]]
"""),
        _markdown("## 8. Figura de EDA\n\nGenerada por `fraud-adaptive report build`."),
        _code("""
from IPython.display import Image, display
display(Image(filename=str(REPORTS / "figures" / "eda_temporal.png")))
"""),
    ]
    return nb


def build_models_notebook(run_id: str = "principal") -> Any:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _markdown("""
# 02 · Modelos, calibración y degradación temporal

Qué se responde aquí:

1. ¿Cuál de las tres familias decide mejor, y según qué criterio?
2. ¿Sirve de algo calibrar?
3. ¿Se degrada el sistema estático, y cómo se distingue eso de un cambio de tasa base?

Lee artefactos ya calculados, sin reentrenar.
"""),
        _code(_PREAMBLE),
        _markdown("## 1. Tuning: 18 fits\n\n3 familias × 3 configuraciones × 2 folds temporales. "
                  "Cada configuración se elige por AP media de validación. Los folds son internos "
                  "y sus métricas no se reportan como resultado final."),
        _code("""
tuning = pd.read_csv(REPORTS / "tables" / "tuning.csv")
tuning.sort_values(["familia", "ap_media"], ascending=[True, False]).round(4)
"""),
        _markdown("## 2. Comparación de las tres familias\n\nLa familia se elige por costo. Una "
                  "familia con mejor AP puede decidir peor si su calibración está sesgada, "
                  "porque la política compara costos esperados calculados con `p`.\n\nLa "
                  "columna `costo_um_tx_umbral_fijo` es el costo del mejor par de umbrales "
                  "globales sobre `p`. Su diferencia con `costo_um_tx_politica` mide lo que "
                  "cuesta ignorar el monto."),
        _code("""
modelos = pd.read_csv(REPORTS / "tables" / "comparacion_modelos.csv")
modelos[["familia", "config", "ap_politica", "costo_um_tx_politica",
         "brier_crudo", "brier_calibrado", "ece_crudo", "ece_calibrado",
         "costo_um_tx_umbral_fijo", "regla", "elegida_para_adaptacion"]].round(5)
"""),
        _markdown("""
### Por qué calibrar

La política compara `p·monto` con `(1−p)·c_FP`, lo que exige que `p` sea una
probabilidad calibrada. Un modelo entrenado con pesos de clase produce puntajes
inflados; usados como probabilidad llevarían a bloquear de más sin que el AP lo
refleje.
"""),
        _code("""
estaticos = read_json(RUN_DIR / "static_results.json")

for familia, info in estaticos["familias"].items():
    cal = info["calibracion"]
    print(f"{familia:<15} Brier {cal['brier_crudo']:.5f} -> {cal['brier_calibrado']:.5f}"
          f"   ECE {cal['ece_crudo']:.5f} -> {cal['ece_calibrado']:.5f}")

print(f"\\nFamilia congelada para adaptación: {estaticos['familia_elegida']}")
print(f"Hash de prerregistro: {estaticos['prerregistro']['hash'][:32]}")
"""),
        _markdown("""
## 3. La política de decisión

Cada transacción recibe la acción de menor costo esperado. El punto de indiferencia
entre aprobar y bloquear es `p* = c_FP / (monto + c_FP)` y depende del monto, de modo
que un corte fijo sobre `p` bloquea de más en montos bajos y de menos en montos altos.

`c_FP` se deriva del objetivo de bloqueo de legítimas: es el menor valor que lo
cumple sobre datos de desarrollo.
"""),
        _code("""
elegida = estaticos["familias"][estaticos["familia_elegida"]]

print("Política:", json.dumps(elegida["policy"], indent=2))
print("Economía congelada:", json.dumps(estaticos["costos"], indent=2))

cal = estaticos["calibracion_c_fp"]
print(f"\\nc_FP calibrado contra un objetivo de bloqueo de legítimas del "
      f"{cal['objetivo']:.1%}")
print(f"   familia de referencia : {cal['familia_referencia']}")
print(f"   valor elegido         : {cal['c_fp']:.1f}")
print(f"   tasa alcanzada        : {cal['tasa_bloqueo_legitimo']:.4f}")
pd.DataFrame(cal["tabla"]).round(5)
"""),
        _code("""
print("El argmin induce un umbral distinto por monto:")
for u in estaticos["umbrales_implicados"]:
    print(f"   monto {u['monto']:8.2f} -> revisar desde p={u['tau_low']:.4f},"
          f" bloquear desde p={u['tau_high']:.4f}")

print(f"\\nCosto con argmin          : {elegida['costo_politica']:.4f} UM/tx")
print(f"Costo con el mejor umbral fijo: {elegida['costo_umbral']:.4f} UM/tx")

print("\\nUmbral para FPR<=1%   :", json.dumps(elegida["umbral_fpr"], indent=2))
print("Umbral para precisión>=80%:", json.dumps(elegida["umbral_precision"], indent=2))
print("\\nReferencias simuladas (UM/tx):")
for nombre, valores in estaticos["baselines"].items():
    print(f"   {nombre:<18} {valores['costo_por_tx']:.4f}")
"""),
        _markdown("""
## 4. Comportamiento del modelo estático

Una caída de AP indica pérdida de capacidad del modelo solo si la tasa base se
mantiene; si la prevalencia también cae, el AP baja por aritmética. Por eso la serie
se muestra junto a la prevalencia.
"""),
        _code("""
from fraud_adaptive.metrics import metrics_by_period

desenlaces = pd.read_parquet(RUN_DIR / "outcomes.parquet")
estatico = desenlaces[desenlaces["estrategia"] == "S0"]

serie = metrics_by_period(estatico, period_column="semana",
                          threshold_binary=elegida["umbral_fpr"]["umbral"])
serie[["semana", "n", "prevalencia", "ap", "skill", "brier", "costo_por_tx"]].round(4)
"""),
        _code("""
from IPython.display import Image, display
display(Image(filename=str(REPORTS / "figures" / "rendimiento_estatico.png")))
display(Image(filename=str(REPORTS / "figures" / "calibracion_costos.png")))
"""),
        _markdown("""
## 5. Lectura

Una caída de AP con prevalencia estable es consistente con un cambio en P(y|X),
aunque la comparación temporal no identifica su causa. El notebook 03 compara esta
serie con las estrategias que se reentrenan.
"""),
    ]
    return nb


def build_drift_notebook(run_id: str = "principal") -> Any:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _markdown("""
# 03 · Drift, adaptación y decisión

Qué se responde aquí:

1. ¿Compensa olvidar datos antiguos, y con qué tamaño de ventana?
2. ¿Los detectores encuentran el cambio, y con cuánto retraso?
3. ¿Qué límites tiene el resultado?

Siete estrategias sobre los mismos eventos, con la misma semilla, la misma política
y la misma familia de modelos.
"""),
        _code(_PREAMBLE),
        _markdown("## 1. El protocolo temporal\n\nEn cada actualización, cada versión usa tres "
                  "roles: predictor, calibrador y validación de promoción. Las dos reservas de "
                  "7 días son idénticas para todas las estrategias, de modo que una diferencia "
                  "de costo se puede atribuir a la ventana."),
        _code("""
splits = read_json(ROOT / CONFIGS["base"]["paths"]["manifests"] / "splits.json")

print(f"L = {splits['label_delay_days']} días   cadencia = {splits['cadence_days']} días")
print(f"Regla de elegibilidad: {splits['eligibility_rule']}\\n")

filas = []
for estrategia, versiones in splits["versions"].items():
    for v in versiones:
        filas.append({
            "version": v["version_id"], "T": v["update_time"], "c": v["cutoff"],
            "predictor": f"[{v['predictor'][0]},{v['predictor'][1]})",
            "dias_fit": v["predictor_days"],
            "calibracion": f"[{v['calibration'][0]},{v['calibration'][1]})",
            "H": f"[{v['promotion_validation'][0]},{v['promotion_validation'][1]})",
        })
pd.DataFrame(filas)
"""),
        _markdown("## 2. Resultado central\n\nUna fila por estrategia con métricas técnicas, de "
                  "decisión y sociales. `S0` es el modelo estático de referencia y `E15`, que "
                  "reentrena con todo el histórico, es la estrategia recomendada. F1 y balanced "
                  "accuracy se evalúan en el umbral de FPR 1 % fijado en validación."),
        _code("""
adaptacion = pd.read_csv(REPORTS / "tables" / "adaptacion.csv")
adaptacion[["estrategia", "ap", "skill", "brier", "f1", "balanced_accuracy", "costo_um_tx",
            "costo_ic_low", "costo_ic_high", "bloqueo_legitimas",
            "n_revisiones", "versiones_activadas"]].round(4)
"""),
        _markdown("### AP por bloque\n\nSi olvidar compensara, las ventanas cortas sostendrían "
                  "el AP en los bloques donde el estático cae."),
        _code("""
adaptacion[["estrategia", "ap_B1", "ap_B2", "ap_B3", "ap_B4"]].round(4)
"""),
        _markdown("### Diferencia pareada frente a las referencias\n\nRemuestreo de bloques "
                  "contiguos de 7 días sobre los mismos eventos. Un intervalo que no cruza el "
                  "cero indica una diferencia consistente."),
        _code("""
cols = [c for c in adaptacion.columns if c.startswith("delta_costo_vs_")]
adaptacion[["estrategia"] + cols].round(4)
"""),
        _markdown("## 3. Objetivos diagnósticos\n\nLos umbrales se fijaron en validación y se "
                  "aplican sin reajuste. La tabla muestra el FPR y la precisión que obtiene cada "
                  "estrategia en test."),
        _code("""
objetivos = CONFIGS["decision"]["diagnostic_targets"]
tabla = adaptacion[["estrategia", "recall_at_fpr1", "fpr_obtenido",
                    "recall_at_precision80", "precision_obtenida"]].round(4)
tabla["cumple_fpr"] = tabla["fpr_obtenido"] <= objetivos["fpr_target"]
tabla["cumple_precision"] = tabla["precision_obtenida"] >= objetivos["precision_target"]
print(f"Objetivos: FPR <= {objetivos['fpr_target']}, precisión >= {objetivos['precision_target']}\\n")
display(tabla)

cumplen = tabla.loc[tabla["cumple_fpr"] & tabla["cumple_precision"], "estrategia"]
print("\\nCumplen ambos objetivos:", ", ".join(cumplen) if len(cumplen) else "ninguna")
"""),
        _markdown("## 4. Señales de drift y sus dos relojes\n\nKS/PSI y el domain classifier (S1) "
                  "están disponibles el mismo día. ADWIN observa el error con etiquetas maduras y "
                  "por eso llega 30 días después."),
        _code("""
drift_path = RUN_DIR / "drift_log.csv"
if drift_path.exists():
    drift = pd.read_csv(drift_path)
    alertas = drift[drift["alerta"].fillna(False).astype(bool)] if "alerta" in drift else pd.DataFrame()
    print(f"Registros de señal: {len(drift)}   alertas: {len(alertas)}")
    if not alertas.empty:
        display(alertas.groupby("senal").agg(
            n_alertas=("senal", "size"),
            primer_dia=("dia_evento", "min"),
            retraso_medio=("retraso_dias", "mean"),
        ).round(2))
else:
    print("Sin bitácora de drift en este run.")
"""),
        _code("""
adwin_path = RUN_DIR / "adwin_detections.csv"
if adwin_path.exists():
    adwin = pd.read_csv(adwin_path)
    print(f"Detecciones de ADWIN: {len(adwin)}")
    display(adwin[["estrategia", "version_id", "dia_evento", "dia_disponibilidad",
                   "retraso_dias", "n_updates"]])
else:
    print("ADWIN no detectó cambios en ninguna versión.")
"""),
        _markdown("### Validación del detector\n\nADWIN se prueba sobre dos series sintéticas, "
                  "una estacionaria y otra con un salto conocido. Estos resultados validan el "
                  "instrumento y no describen el fraude."),
        _code("""
pd.read_csv(REPORTS / "tables" / "adwin_selftest.csv")
"""),
        _markdown("## 5. Sensibilidad económica\n\nSe recalcula el costo de las predicciones "
                  "guardadas de W60 con las acciones congeladas y otros parámetros económicos. "
                  "Mide cuánto cambia el costo, sin reoptimizar la política."),
        _code("""
sens_path = REPORTS / "tables" / "sensibilidad_economica.csv"
if sens_path.exists():
    display(pd.read_csv(sens_path).round(4))
else:
    print("Sensibilidad no disponible en este run.")
"""),
        _markdown("## 6. Equidad operativa\n\nSegmentos de negocio sobre la estrategia W60. "
                  "IEEE-CIS no tiene atributos protegidos, de modo que la tabla mide disparidad "
                  "operativa."),
        _code("""
seg_path = REPORTS / "tables" / "segmentos.csv"
if seg_path.exists():
    seg = pd.read_csv(seg_path)
    display(seg[seg["soporte_suficiente"]][
        ["variable", "grupo", "n_legitimas", "tasa_bloqueo", "brecha_pp", "alerta_brecha"]
    ].round(4))
    print(f"\\nGrupos con evidencia insuficiente: {int(seg['evidencia_insuficiente'].sum())}")
    print("Se conservan en la tabla: ocultarlos daría una falsa impresión de cobertura.")
"""),
        _code("""
from IPython.display import Image, display
display(Image(filename=str(REPORTS / "figures" / "drift_adaptacion.png")))
"""),
        _markdown("## 7. Costo de cómputo"),
        _code("""
display(pd.read_csv(REPORTS / "tables" / "costos_computo.csv").round(2))

presupuesto = read_json(RUN_DIR / "budget.json")
print(f"\\nConsumido: {presupuesto['consumed_minutes']:.1f} min de "
      f"{presupuesto['total_minutes']:.0f}  ({presupuesto['remaining_minutes']:.1f} restantes)")
"""),
        _markdown("""
## 8. Alcance

- Es un backtest retrospectivo, y la selección previa del dataset ya examinó
  periodos tardíos.
- Los costos son simulados bajo supuestos declarados.
- Cada estrategia se entrena con una sola semilla; las 18 combinaciones de robustez
  varían el remuestreo y no el entrenamiento.
- W60 se eligió en desarrollo. Que otra ventana rinda mejor en test es un
  diagnóstico retrospectivo y no justifica reajustar.
"""),
    ]
    return nb


def write_notebooks(output_dir: str | Path, run_id: str = "principal") -> dict[str, Path]:
    """Escribe los tres notebooks sin ejecutarlos, apuntando a ``run_id``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    built = {
        "01_eda_temporal.ipynb": build_eda_notebook(run_id),
        "02_modelos_temporales.ipynb": build_models_notebook(run_id),
        "03_drift_adaptacion.ipynb": build_drift_notebook(run_id),
    }
    out: dict[str, Path] = {}
    for name, notebook in built.items():
        notebook.metadata["kernelspec"] = {
            "display_name": "Python 3", "language": "python", "name": "python3",
        }
        notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
        for celda in notebook.cells:
            celda.source = _con_run(celda.source, run_id)
        path = output_dir / name
        with open(path, "w", encoding="utf-8") as handle:
            nbf.write(notebook, handle)
        out[name] = path
    return out
