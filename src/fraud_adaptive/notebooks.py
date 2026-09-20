"""Generacion de los tres notebooks ejecutables.

Los notebooks NARRAN resultados; no duplican logica. Cada celda llama a los
modulos de ``fraud_adaptive`` y lee artefactos ya verificados por hash, de modo que
abrirlos no dispara 35 reajustes ni depende del orden manual de ejecucion.

Se generan por codigo, no a mano, para que el contenido no se desincronice de la
API real: si una funcion cambia de firma, el notebook regenerado la refleja.
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
RUN_DIR = ROOT / CONFIGS["base"]["paths"]["runs"] / "principal"
REPORTS = ROOT / CONFIGS["base"]["paths"]["reports"]

MANIFEST = read_json(ROOT / CONFIGS["base"]["paths"]["manifests"] / "sources.json")
DATA_SOURCE = MANIFEST.get("data_source", "desconocido")

if DATA_SOURCE == "sintetico_sustituto":
    print("AVISO: los datos son un SUSTITUTO SINTETICO. Ninguna cifra describe IEEE-CIS.")
else:
    print("Datos: IEEE-CIS Fraud Detection (particion train).")
print("Run:", RUN_DIR)
"""


def build_eda_notebook() -> Any:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _markdown("""
# 01 · EDA temporal e integridad de los datos

Qué se responde aquí:

1. ¿Las dos fuentes se integran sin perder ni duplicar filas?
2. ¿Cómo cambian el fraude, el volumen y los faltantes a lo largo de 182 días?
3. ¿Las features de historial son realmente causales?

El notebook **no** recalcula nada pesado: lee el Parquet y los manifests que
produjo `fraud-adaptive data prepare`.
"""),
        _code(_PREAMBLE),
        _markdown("## 1. Auditoría del join\n\nEl número de filas debe ser invariante: un join "
                  "many-to-many las multiplicaría en silencio."),
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
        _markdown("## 2. Los dos relojes\n\n`TransactionDT` es un **delta en segundos**, no una "
                  "fecha. La hora derivada es relativa a un origen desconocido: sirve como ciclo "
                  "de 24 h, no identifica hora local ni día laboral."),
        _code("""
eventos = pd.read_parquet(ROOT / CONFIGS["base"]["paths"]["processed"] / "eventos.parquet")

print(f"Eventos: {len(eventos):,}   columnas: {eventos.shape[1]}")
print(f"Días cubiertos: {eventos['dia'].min()} a {eventos['dia'].max()}")
print(f"Transacciones por día (mediana): {int(eventos.groupby('dia').size().median()):,}")
eventos[["TransactionDT", "s_rel", "dia", "semana", "hora"]].head()
"""),
        _markdown("## 3. Perfil semanal con intervalos de Wilson\n\nCon prevalencias del ~3,5 % y "
                  "miles de filas por semana, el intervalo de Wald puede salirse de [0,1]; Wilson no."),
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
        _markdown("## 5. Anomalías antes de culpar al drift\n\nUn hueco de captura o un pico de "
                  "duplicados explican una alerta mejor que un cambio de comportamiento. Se "
                  "descartan primero."),
        _code("""
from fraud_adaptive.data import detect_anomalies

print(json.dumps(detect_anomalies(eventos, amount_col="TransactionAmt", target="isFraud"),
                 indent=2, ensure_ascii=False))
"""),
        _markdown("## 6. Calidad de los proxies de entidad\n\nUn proxy agrupa comportamiento; "
                  "**no identifica a una persona**. Se mide en vez de asumirse."),
        _code("""
for nombre, calidad in MANIFEST["features"]["proxies"].items():
    print(f"{nombre}:")
    for clave, valor in calidad.items():
        if clave != "nota":
            print(f"    {clave:<30} {valor}")
    print()
"""),
        _markdown("""
## 7. Causalidad: la prueba que importa

Las features se emiten **antes** de actualizar el estado. La comprobación decisiva
es que añadir un evento futuro no mueva ninguna fila anterior.
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
# Solo las features numéricas de historial: `card_proxy` es la clave de entidad, texto.
cols = [c for c in a.columns
        if c.startswith("card_") and c != "card_proxy" and pd.api.types.is_numeric_dtype(a[c])]

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


def build_models_notebook() -> Any:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _markdown("""
# 02 · Modelos, calibración y degradación temporal

Qué se responde aquí:

1. ¿Cuál de las tres familias decide mejor, y según qué criterio?
2. ¿Sirve de algo calibrar?
3. ¿Se degrada el sistema estático, y cómo se distingue eso de un cambio de tasa base?

Lee artefactos ya calculados: **no reentrena**.
"""),
        _code(_PREAMBLE),
        _markdown("## 1. Tuning: 18 fits\n\n3 familias × 3 configuraciones × 2 folds forward. "
                  "La selección es por AP media de validación; los folds son **internos** y sus "
                  "métricas nunca se reportan como resultado final."),
        _code("""
tuning = pd.read_csv(REPORTS / "tables" / "tuning.csv")
tuning.sort_values(["familia", "ap_media"], ascending=[True, False]).round(4)
"""),
        _markdown("## 2. Comparación de las tres familias\n\nEl criterio de decisión es el "
                  "**costo**, no el AP. Una familia con mejor AP puede decidir peor si su "
                  "calibración desplaza los umbrales."),
        _code("""
modelos = pd.read_csv(REPORTS / "tables" / "comparacion_modelos.csv")
modelos[["familia", "config", "ap_politica", "costo_um_tx_politica",
         "brier_crudo", "brier_calibrado", "ece_crudo", "ece_calibrado",
         "tau_low", "tau_high", "elegida_para_adaptacion"]].round(5)
"""),
        _markdown("""
### Por qué calibrar

La política compara `p·monto` con `(1−p)·c_FP`. Esa aritmética solo tiene sentido si
`p` es una probabilidad, no un puntaje ordenado. Un modelo con `scale_pos_weight`
produce puntajes inflados: usarlos como probabilidad haría bloquear de más, y el
AP no lo notaría.
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
        _markdown("## 3. Umbrales: nunca 0.5\n\nSe eligen por costo sobre la reserva `[76,83)`. "
                  "También se fijan dos umbrales **diagnósticos**, que se aplican congelados al test."),
        _code("""
elegida = estaticos["familias"][estaticos["familia_elegida"]]

print("Política económica:", json.dumps(elegida["policy"], indent=2))
print("\\nUmbral para FPR<=1%   :", json.dumps(elegida["umbral_fpr"], indent=2))
print("Umbral para precisión>=80%:", json.dumps(elegida["umbral_precision"], indent=2))
print("\\nReferencias simuladas (UM/tx):")
for nombre, valores in estaticos["baselines"].items():
    print(f"   {nombre:<18} {valores['costo_por_tx']:.4f}")
"""),
        _markdown("""
## 4. Degradación del sistema estático

La evidencia clave no es que el AP baje, sino que baje **mientras la tasa base se
mantiene plana**. Si la prevalencia cayera a la vez, el AP bajaría por aritmética y
no habría nada que concluir sobre el modelo.
"""),
        _code("""
from fraud_adaptive.metrics import metrics_by_period

desenlaces = pd.read_parquet(RUN_DIR / "outcomes.parquet")
estatico = desenlaces[desenlaces["estrategia"] == "S0"]

serie = metrics_by_period(estatico, period_column="semana",
                          threshold_binary=elegida["policy"]["tau_high"])
serie[["semana", "n", "prevalencia", "ap", "skill", "brier", "costo_por_tx"]].round(4)
"""),
        _code("""
from IPython.display import Image, display
display(Image(filename=str(REPORTS / "figures" / "rendimiento_estatico.png")))
display(Image(filename=str(REPORTS / "figures" / "calibracion_costos.png")))
"""),
        _markdown("""
## 5. Lectura

Si el AP cae y la prevalencia no, el modelo perdió capacidad discriminante sobre
las mismas clases: es consistente con un cambio en P(y|X). La comparación temporal
por sí sola **no identifica causalmente** el drift, y el informe no lo afirma.
"""),
    ]
    return nb


def build_drift_notebook() -> Any:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        _markdown("""
# 03 · Drift, adaptación por olvido y decisión

Qué se responde aquí:

1. ¿Compensa olvidar? ¿Qué tamaño de ventana?
2. ¿Los detectores encuentran el cambio, y con cuánto retraso?
3. ¿Qué **no** consigue el sistema?

Este es el experimento central: cinco estrategias sobre **los mismos eventos**, la
misma semilla, la misma política y la misma familia.
"""),
        _code(_PREAMBLE),
        _markdown("## 1. El protocolo temporal\n\nLas tres reservas de 7 días son idénticas para "
                  "todas las estrategias. Eso es lo que hace que una diferencia de costo sea "
                  "atribuible al **tamaño de ventana** y no a otra cosa."),
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
        _markdown("## 2. Resultado central\n\nUna fila por estrategia, con las tres familias de "
                  "métricas. `S0` y `E15` son **referencias no desplegables**."),
        _code("""
adaptacion = pd.read_csv(REPORTS / "tables" / "adaptacion.csv")
adaptacion[["estrategia", "ap", "skill", "brier", "f1", "costo_um_tx",
            "costo_ic_low", "costo_ic_high", "bloqueo_legitimas",
            "n_revisiones", "versiones_activadas"]].round(4)
"""),
        _markdown("### Degradación por bloque\n\nAquí se ve si el olvido sirve: una ventana corta "
                  "debería sostener el AP mientras el estático cae."),
        _code("""
adaptacion[["estrategia", "ap_B1", "ap_B2", "ap_B3", "ap_B4"]].round(4)
"""),
        _markdown("### Diferencia pareada frente a las referencias\n\nRemuestreo de bloques "
                  "contiguos de 7 días sobre los **mismos eventos**. Un intervalo que no cruza "
                  "el cero indica una diferencia consistente."),
        _code("""
cols = [c for c in adaptacion.columns if c.startswith("delta_costo_vs_")]
adaptacion[["estrategia"] + cols].round(4)
"""),
        _markdown("## 3. ¿Se cumplen los objetivos declarados?\n\nUn objetivo **puede incumplirse**. "
                  "Se reporta el nivel realmente obtenido, no el objetivo."),
        _code("""
objetivos = CONFIGS["decision"]["diagnostic_targets"]
tabla = adaptacion[["estrategia", "recall_at_fpr1", "fpr_obtenido",
                    "recall_at_precision80", "precision_obtenida"]].round(4)
print(f"Objetivos: FPR <= {objetivos['fpr_target']}, precisión >= {objetivos['precision_target']}\\n")
display(tabla)

print("\\nFPR objetivo alcanzado en test:",
      bool((tabla['fpr_obtenido'] <= objetivos['fpr_target']).any()))
print("Precisión objetivo alcanzada en test:",
      bool((tabla['precision_obtenida'] >= objetivos['precision_target']).any()))
print("\\nLos umbrales se eligieron en validación y se aplicaron CONGELADOS.")
print("Que no se sostengan en test es un resultado, no un error a corregir moviendo el umbral.")
"""),
        _markdown("## 4. Señales de drift y sus dos relojes\n\nKS/PSI y S1 están disponibles el "
                  "mismo día. ADWIN observa el **error real** y por eso llega L=30 días después."),
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
        _markdown("### El detector, probado contra verdad conocida\n\nValidación del "
                  "**instrumento**: no son resultados sobre fraude."),
        _code("""
pd.read_csv(REPORTS / "tables" / "adwin_selftest.csv")
"""),
        _markdown("## 5. Sensibilidad económica\n\nRescoring de predicciones guardadas con la "
                  "política **congelada**. Responde *cuánto cambiaría la conclusión*, no *qué "
                  "política sería óptima*."),
        _code("""
sens_path = REPORTS / "tables" / "sensibilidad_economica.csv"
if sens_path.exists():
    display(pd.read_csv(sens_path).round(4))
else:
    print("Sensibilidad no disponible en este run.")
"""),
        _markdown("## 6. Equidad operativa\n\nSegmentos de **negocio**. IEEE-CIS no tiene "
                  "atributos protegidos verificables: esto no es una auditoría demográfica."),
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
## 8. Qué no demuestra este experimento

- No es una validación prospectiva: es un backtest retrospectivo, y la selección
  previa del dataset ya examinó periodos tardíos.
- Los costos son simulados bajo supuestos declarados; no hay ahorro causal medido.
- Una sola semilla no permite afirmar variabilidad entre semillas.
- La ventana recomendada es la elegida **en desarrollo**. Si otra resulta mejor en
  el test final, eso es un diagnóstico retrospectivo, no un permiso para retunear.
"""),
    ]
    return nb


def write_notebooks(output_dir: str | Path) -> dict[str, Path]:
    """Escribe los tres notebooks sin ejecutarlos."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    built = {
        "01_eda_temporal.ipynb": build_eda_notebook(),
        "02_modelos_temporales.ipynb": build_models_notebook(),
        "03_drift_adaptacion.ipynb": build_drift_notebook(),
    }
    out: dict[str, Path] = {}
    for name, notebook in built.items():
        notebook.metadata["kernelspec"] = {
            "display_name": "Python 3", "language": "python", "name": "python3",
        }
        notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
        path = output_dir / name
        with open(path, "w", encoding="utf-8") as handle:
            nbf.write(notebook, handle)
        out[name] = path
    return out
