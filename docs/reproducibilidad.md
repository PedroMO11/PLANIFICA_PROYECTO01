# Reproducibilidad

Cómo volver a obtener exactamente estos resultados, y qué se garantiza y qué no.

---

## 1. Entorno

Gestor de paquetes: **uv**. Python fijado a 3.12 (LightGBM y river publican
ruedas estables para esa versión).

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install -e ".[serving,dev]"
```

Versiones exactas en `requirements.lock` (desarrollo) y `requirements-serving.lock`
(imagen de serving, sin dependencias de entrenamiento).

Cada `runs/<run_id>/manifest.json` graba el entorno real de esa corrida: versión de
Python, plataforma, procesador, número de CPUs, commit de git y versión de cada
paquete científico. Un resultado siempre es atribuible a un entorno concreto.

---

## 2. Datos

### Opción A — datos reales de IEEE-CIS

```bash
python -m fraud_adaptive data instructions   # imprime los pasos exactos
python -m fraud_adaptive data inspect        # verifica hash y esquema
```

Requiere aceptar las reglas de la competencia y un token individual de Kaggle,
guardado **fuera del repositorio** (`%USERPROFILE%\.kaggle\kaggle.json`). Solo se
descargan `train_transaction.csv` y `train_identity.csv`: los `test_*` no tienen
etiqueta y el código los rechaza de forma explícita.

### Opción B — sustituto sintético

```bash
python -m fraud_adaptive data surrogate --scale 1.0
```

Genera un dataset con el **mismo esquema** (394 columnas), el mismo eje temporal
(182 días, ~3 200 transacciones/día), la misma prevalencia (~3,5 %) y la misma
cobertura de identidad (~24 %), con drift inyectado de parámetros conocidos.

Existe para poder ejecutar y verificar el pipeline completo en una máquina sin
acceso a Kaggle. **Ninguna cifra obtenida sobre él describe IEEE-CIS.** Todo
artefacto queda marcado con `data_source: sintetico_sustituto` en su manifest, y
las figuras llevan el rótulo impreso.

Los comandos posteriores son idénticos en ambos casos: al reemplazar los CSV por
los reales y ejecutar con `--rebuild`, el mismo pipeline produce resultados reales.

---

## 3. Cadena completa

```bash
python -m fraud_adaptive all
```

Equivale a, en orden y de forma reanudable:

```bash
python -m fraud_adaptive data prepare      # join, relojes, features causales, Parquet
python -m fraud_adaptive train tune        # 18 fits (3 familias x 3 configs x 2 folds)
python -m fraud_adaptive train static      # 3 fits finales, Platt, umbrales, familia
python -m fraud_adaptive adapt run         # backtest de 5 estrategias
python -m fraud_adaptive selftest adwin    # detector contra verdad conocida
python -m fraud_adaptive report build      # tablas, figuras y documentos
```

---

## 4. Caché y reconstrucción

`data/processed/eventos.parquet` se reutiliza si existe. Para reconstruir desde
los CSV: `data prepare --rebuild`.

Los fits de tuning se saltan si hay checkpoint en `runs/<run_id>/checkpoints/`.
Para rehacerlos, borra el checkpoint correspondiente o usa otro `--run-id`.

**Una caché inválida provoca error o reconstrucción explícita, nunca reutilización
silenciosa.** El manifest guarda el hash del código y de la configuración; si
cambian, los artefactos dejan de considerarse equivalentes.

---

## 5. Presupuesto de cómputo

Tope acumulado de **480 minutos**, con 98 de reserva. `runs/<run_id>/budget.json`
registra cada tarea con su duración y estado.

El contador **no se reinicia al reanudar**: el tope es del proyecto, no de la
sesión. Una tarea que falla o que excede su límite igual consume presupuesto y
queda marcada — un timeout nunca se presenta como un fit exitoso.

---

## 6. Qué se garantiza

| Garantía | Alcance |
|---|---|
| Mismo entorno, mismos datos, misma semilla → mismos resultados | Sí, dentro de la misma máquina |
| Resultados idénticos bit a bit entre máquinas distintas | **No se promete.** BLAS, versión de CPU y orden de reducción en punto flotante pueden diferir |
| Trazabilidad de cada cifra a un run y un hash | Sí: `reports/indice_evidencia.csv` |
| Ausencia de fuga temporal | Verificada por 127 pruebas automáticas |

Semilla 42 en todo el núcleo. Una sola semilla **no** permite afirmar
variabilidad entre semillas, y el informe no lo hace.

---

## 7. Hardware de referencia de esta corrida

Los tiempos del informe se midieron en: AMD Ryzen 7 7800X3D (8 núcleos), 31 GB
RAM, Windows 11. El código se limita a **4 hilos** por configuración, no por
capacidad de la máquina: cambiarlo alteraría la comparabilidad con el presupuesto
planificado.

---

## 8. Estructura de un run

```
runs/<run_id>/
├── manifest.json        entorno, hashes de config y código, artefactos con SHA-256
├── budget.json          consumo acumulado por tarea
├── tasks.json           detalle de cada tarea
├── authorization.json   manifest de autorización humana previa (C23)
├── events.jsonl         bitácora append-only
├── checkpoints/         estado reanudable por tarea
├── static_results.json  familia elegida, umbrales, prerregistro
├── predictions.parquet  predicciones inmutables con su versión
├── outcomes.parquet     desenlaces tras la madurez de la etiqueta
└── replay.sqlite        ledger del replay (cupo y decisiones)
```
