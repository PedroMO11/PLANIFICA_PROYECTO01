# Reproducibilidad

Cómo repetir la corrida entregada y qué garantías ofrece.

---

## 1. Entorno

Gestor de paquetes: uv, con Python 3.12 (LightGBM y river publican
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
paquete científico, de modo que cada resultado queda asociado a su entorno.

---

## 2. Datos

### Opción A: datos reales de IEEE-CIS

```bash
python -m fraud_adaptive data instructions   # imprime los pasos exactos
python -m fraud_adaptive data inspect        # verifica hash y esquema
```

Requiere aceptar las reglas de la competencia y un token individual de Kaggle,
guardado fuera del repositorio (`%USERPROFILE%\.kaggle\kaggle.json`). Solo se
descargan `train_transaction.csv` y `train_identity.csv`: los `test_*` no tienen
etiqueta y el código los rechaza de forma explícita.

### Opción B: sustituto sintético

```bash
python -m fraud_adaptive data surrogate --scale 1.0
```

Genera un dataset con el mismo esquema (394 columnas), el mismo eje temporal
(182 días, unas 3 200 transacciones diarias), una prevalencia cercana a 3,5 % y una
cobertura de identidad cercana a 24 %, con drift inyectado de parámetros conocidos.

Sirve para ejecutar y verificar el pipeline en una máquina sin acceso a Kaggle. Sus
cifras no describen IEEE-CIS: cada artefacto queda marcado con
`data_source: sintetico_sustituto` y las figuras llevan el rótulo impreso.

Los comandos posteriores son los mismos en ambos casos. Al reemplazar los CSV por
los reales y ejecutar con `--rebuild`, el pipeline produce los resultados sobre
IEEE-CIS.

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
python -m fraud_adaptive adapt run         # backtest de las siete estrategias
python -m fraud_adaptive selftest adwin    # detector contra verdad conocida
python -m fraud_adaptive report build      # tablas, figuras y documentos
```

---

## 4. Caché y reconstrucción

`data/processed/eventos.parquet` se reutiliza si existe. Para reconstruir desde
los CSV: `data prepare --rebuild`.

Los fits de tuning se saltan si hay checkpoint en `runs/<run_id>/checkpoints/`.
Para rehacerlos, borra el checkpoint correspondiente o usa otro `--run-id`.

El manifest guarda el hash del código y de la configuración. Si alguno cambia, los
artefactos previos dejan de considerarse equivalentes y el pipeline falla o
reconstruye de forma explícita.

---

## 5. Presupuesto de cómputo

El tope acumulado es de 480 minutos, con 98 de reserva. `runs/<run_id>/budget.json`
registra cada tarea con su duración y estado. El contador se conserva al reanudar,
porque el tope corresponde al proyecto completo. Una tarea que falla o excede su
límite consume presupuesto y queda marcada con su estado real. La corrida v3 usó
39,5 minutos.

---

## 6. Qué se garantiza

| Garantía | Alcance |
|---|---|
| Mismo entorno, mismos datos, misma semilla → mismos resultados | Sí, dentro de la misma máquina |
| Resultados idénticos bit a bit entre máquinas distintas | Sin garantía: BLAS, la CPU y el orden de reducción en punto flotante pueden diferir |
| Trazabilidad de cada cifra a una corrida y un hash | Sí, en `reports/indice_evidencia.csv` |
| Ausencia de fuga temporal | Verificada por 150 pruebas automáticas |

La semilla 42 se usa en todo el pipeline. Con una sola semilla de entrenamiento no
se puede estimar la variabilidad entre semillas, y el informe no la reporta.

Una segunda corrida (`v3_repro`) reprodujo sin diferencias los 18 fits de tuning, el
hash de prerregistro y el costo de S0, E15, W30, W60 y W90
(`reports/verificacion_reproducibilidad.json`).

---

## 7. Hardware de referencia de esta corrida

Los tiempos del informe se midieron en un AMD Ryzen 7 7800X3D de 8 núcleos con
31 GB de RAM y Windows 11. El código usa 4 hilos por configuración para mantener la
comparabilidad con el presupuesto planificado.

---

## 8. Estructura de un run

```
runs/<run_id>/
├── manifest.json        entorno, hashes de config y código, artefactos con SHA-256
├── budget.json          consumo acumulado por tarea
├── tasks.json           detalle de cada tarea
├── authorization.json   autorización humana previa de la corrida offline
├── events.jsonl         bitácora append-only
├── checkpoints/         estado reanudable por tarea
├── static_results.json  familia elegida, umbrales, prerregistro
├── predictions.parquet  predicciones inmutables con su versión
├── outcomes.parquet     desenlaces tras la madurez de la etiqueta
└── replay.sqlite        ledger del replay (cupo y decisiones)
```
