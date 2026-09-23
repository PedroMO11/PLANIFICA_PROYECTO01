# Sistema adaptativo de decisión ante fraude transaccional

Proyecto 1 del curso Planificación y Toma de Decisiones en IA (UTEC), sobre el
dataset IEEE-CIS Fraud Detection.

El sistema decide para cada transacción entre aprobar, enviar a revisión o
bloquear, minimizando el costo esperado con un cupo de 150 revisiones diarias. Las
etiquetas confirmadas llegan 30 días después del evento, y el modelo se readapta
cada 15 días con autorización humana.

## Entregables

| Entregable | Ubicación |
|---|---|
| Documento técnico (8 páginas) | [reports/informe_final.pdf](reports/informe_final.pdf), fuente en [reports/informe_final.md](reports/informe_final.md) |
| Infografía | [reports/infografia.pdf](reports/infografia.pdf), fuente en [reports/infografia.svg](reports/infografia.svg) |
| Código | `src/fraud_adaptive/`, `tests/`, `configs/` y los notebooks de `notebooks/` |
| Simulación de despliegue | servicio HTTP, replay con ledger, `Dockerfile` y `deploy/` |
| Guion de exposición | [reports/guion_presentacion.md](reports/guion_presentacion.md) |

## Resultados

| Resultado | Valor |
|---|---|
| Costo simulado frente a aprobar todo | de 5,40 a 1,98 UM por transacción (−63,2 %) |
| Estrategia recomendada | E15: reentrenar cada 15 días con todo el histórico |
| E15 frente al modelo estático | −0,108 UM/tx, intervalo de 95 % [−0,205; 0,002] |
| Ventanas deslizantes W30, W60 y W90 | entre 2,06 y 2,18 UM/tx; ninguna supera al estático de forma estable |
| Drift en P(X) | domain classifier con AUC de 0,551 (alerta en 0,75) |
| Drift en P(y\|X) | 16 días más de antigüedad cuestan 0,106 UM/tx con el volumen fijo, en las 18 combinaciones de robustez |
| Cupo de revisión | 150 casos diarios respetados los 62 días del test |
| Latencia sobre HTTP | p95 de 90,1 ms |

El drift de IEEE-CIS está en la relación entre features y fraude y es menor que el
costo de entrenar con menos datos, por eso reentrenar sin descartar histórico rinde
más que las ventanas deslizantes. El análisis completo está en el informe.

## Instalación y ejecución

```bash
uv python install 3.12
uv venv --python 3.12
uv pip install -e ".[serving,dev]"

python -m fraud_adaptive data instructions   # pasos para descargar IEEE-CIS desde Kaggle
python -m fraud_adaptive --run-id v3 all     # cadena completa, reanudable
```

Los CSV de Kaggle, `data/processed/`, `models/` y `runs/` no se versionan. La
descarga requiere aceptar las reglas de la competencia y un token personal; solo se
usan `train_transaction.csv` y `train_identity.csv`. Sin acceso a Kaggle,
`data surrogate` genera un dataset sintético con el mismo esquema para ejecutar el
pipeline; sus artefactos quedan marcados con `data_source: sintetico_sustituto`.
Los detalles están en [docs/reproducibilidad.md](docs/reproducibilidad.md).

## Comandos

| Comando | Qué hace |
|---|---|
| `data inspect` | Hash, esquema y origen de las fuentes |
| `data instructions` | Pasos de descarga desde Kaggle |
| `data surrogate` | Genera el dataset sintético sustituto |
| `data prepare` | Join auditado, relojes y features causales en Parquet |
| `train tune` | 18 fits: 3 familias × 3 configuraciones × 2 folds temporales |
| `train static` | Modelos finales, calibración, política y prerregistro |
| `adapt run` | Backtest de las siete estrategias sobre los mismos eventos |
| `selftest adwin` | Prueba ADWIN sobre series con cambio conocido |
| `report build` | Tablas, figuras y reportes de evidencia |
| `package export --strategy E15` | Exporta un paquete de E15, W30, W60 o W90 |
| `serve --package <dir>` | Servicio HTTP local con `/health` y `/predict` |
| `replay --package <dir>` | Replay secuencial con ledger SQLite |
| `verify --against <run>` | Compara dos corridas |
| `deliver` | Ejecuta los notebooks, exporta el PDF y escribe el manifiesto |
| `all` | Cadena completa |

## Estructura

```
configs/            parámetros de datos, protocolo temporal, modelos, decisión y adaptación
data/manifests/     hashes y esquema de las fuentes, particiones temporales
docs/               contrato, protocolo, decisiones de diseño, arquitectura y antecedentes
deploy/             runbook de GCP, política de promoción y rollback
notebooks/          EDA temporal, modelos y adaptación, con salidas de la corrida v3
reports/            informe, infografía, guion, tablas, figuras y evidencia
src/fraud_adaptive/
├── data.py         carga, validación, join 1:1, relojes y EDA
├── features.py     features causales por proxy de entidad
├── splits.py       roles temporales y madurez de etiquetas
├── models.py       preprocesamiento por fit y tres familias de modelos
├── calibration.py  Platt por versión, Brier y ECE
├── decision.py     costos esperados, tres acciones y cupo
├── metrics.py      métricas técnicas, de decisión y sociales; bootstrap por bloques
├── drift.py        KS/PSI, domain classifier y ADWIN
├── adaptation.py   paquetes de modelo y gates de promoción
├── backtest.py     simulación prequential con dos relojes
├── monitoring.py   alertas, escalamiento y rollback
├── serving.py      API HTTP
├── replay.py       orquestador local con ledger transaccional
├── pipeline.py     encadenamiento de fases
├── reporting.py    tablas, figuras y manifiestos
├── export.py       exportación del informe a PDF
├── tracking.py     manifests, hashes, presupuesto y checkpoints
└── synthetic.py    generador del dataset sustituto
tests/              150 pruebas automáticas
```

## Protocolo temporal

```
día:  0        30   45   60      69   76   83   90        120  135  150  165  182
      |---------|----|----|-------|----|----|----|---------|----|----|----|----|
      [ tuning fold 1 ]
      [ tuning fold 2      ]
      [     fit base / S0          ]
                                   [cal][pol][ H ]
                                                  [warmup X ]
                                                            [ B1][ B2][ B3][ B4 ]
```

En desarrollo se eligen la familia, la política, los umbrales y la ventana, y el hash
de prerregistro los sella antes de abrir el test. En cada actualización `T` de
{120, 135, 150, 165}, con `c = T − 30`:

| Rol | Intervalo | Días |
|---|---|---|
| Predictor | `[c−W, c−14)` | 16 con W30, 46 con W60, 76 con W90 |
| Calibrador | `[c−14, c−7)` | 7 |
| Validación de promoción | `[c−7, c)` | 7 |

## Pruebas

```bash
python -m pytest
```

Hay 150 pruebas: 149 pasan y una se omite porque el dataset reducido de las pruebas
no alcanza el soporte mínimo en la cola de calibración.

| Archivo | Qué verifica |
|---|---|
| `test_point_in_time_features.py` | Causalidad de las features y empates temporales |
| `test_temporal_integrity.py` | Roles disjuntos y preprocesamiento por fit |
| `test_delayed_feedback.py` | Que ninguna decisión use etiquetas inmaduras |
| `test_decision_capacity.py` | Costos, cupo y admisión causal a la cola |
| `test_adaptation_windows.py` | Ventanas de adaptación y detectores de drift |
| `test_promotion_holdout.py` | Rechazo del gate con una validación contaminada |
| `test_backtest_end_to_end.py` | Recorrido completo del backtest con datos reducidos |
| `test_checkpoint_resume.py` | Reanudación, presupuesto y escritura atómica |
| `test_serving_contract.py` | Contrato HTTP, 503 sin modelo y paridad con el cálculo offline |
| `test_replay_idempotency.py` | Idempotencia del ledger y atomicidad del cupo |
| `test_verification.py` | Comparación entre corridas |
| `test_credentials.py` | Carga del token desde `.env` sin exponerlo en logs |

Una segunda corrida independiente (`v3_repro`) reprodujo los 18 fits de tuning, el
hash de prerregistro y el costo de S0, E15, W30, W60 y W90 sin diferencias
([reports/verificacion_reproducibilidad.json](reports/verificacion_reproducibilidad.json)).
La igualdad está comprobada en la misma máquina; entre máquinas distintas pueden
aparecer diferencias de punto flotante.

## Despliegue

Todo lo del repositorio se ejecuta en local y ningún comando crea recursos en la
nube.

| Entregado y verificado | Paso manual del equipo |
|---|---|
| Paquete del modelo con hashes | Publicar la imagen en Artifact Registry |
| Imagen Docker construida y probada | Crear el servicio en Cloud Run |
| Servicio HTTP y replay con ledger | Apuntar el replay al endpoint remoto |
| Runbook y política de promoción | Promover, revertir y cerrar recursos |

[deploy/arquitectura_gcp.mmd](deploy/arquitectura_gcp.mmd) distingue lo entregado del
diseño futuro (Pub/Sub, Firestore, BigQuery y Cloud Scheduler).

## Documentación

| Documento | Contenido |
|---|---|
| [docs/contrato_sistema.md](docs/contrato_sistema.md) | Objetivos, métricas, política económica y límites de autonomía |
| [docs/protocolo_experimental.md](docs/protocolo_experimental.md) | Particiones, controles de fuga, prerregistro y gates |
| [docs/decisiones_de_diseno.md](docs/decisiones_de_diseno.md) | Decisiones de diseño y la medición que sustenta cada una |
| [docs/registro_decisiones.md](docs/registro_decisiones.md) | Ambigüedades del enunciado y cómo se resolvieron |
| [docs/reproducibilidad.md](docs/reproducibilidad.md) | Cómo repetir la corrida y qué se garantiza |
| [docs/arquitectura.dot](docs/arquitectura.dot) | Fuente del diagrama de arquitectura |
| [docs/antecedentes/](docs/antecedentes/) | Propuesta del avance y benchmark de selección del dataset |
| [deploy/gcp_runbook.md](deploy/gcp_runbook.md) | Pasos de despliegue manual |
| [deploy/promocion_rollback.md](deploy/promocion_rollback.md) | Criterios de promoción y rollback |

## Límites

- Los costos son simulados con `c_FP = 25`, `c_R = 1`, `r_H = 0,90` y `f_H = 0,02`.
- El analista es simulado y su veredicto no se usa para entrenar.
- Los segmentos son variables de negocio; IEEE-CIS no tiene atributos protegidos.
- El cupo está garantizado para un orquestador secuencial.
- La latencia se midió en local.
- Es un backtest retrospectivo, y la selección del dataset ya había examinado
  periodos tardíos.
- `L = 30` es un supuesto de simulación.
