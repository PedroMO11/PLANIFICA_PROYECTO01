# Sistema adaptativo de decisión ante fraude transaccional

Proyecto del curso **Planificación y Toma de Decisiones en IA** (UTEC).
Dataset: IEEE-CIS Fraud Detection.

El sistema no se limita a predecir fraude: **decide** entre aprobar, revisar o
bloquear cada transacción minimizando el costo económico esperado bajo una
capacidad limitada de revisión humana, y se **adapta** al paso del tiempo mediante
olvido por ventanas fijas, con etiquetas que llegan 30 días tarde.

---

## Qué problema resuelve

Un modelo entrenado hoy deja de servir mañana, y no se entera hasta un mes después.

| Restricción real | Cómo la enfrenta el sistema |
|---|---|
| La etiqueta confirmada llega **30 días** después del evento | Dos relojes separados: el de evento y el de disponibilidad. Nada se entrena ni se mide con una etiqueta que aún no maduró |
| Solo se pueden revisar **150 casos al día** | Cola con cupo reservado de forma atómica, prioridad entre los casos ya llegados y desborde a la acción automática más barata |
| Un falso positivo cuesta fricción; un falso negativo cuesta el monto | Política que compara costos esperados, no un umbral de 0.5 |
| La distribución cambia con el tiempo | Ventanas de olvido W30/W60/W90 con cadencia de 15 días, frente a referencias estática y expansiva |
| Reentrenar de forma autónoma es inaceptable | Toda promoción pasa por gates verificables **y** por una persona |

---

## Inicio rápido

```bash
# 1. Entorno (usa uv)
uv python install 3.12
uv venv --python 3.12
uv pip install -e ".[serving,dev]"

# 2. Datos
python -m fraud_adaptive data instructions   # para los datos reales de Kaggle
# o, si no tienes acceso a Kaggle:
python -m fraud_adaptive data surrogate --scale 1.0

# 3. Cadena completa (reanudable)
python -m fraud_adaptive all
```

> ### ⚠️ Sobre los datos de esta corrida
>
> Los resultados incluidos se obtuvieron sobre un **dataset sustituto sintético**,
> porque IEEE-CIS exige un token individual de Kaggle que no puede versionarse.
> El sustituto replica el esquema (394 columnas), el eje temporal (182 días,
> ~3 200 tx/día), la prevalencia (~3,5 %) y la cobertura de identidad (~24 %), con
> drift inyectado de parámetros conocidos.
>
> **Ninguna cifra sobre datos sustitutos describe el fraude real.** Cada artefacto
> lleva `data_source: sintetico_sustituto` en su manifest y el rótulo impreso en
> las figuras. Al colocar los CSV reales y ejecutar con `--rebuild`, los mismos
> comandos producen resultados reales.

---

## Comandos

| Comando | Qué hace |
|---|---|
| `data inspect` | Hash, esquema y origen de las fuentes |
| `data instructions` | Pasos de descarga desde Kaggle (no ejecuta nada) |
| `data surrogate` | Genera el dataset sustituto |
| `data prepare` | Join auditado, relojes, features causales → Parquet |
| `train tune` | 18 fits: 3 familias × 3 configuraciones × 2 folds forward |
| `train static` | 3 modelos finales, Platt, umbrales por costo, familia congelada |
| `adapt run` | Backtest de las 5 estrategias sobre los mismos eventos |
| `selftest adwin` | Prueba el detector contra streams con cambio conocido |
| `report build` | Tablas, figuras y documentos de evidencia |
| `serve --package <dir>` | Servicio HTTP local (`/health`, `/predict`) |
| `replay --package <dir>` | Replay secuencial con ledger SQLite |
| `all` | La cadena completa |

---

## Cómo está construido

```
src/fraud_adaptive/
├── tracking.py     manifests, hashes, presupuesto y checkpoints (sustituye a MLflow)
├── data.py         carga, validación, join 1:1, relojes, EDA, KS/PSI
├── features.py     features causales por entidad, point-in-time
├── splits.py       roles temporales, elegibilidad de etiquetas  ← árbitro de causalidad
├── models.py       preprocesamiento por fold + tres familias
├── calibration.py  Platt por versión, Brier, confiabilidad, ECE
├── decision.py     costos esperados, tres acciones, cupo, umbrales
├── metrics.py      métricas técnicas, de decisión y sociales; bootstrap por bloques
├── drift.py        KS/PSI, S1, S2, S3, ADWIN
├── adaptation.py   construcción de paquetes y gates de promoción
├── backtest.py     replay prequential con dos relojes
├── monitoring.py   política de alertas, escalamiento y rollback
├── serving.py      API HTTP
├── replay.py       orquestador local con ledger transaccional
├── pipeline.py     encadenado de fases
└── synthetic.py    generador del dataset sustituto
```

### Cinco invariantes que el código hace cumplir

No son intenciones documentadas: cada una tiene una prueba que falla si se rompe.

1. **Ninguna transformación se ajusta fuera del tramo de fit de su versión.**
   Alterar las filas futuras no mueve medianas, escalado ni vocabulario.
2. **Una etiqueta solo es visible cuando `available_at < job_time`**, con
   desigualdad estricta. Invertir las etiquetas inmaduras no cambia el predictor.
3. **Las features se emiten antes de actualizar el estado.** Un evento futuro de
   monto extremo no altera ninguna fila anterior, y los eventos con el mismo
   timestamp leen todos el mismo pasado.
4. **Los cuatro roles por versión son disjuntos**, verificado sobre los
   `TransactionID` reales y no solo sobre los intervalos.
5. **El cupo se reserva de forma atómica e idempotente.** Reenviar un evento
   devuelve su decisión anterior sin consumir un segundo cupo.

---

## El protocolo temporal en una imagen

```
día:  0        30   45   60      69   76   83   90        120  135  150  165  182
      |---------|----|----|-------|----|----|----|---------|----|----|----|----|
      [ tuning fold 1 ]
      [ tuning fold 2      ]
      [     fit base / S0          ]
                                   [cal][pol][ H ]
                                                  [warmup X ]
                                                            [ B1][ B2][ B3][ B4 ]
                                                              ↑    ↑    ↑    ↑
                                                         actualizaciones (cadencia 15)
```

En cada actualización `T`, con `c = T − 30`:

| Rol | Intervalo | Días |
|---|---|---|
| Predictor | `[c−W, c−21)` | 9 / 39 / 69 |
| Calibrador | `[c−21, c−14)` | 7 |
| Reserva de política | `[c−14, c−7)` | 7 |
| Validación de promoción | `[c−7, c)` | 7 |

Las tres reservas son idénticas para todas las estrategias: eso es lo que hace que
una diferencia de costo sea atribuible al tamaño de ventana y no a otra cosa.

---

## Documentación

| Documento | Para qué |
|---|---|
| [docs/contrato_sistema.md](docs/contrato_sistema.md) | Objetivos, métricas, política económica y límites de autonomía |
| [docs/protocolo_experimental.md](docs/protocolo_experimental.md) | Prerregistro: particiones, controles de fuga, gates |
| [docs/reproducibilidad.md](docs/reproducibilidad.md) | Cómo repetir la corrida y qué se garantiza |
| [docs/decisiones_pendientes.md](docs/decisiones_pendientes.md) | Registro de decisiones resueltas (C1–C28) |
| [docs/arquitectura.mmd](docs/arquitectura.mmd) | Diagrama del sistema |
| [deploy/gcp_runbook.md](deploy/gcp_runbook.md) | Pasos manuales de despliegue para el equipo |
| [deploy/promocion_rollback.md](deploy/promocion_rollback.md) | Cuándo promover, cuándo revertir y quién decide |
| [reports/informe_final.md](reports/informe_final.md) | Informe de 8 páginas |

---

## Local frente a nube

**Todo lo de este repositorio es local.** Ningún comando crea recursos cloud,
ejecuta `gcloud` ni despliega nada.

| Entregado y verificado aquí | Ejecuta el equipo a mano |
|---|---|
| Paquete del modelo con hashes | `docker build` y `push` |
| `Dockerfile` y contrato de Cloud Run | Crear el servicio en Cloud Run |
| Servicio HTTP y replay con ledger | Apuntar el replay al endpoint remoto |
| Runbook y política de promoción | Promover, revertir y cerrar recursos |

El diagrama [deploy/arquitectura_gcp.mmd](deploy/arquitectura_gcp.mmd) distingue
con línea continua lo entregado y con línea punteada el diseño futuro. Pub/Sub,
Firestore, BigQuery y Cloud Scheduler **no** están implementados.

---

## Pruebas

```bash
python -m pytest
```

120 pruebas (119 pasan; 1 se omite porque el sustituto reducido de las pruebas no
tiene soporte suficiente en la cola de calibración). Agrupadas por la propiedad que
protegen:

| Archivo | Qué protege |
|---|---|
| `test_point_in_time_features.py` | Causalidad de las features y empates temporales |
| `test_temporal_integrity.py` | Roles disjuntos y preprocesamiento por fold |
| `test_delayed_feedback.py` | Que nada dependa de una etiqueta inmadura |
| `test_decision_capacity.py` | Costos, cupo y causalidad de la cola |
| `test_adaptation_windows.py` | Ventanas de olvido y detectores de drift |
| `test_promotion_holdout.py` | Que un H contaminado haga rechazar el gate |
| `test_checkpoint_resume.py` | Reanudación, presupuesto y escritura atómica |
| `test_serving_contract.py` | Contrato HTTP, 503 sin modelo, paridad offline/API |
| `test_replay_idempotency.py` | Idempotencia del ledger y atomicidad del cupo |

---

## Límites declarados

- Los costos son **simulados** bajo supuestos (c_FP=5, c_R=1, r_H=0.90, f_H=0.02).
  No hay ahorro causal medido sobre pagos reales.
- El analista es simulado; su veredicto nunca entrena al modelo y solo se calcula
  cuando la etiqueta madura.
- Los segmentos son variables de negocio. IEEE-CIS no tiene atributos protegidos
  verificables: esto es disparidad **operativa**, no auditoría demográfica.
- El cupo está garantizado para un orquestador **secuencial**. La demo no certifica
  decisiones concurrentes de producción.
- La latencia es **local**; no representa una región cloud ni un SLA.
- Es un **backtest retrospectivo**, no una validación prospectiva. La selección
  previa del dataset ya examinó periodos tardíos, y se declara como limitación.
- `L=30` es un supuesto de simulación, no un plazo regulatorio.

---

## Antecedentes (no se modifican)

`propuesta_proyecto1_final.md`, `concept_drift_findings.md` y
`concept_drift_benchmark_instructions.md` son el registro histórico del avance. El
benchmark que eligió IEEE-CIS se cita como evidencia **consistente** con concept
drift, no como prueba causal, y su preprocesamiento tenía limitaciones que se
declaran en lugar de corregirse retroactivamente.
