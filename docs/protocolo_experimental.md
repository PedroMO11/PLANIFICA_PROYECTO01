# Protocolo experimental (prerregistro)

Este documento se fija **antes** de abrir el periodo de test. Su contenido se
resume en un hash que el pipeline verifica: si la configuración cambia entre la
selección y el test, la corrida se detiene en lugar de producir un resultado
retuneado.

---

## 1. Núcleo cerrado

| Parámetro | Valor | Razón |
|---|---|---|
| Semilla | 42 | Única en el núcleo. Una semilla **no** permite afirmar variabilidad entre semillas |
| Retraso de etiqueta `L` | 30 días | Supuesto de simulación, no un plazo regulatorio |
| Cadencia de reentrenamiento | 15 días | Única cadencia evaluada |
| Ventanas `W` | {30, 60, 90} días | Incluyen predictor **y** las dos reservas |
| Estrategias | S0, E15, W30, W60, W90, R46_medio, R46_antiguo | Dos bloques: volumen variable y antigüedad variable |
| Configuraciones por familia | 3 | Congeladas antes del tuning |
| Folds de tuning | 2, forward | Selección interna, nunca resultado final |
| Presupuesto de cómputo | 480 min acumulados | Incluye reintentos; reanudar no reinicia el contador |

---

## 2. Los dos relojes

| Reloj | Qué mide | Gobierna |
|---|---|---|
| **De evento** | Día en que ocurrió la transacción y se emitió la decisión | Qué features son visibles |
| **De disponibilidad** | `available_at = día_evento + L` | Cuándo una etiqueta entra a un fit, a una métrica o a ADWIN |

Regla de elegibilidad: `available_at < job_time`, con desigualdad **estricta**.

**Distinción que rompe el pipeline si se confunde.** `cutoff = T − L` delimita qué
días de evento pueden usarse y es el origen desde el que se miden los roles.
`job_time = T` es cuando corre el ajuste y es contra lo que se compara
`available_at`. Con T=120 y L=30, el último día de evento con etiqueta confirmada
es el **89**. Comparar contra `cutoff` en lugar de contra `job_time` exigiría
`día < 60` y dejaría vacías las colas de calibración, política y validación.

---

## 3. Particiones

Todos los intervalos son semiabiertos `[inicio, fin)` en días relativos.

| Tramo | Intervalo | Uso permitido |
|---|---|---|
| Tuning fold 1 | fit `[0,30)`, val `[30,45)` | Selección interna |
| Tuning fold 2 | fit `[0,45)`, val `[45,60)` | Selección interna, mismos IDs de validación para las tres familias |
| Fit base / S0 | `[0,69)` | Ajuste final con hiperparámetros ya elegidos |
| Calibración inicial | `[69,76)` | Platt, sin filas del predictor |
| Política y selección de W | `[76,83)` | Umbrales, familia y ventana. **Métricas de selección, no resultados** |
| Validación de promoción | `[83,90)` | Gate de bootstrap. **No es test final** |
| Warmup (solo X) | `[90,120)` | Calienta historial y referencia. **Sin scores ni acciones** |
| Test B1–B4 | `[120,135)`, `[135,150)`, `[150,165)`, `[165,182)` | Evaluación secuencial |

### Roles por versión en cada actualización T ∈ {120, 135, 150, 165}

Con `c = T − 30`:

| Rol | Intervalo | Días |
|---|---|---|
| Predictor | `[c−W, c−14)` | W − 14 (= 16, 46 o 76) |
| Calibrador | `[c−14, c−7)` | 7 |
| Validación de promoción (H) | `[c−7, c)` | 7 |

Hay **dos** reservas por versión, no tres. Los umbrales de referencia se eligen una
sola vez en desarrollo y quedan congelados, de modo que reservar una ventana de
política en cada actualización excluía siete días del fit sin cumplir función.
Liberarla devuelve esos días al predictor y beneficia sobre todo a las ventanas
cortas, que son las que el experimento evalúa.

Las dos reservas son **idénticas** para todas las estrategias. Eso es lo que aísla
el efecto del tamaño de ventana: si W30 tuviera una cola de calibración distinta de
W90, la diferencia de costo dejaría de ser atribuible al olvido.

En T=120, S0, E15 y W90 comparten exactamente el predictor `[0,76)`: es **un solo
ajuste**, no tres, y así se cuenta en el costo de cómputo.

### 3.b Los dos bloques del experimento

Variar solo `W` cambia a la vez el volumen de entrenamiento y la antigüedad de la
información, de modo que una diferencia de costo no sería atribuible a ninguno de
los dos. El experimento los separa.

| Bloque | Estrategias | Qué varía | Qué se mantiene |
|---|---|---|---|
| Volumen variable | W30, W60, W90 | días de fit | punto de corte en `c−14` |
| Antigüedad variable | W60, R46_medio, R46_antiguo | punto de corte | 46 días de fit |

`R46_medio` toma 46 días terminando 30 días antes del corte. `R46_antiguo` toma los
primeros 46 días del histórico. `build_version_roles` rechaza de forma explícita una
combinación de `lag_days` y `fit_days` que no quepa en el corte, en lugar de truncar
la ventana en silencio.

---

## 4. Orden de construcción de un paquete

No es intercambiable:

1. **Predictor** `[c−W, c−14)` — ajusta preprocesamiento y modelo
2. **Calibrador** `[c−14, c−7)` — Platt sobre scores del predictor ya ajustado
3. **Validación** `[c−7, c)` — gate de promoción

La política no consume una reserva propia. Se congela en desarrollo y consiste en
el modelo de costos, cuyo `c_FP` se calibra una sola vez contra el objetivo de
bloqueo de legítimas.

Cada paso usa scores producidos por el anterior sobre datos que ese paso no vio.
Invertir el orden haría que el calibrador corrigiera un modelo distinto del que se
despliega.

**Si un rol no alcanza el soporte mínimo** (fit: 200 fraudes / 2 000 legítimas;
calibración y H: 50 / 500), la versión se marca **no válida**. No se amplía la
ventana ni se incorporan etiquetas inmaduras: eso cambiaría en silencio la W que
se está midiendo.

## 4.b Selección de la ventana desplegable

La ventana se elige con datos de desarrollo, antes de abrir el test. La evaluación
ocurre sobre la reserva de validación `[c−7, c)`, posterior al predictor y al
calibrador de cada paquete.

Compiten únicamente W30, W60 y W90. S0 y E15 quedan excluidas porque son
referencias de comparación y el plan prohíbe entregarlas como sistema. Gana el
menor costo observado, con desempate a un 1 % en favor de la ventana menor, porque
a igualdad de costo la ventana corta reentrena con menos datos.

El comando `fraud-adaptive adapt run` ejecuta esta selección y guarda el resultado
en `runs/<run_id>/seleccion_ventana.json`. Si ninguna ventana resulta válida el
sistema queda en pausa y no degrada a S0 ni a E15.

Sobre IEEE-CIS la selección eligió W60 con 1,1909 UM/tx, frente a 1,1963 de W90 y
1,4743 de W30. Que otra estrategia obtenga mejor costo en el test es un
diagnóstico retrospectivo y no autoriza a cambiar la recomendación.

---

## 5. Controles de fuga verificados

Cada fila tiene una prueba automática que falla si el control se rompe.

| Riesgo | Control | Prueba |
|---|---|---|
| Uso de los `test_*` de Kaggle | Allowlist de los dos `train_*` | `assert_no_forbidden_files` |
| Join many-to-many | `validate="one_to_one"` + conteo invariante | `test_temporal_integrity` |
| Preprocesamiento global | Ajuste dentro del fit | `test_preprocesamiento_no_cambia_al_alterar_filas_futuras` |
| Ventana que incluye la fila actual | Emitir antes de actualizar estado | `test_la_fila_actual_nunca_entra_en_su_propio_agregado` |
| Empates temporales | El grupo lee el mismo pasado; empuje canónico por ID | `test_permutar_ids_empatados_no_cambia_las_features` |
| Evento futuro que altera el pasado | — | `test_un_evento_futuro_no_altera_features_anteriores` |
| Etiquetas inmaduras | Madurez contra `job_time` | `test_mutar_etiquetas_inmaduras_no_cambia_el_predictor` |
| Roles contaminados | Cuatro conjuntos de IDs disjuntos | `test_roles_reales_no_comparten_ids` |
| H contaminado | Gate rechaza si H tocó fit/Platt/política | `test_promotion_holdout` |
| Memorización por ID | `TransactionID` y tiempo absoluto prohibidos | `assert_no_forbidden_features` |
| Cupo duplicado en reintentos | Ledger transaccional idempotente | `test_replay_idempotency` |
| Retuning tras ver el test | Hash de prerregistro verificado | `run_adaptation` aborta |

---

## 6. Detectores y su alcance

| Señal | Qué prueba | Qué **no** prueba | Disponibilidad |
|---|---|---|---|
| **KS/PSI** | Desplazamiento de covariables | Cambio en P(y\|X) | Mismo día |
| **S1** domain classifier | P(X) cambió (AUC ≥ 0.75) | Cambio en P(y\|X) | Una vez por bloque |
| **S2** scores | Distribución de scores desplazada | Que la causa sea el entorno y no un cambio de versión | Mismo día |
| **S3** analista | Tasa de confirmación | Ground truth; no es feedback rápido | Solo al madurar `y` |
| **ADWIN** sobre Brier | Cambio en el **error real** | — | L = 30 días después |
| **S4** stale-model gap | — | Antecedente histórico; no se recomputa | — |

ADWIN: `delta=0.002`, `clock=32`, un detector por versión, cada etiqueta madura lo
actualiza **exactamente una vez**. Sus parámetros se prueban contra dos streams
sintéticos con cambio conocido (`fraud-adaptive selftest adwin`) y **no** se
retocan después de mirar los datos reales.

---

## 7. Escalamiento de alertas

```
S1 o S2        → alerta y abstención (ampliar revisión), sin retuning automático
+ S3           → recomendación de reentrenamiento
ADWIN (error)  → reentrenamiento recomendado, con autorización humana
```

Ninguna alerta despliega por sí sola.

---

## 8. Gates de promoción

1. **Integridad**: roles disjuntos, soporte suficiente, artefactos completos, cupo ≤ 150. Una falla de fuga bloquea la versión.
2. **Bootstrap inicial**: costo en H ≤ costo de aprobar todo.
3. **No inferioridad**: costo ≤ 1.01 × champion, ΔAP ≥ −0.01, ΔBrier ≤ 0.005.
4. **Social**: el bloqueo de legítimas no crece > 2 pp en segmentos con soporte.
5. **Humano**: obligatorio aunque todo lo anterior pase.

La regla es de **no inferioridad tolerante a ruido**, no de superioridad: exigir
una mejora en cada ciclo llevaría a no actualizar nunca bajo ruido, o a ajustar la
tolerancia hasta que pase.

---

## 9. Incertidumbre

Intervalos por **bootstrap de bloques contiguos de 7 días**, 200 remuestreos. El
bootstrap i.i.d. sobre filas sería demasiado optimista: dentro de una semana los
eventos comparten régimen. Las comparaciones entre estrategias son **pareadas**
sobre los mismos eventos y los mismos bloques.

Es un intervalo **descriptivo de una sola semilla**. No mide variabilidad entre
semillas ni entre inicializaciones.

---

## 10. Exposición exploratoria previa (declarada)

La selección de IEEE-CIS en el benchmark histórico (`concept_drift_findings.md`)
ya examinó periodos tardíos del dataset, incluidos bloques que aquí son test. Se
declara como limitación: **esto sigue siendo un backtest retrospectivo, no una
validación prospectiva**. La ventana, la familia y los umbrales se eligen solo con
desarrollo, y el test final no se usa para elegir un ganador desplegable.

---

## 11. Fuera del núcleo

Extensiones que **no** son criterio de cierre: semillas 43/44, benchmark v2 y
ablaciones D*, L ∈ {60, 90, 120}, cadencias 7/30, W14, modelos incrementales,
calibración isotónica, masked-label, MLflow, Terraform, shadow/canary y servicios
cloud distribuidos.
