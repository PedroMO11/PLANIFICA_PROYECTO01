# Protocolo experimental

Este protocolo se fijó antes de abrir el periodo de test. Su contenido se resume en
un hash que el pipeline verifica; si la configuración cambia entre la selección y el
test, la corrida se detiene.

## 1. Parámetros fijos

| Parámetro | Valor | Observación |
|---|---|---|
| Semilla | 42 | Una sola semilla de entrenamiento |
| Retraso de etiqueta `L` | 30 días | Supuesto de simulación |
| Cadencia de reentrenamiento | 15 días | Las cadencias de 30 y 45 días se midieron aparte |
| Ventanas `W` | 30, 60 y 90 días | Incluyen el predictor y las dos reservas |
| Estrategias | S0, E15, W30, W60, W90, R46_medio, R46_antiguo | Volumen variable, antigüedad variable y referencias |
| Configuraciones por familia | 3 | Fijadas antes del tuning |
| Folds de tuning | 2, temporales | Selección interna |
| Presupuesto de cómputo | 480 minutos acumulados | Incluye reintentos; reanudar no reinicia el contador |

## 2. Relojes

| Reloj | Qué mide | Qué gobierna |
|---|---|---|
| De evento | Día de la transacción y de la decisión | Qué features son visibles |
| De disponibilidad | `available_at = día_evento + L` | Cuándo una etiqueta entra a un fit, a una métrica o a ADWIN |

Una etiqueta es elegible si `available_at < job_time`, con desigualdad estricta.

`cutoff = T − L` delimita qué días de evento pueden usarse y es el origen de los
roles. `job_time = T` es el momento del ajuste y es contra lo que se compara
`available_at`. Con T = 120 y L = 30, el último día de evento con etiqueta
confirmada es el 89. Si la madurez se comparara contra `cutoff`, solo serían
elegibles los días anteriores al 60 y las colas de calibración y validación
quedarían vacías.

## 3. Particiones

Todos los intervalos son semiabiertos `[inicio, fin)` en días relativos.

| Tramo | Intervalo | Uso |
|---|---|---|
| Tuning fold 1 | fit `[0,30)`, validación `[30,45)` | Selección interna |
| Tuning fold 2 | fit `[0,45)`, validación `[45,60)` | Selección interna con los mismos IDs para las tres familias |
| Fit base y S0 | `[0,69)` | Ajuste final con los hiperparámetros elegidos |
| Calibración inicial | `[69,76)` | Platt, sin filas del predictor |
| Política y selección de W | `[76,83)` | Umbrales, familia y ventana |
| Validación de promoción | `[83,90)` | Gate de bootstrap |
| Warmup | `[90,120)` | Solo historial de features y referencia de covariables, sin scores ni acciones |
| Test B1–B4 | `[120,135)`, `[135,150)`, `[150,165)`, `[165,182)` | Evaluación secuencial |

### Roles por versión

En cada actualización T ∈ {120, 135, 150, 165}, con `c = T − 30`:

| Rol | Intervalo | Días |
|---|---|---|
| Predictor | `[c−W, c−14)` | W − 14: 16, 46 o 76 |
| Calibrador | `[c−14, c−7)` | 7 |
| Validación de promoción (H) | `[c−7, c)` | 7 |

La política y sus umbrales se eligen una vez en desarrollo y quedan congelados, así
que cada versión usa dos reservas. Una tercera reserva de política por
actualización restaba siete días al predictor sin cumplir función; sin ella W30
entrena con 16 días en lugar de 9.

Las reservas son idénticas para todas las estrategias, de modo que una diferencia
de costo se puede atribuir al tamaño de la ventana.

En T = 120, E15 y W90 comparten el predictor `[0,76)` y se cuentan como un solo
ajuste en el costo de cómputo.

### Bloques del experimento

Variar solo `W` cambia a la vez el volumen de entrenamiento y la antigüedad de la
información. El experimento los separa en dos bloques.

| Bloque | Estrategias | Qué varía | Qué se mantiene |
|---|---|---|---|
| Volumen variable | W30, W60, W90 | Días de fit | Corte en `c−14` |
| Antigüedad variable | W60, R46_medio, R46_antiguo | Punto de corte | 46 días de fit |

`R46_medio` usa 46 días que terminan 30 días antes que los de W60. `R46_antiguo` usa
los primeros 46 días del histórico y no se reentrena. `build_version_roles` rechaza
una combinación de `lag_days` y `fit_days` que no quepa antes del corte, en lugar de
recortar la ventana.

## 4. Construcción de un paquete

1. Predictor `[c−W, c−14)`: ajusta preprocesamiento y modelo.
2. Calibrador `[c−14, c−7)`: Platt sobre los scores del predictor ya ajustado.
3. Validación `[c−7, c)`: gate de promoción.

Cada paso usa scores del anterior sobre datos que ese paso no vio, de modo que el
calibrador corrige exactamente el modelo que se despliega.

Si un rol no alcanza el soporte mínimo (fit: 200 fraudes y 2 000 legítimas;
calibración y H: 50 y 500), la versión se marca no válida. El sistema no amplía la
ventana ni usa etiquetas inmaduras, porque eso cambiaría la W que se mide.

## 5. Selección de la ventana

La ventana deslizante se elige con datos de desarrollo, antes de abrir el test,
sobre la reserva de validación de cada paquete. Compiten W30, W60 y W90, que son las
estrategias de olvido que el enunciado prioriza. Gana el menor costo observado, con
desempate a 1 % en favor de la ventana menor.

`fraud-adaptive adapt run` guarda la selección en
`runs/<run_id>/seleccion_ventana.json`. En la corrida v3:

| Estrategia | Días de fit | Costo UM/tx en validación |
|---|---|---|
| W60 | 46 | 2,2527 |
| W90 | 76 | 2,2708 |
| W30 | 16 | 2,3501 |

W60 queda elegida como ventana deslizante. El informe recomienda operar E15 a partir
de la comparación en test, donde es la estrategia más barata; esa recomendación se
declara como resultado del test y no como selección prerregistrada. Si ninguna
versión resulta válida en operación, el sistema se pausa.

## 6. Controles de fuga

Cada control tiene una prueba automática que falla si se rompe.

| Riesgo | Control | Prueba |
|---|---|---|
| Uso de los `test_*` de Kaggle | Solo se aceptan los dos `train_*` | `assert_no_forbidden_files` |
| Join de muchos a muchos | `validate="one_to_one"` y conteo invariante | `test_temporal_integrity` |
| Preprocesamiento global | Ajuste dentro de cada fit | `test_preprocesamiento_no_cambia_al_alterar_filas_futuras` |
| Ventana que incluye la fila actual | La feature se emite antes de actualizar el estado | `test_la_fila_actual_nunca_entra_en_su_propio_agregado` |
| Empates temporales | El grupo empatado lee el mismo pasado | `test_permutar_ids_empatados_no_cambia_las_features` |
| Evento futuro que altera el pasado | Estado incremental por entidad | `test_un_evento_futuro_no_altera_features_anteriores` |
| Etiquetas inmaduras | Madurez contra `job_time` | `test_mutar_etiquetas_inmaduras_no_cambia_el_predictor` |
| Roles contaminados | Conjuntos de `TransactionID` disjuntos | `test_roles_reales_no_comparten_ids` |
| Validación H contaminada | El gate rechaza si H participó en un ajuste | `test_promotion_holdout` |
| Memorización por ID o fecha | `TransactionID` y tiempo absoluto excluidos | `assert_no_forbidden_features` |
| Cupo duplicado en reintentos | Ledger transaccional idempotente | `test_replay_idempotency` |
| Reajuste tras ver el test | Hash de prerregistro | `run_adaptation` se detiene |

## 7. Detectores

| Señal | Qué detecta | Qué no puede afirmar | Disponibilidad |
|---|---|---|---|
| KS/PSI | Desplazamiento de covariables | Cambio en P(y\|X) | Mismo día |
| S1, domain classifier | Cambio en P(X) (alerta con AUC ≥ 0,75) | Cambio en P(y\|X) | Una vez por bloque |
| S2, PSI de scores | Distribución de scores desplazada | Si la causa es el entorno o un cambio de versión | Mismo día |
| S3, analista | Tasa de confirmación | No es ground truth | Al madurar la etiqueta |
| ADWIN sobre Brier | Cambio en el error | | 30 días después |

ADWIN usa `delta = 0,002` y `clock = 32`, con un detector por versión que cada
etiqueta madura actualiza una sola vez. Sus parámetros se probaron sobre dos series
sintéticas con cambio conocido (`fraud-adaptive selftest adwin`) y no se ajustaron
después de ver los datos reales.

## 8. Escalamiento de alertas

```
S1 o S2        → alerta y ampliación de la revisión, sin reajuste automático
+ S3           → recomendación de reentrenamiento
ADWIN (error)  → reentrenamiento recomendado, sujeto a autorización humana
```

## 9. Gates de promoción

1. Integridad: roles disjuntos, soporte suficiente, artefactos completos y cupo de
   150. Una falla de fuga bloquea la versión.
2. Arranque: costo en H inferior al de aprobar todo.
3. No inferioridad: costo ≤ 1,01 veces el del modelo vigente, ΔAP ≥ −0,01 y
   ΔBrier ≤ 0,005.
4. Social: el bloqueo de legítimas no crece más de 2 pp en segmentos con soporte.
5. Humano: obligatorio aunque los anteriores pasen.

Se exige no inferioridad con tolerancia al ruido. Pedir una mejora estricta en cada
ciclo llevaría a no actualizar casi nunca, o a relajar la tolerancia hasta que pase.

## 10. Incertidumbre

Los intervalos se obtienen por bootstrap de bloques contiguos de 7 días con 200
remuestreos, porque dentro de una semana los eventos comparten régimen y un
bootstrap por filas sería demasiado estrecho. Las comparaciones entre estrategias
son pareadas sobre los mismos eventos y bloques. La robustez de cada diferencia se
revisa con 6 semillas de remuestreo y 3 tamaños de bloque; la semilla de
entrenamiento es una sola.

## 11. Exposición previa del test

El benchmark que seleccionó IEEE-CIS (`docs/antecedentes/benchmark_seleccion_dataset.md`)
examinó periodos que aquí son test. Por eso el trabajo es un backtest retrospectivo.
La ventana, la familia y los umbrales se eligieron solo con desarrollo.

## 12. Fuera del alcance

Semillas de entrenamiento adicionales, repetir el benchmark de selección, L de 60,
90 o 120 días, cadencias menores de 15 días, W14, modelos incrementales, calibración
isotónica, corrección de selective labels, MLflow, Terraform, despliegue shadow o
canary y servicios distribuidos en la nube.
