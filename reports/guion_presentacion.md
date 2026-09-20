# Guion de exposición. 18 minutos más 5 a 10 de preguntas

**Destinatario** los cinco integrantes del equipo que exponen.

El docente puede pedir a cualquier integrante que explique cualquier componente.
El reparto siguiente asigna turnos, no áreas de conocimiento.

---

## Reparto y tiempos

| Tiempo | Bloque | Expone |
|---|---|---|
| 0:00 a 2:00 | Problema de decisión y compromisos del avance | A |
| 2:00 a 4:30 | Datos, integración y señales de drift | B |
| 4:30 a 6:30 | Cortes temporales y ausencia de fuga | B y C |
| 6:30 a 9:00 | Tres modelos, calibración y política económica | C |
| 9:00 a 12:30 | Experimento central y resultado negativo | D |
| 12:30 a 15:00 | Demo del replay y propuesta de despliegue | E |
| 15:00 a 17:00 | Infografía, equidad operativa y límites | A y E |
| 17:00 a 18:00 | Conclusión y validación futura | A |

Con cuatro integrantes, D asume el bloque de despliegue con revisión de A.

---

## 0:00 a 2:00. Problema (A)

El sistema decide entre aprobar, revisar y bloquear bajo cuatro restricciones. La
etiqueta llega 30 días tarde. La capacidad humana es de 150 revisiones diarias.
Los errores cuestan cantidades distintas. La distribución cambia con el tiempo.

Los objetivos O1 a O5 del avance se conservan y se convierten en criterios
verificables.

Conviene anticipar el resultado principal desde el inicio. El olvido por ventanas
fijas no mejora el costo sobre IEEE-CIS. Presentarlo al comienzo evita que parezca
una concesión forzada durante las preguntas.

---

## 2:00 a 4:30. Datos y señales (B)

Dos fuentes integradas por left join uno a uno, con el número de filas invariante
y cero identificadores huérfanos. La cobertura de identidad es del 24,42 % y su
ausencia se conserva como información en `has_identity`.

`TransactionDT` es un delta en segundos, no una fecha. La hora derivada es
relativa y no identifica hora local ni día laboral.

Mostrar `reports/figures/eda_temporal.png`.

El dato que conviene destacar es la estabilidad. El domain classifier obtiene un
AUC de 0,654, por debajo del umbral de alerta de 0,75. Las covariables cambian
poco durante los 182 días. Ese hecho explica el resultado del experimento central.

---

## 4:30 a 6:30. Cortes temporales (B y C)

Dibujar la línea temporal.

```
0        60   69   76   83   90        120  135  150  165  182
|---------|----|----|----|---------|----|----|----|----|
 desarrollo   fit  cal  pol   H   warmup   B1   B2   B3   B4
```

Cuatro roles disjuntos por versión en el orden predictor, calibrador, política y
validación. Las tres reservas de 7 días son idénticas para todas las estrategias,
lo que permite atribuir una diferencia de costo al tamaño de ventana.

Ante una pregunta sobre fuga, la respuesta es que existen 137 pruebas
automáticas. La más directa invierte todas las etiquetas aún inmaduras y comprueba
que el predictor no cambia.

---

## 6:30 a 9:00. Modelos y decisión (C)

| Familia | AP | Costo UM/tx |
|---|---|---|
| LightGBM `lgbm_31` | 0,6105 | 1,1963 |
| Random Forest `rf_200_20` | 0,5404 | 1,4957 |
| Logística `lr_c1_bal` | 0,3986 | 1,5325 |

La familia se elige por costo y no por AP. Son criterios distintos.

La calibración admite una demostración cuantitativa. La configuración ganadora de
la logística usa `class_weight=balanced` e infla los puntajes hasta un Brier de
0,1499 y un ECE de 0,2939. Platt los corrige a 0,0317 y 0,0044, una mejora del ECE
por un factor de 67. Sin esa corrección, `p * monto` produciría un exceso de
bloqueos que el AP no reflejaría.

Los umbrales resultan de minimizar el costo y no se fijan en 0,5.

---

## 9:00 a 12:30. Experimento central (D)

Mostrar `reports/figures/drift_adaptacion.png`. Cinco estrategias, los mismos
eventos, la misma semilla y la misma política.

| Estrategia | AP | Costo UM/tx | Δ frente a S0 con IC 95 % |
|---|---|---|---|
| E15 referencia | 0,4870 | 1,4961 | −0,032 [−0,058 y −0,009] |
| W90 | 0,4775 | 1,5230 | −0,005 [−0,035 y +0,026] |
| S0 referencia | 0,4799 | 1,5282 | referencia |
| W60 elegida | 0,4636 | 1,6262 | +0,098 [+0,009 y +0,196] |
| W30 | 0,4267 | 1,7924 | +0,264 [+0,174 y +0,367] |

El mensaje central es que el olvido no compensa en este dataset. La ventana
elegida en desarrollo resulta un 6,4 % más cara que el modelo estático. La ventana
más corta es la peor de las cinco.

Conviene explicar dos puntos con cuidado.

El primero es que la ventana se eligió en desarrollo, sobre la reserva de política,
sin observar el test. W60 obtuvo 1,1909 UM/tx frente a 1,1963 de W90 y 1,4743 de
W30. El desarrollo ya identificó que la ventana corta era la peor opción. Lo que
no pudo anticipar es que ninguna superaría al estático.

El segundo es la explicación del resultado. Las covariables son estables, con un
AUC de S1 de 0,654, mientras que ADWIN registra 33 detecciones de deriva en el
error. Existe deriva, pero reentrenar con ventanas cortas no la corrige. W30
entrena con 1 291 fraudes frente a los 8 269 de W90, de modo que el costo en
varianza supera el beneficio de frescura.

Conviene mencionar también que ninguna estrategia alcanzó el FPR máximo del 1 % ni
la precisión mínima del 80 % en test con los umbrales congelados. El umbral no se
ajustó porque hacerlo convertiría un objetivo incumplido en un resultado ajustado
a posteriori.

---

## 12:30 a 15:00. Demo y despliegue (E)

Ejecutar en vivo, con capturas de respaldo disponibles.

```bash
python -m fraud_adaptive replay --package models/W60_T165 --max-events 5000 --fixtures
```

Señalar que las tres acciones aparecen sobre datos reales, que el cupo se respeta
en 150 de 150 y que la prueba de idempotencia pasa.

La separación de responsabilidades merece explicación. El servicio calcula la
decisión. El replay la confirma y reserva el cupo. Un cliente arbitrario podría
declarar un cupo falso, por lo que la demo exige un único orquestador. Hacer el
cupo autoritativo dentro del servicio requeriría estado distribuido, que figura
como diseño futuro.

La imagen Docker está construida y verificada. Arranca con HEALTHCHECK en verde,
devuelve probabilidades idénticas al cálculo offline en 5 de 5 casos, alcanza un
p95 de 87 ms sobre HTTP y responde 503 sin paquete montado. Al equipo le queda la
publicación en el registro y el despliegue, no la construcción.

Pub/Sub, BigQuery y Cloud Scheduler figuran en el diagrama como diseño, sin
recursos creados.

---

## 15:00 a 17:00. Infografía y límites (A y E)

Mostrar `reports/infografia.pdf`.

Equidad operativa. De 14 segmentos con soporte suficiente, 6 superan el umbral de
2 puntos porcentuales. La mayor diferencia corresponde a `DeviceType` desktop con
6,88 puntos. Conviene precisar que se trata de disparidad operativa sobre
variables de negocio y no de una auditoría demográfica, porque IEEE-CIS no
contiene atributos protegidos verificables.

El gate social bloqueó la promoción de las cinco estrategias en T igual a 135. Los
gates no son decorativos.

La restricción dominante es la capacidad. La demanda de revisión supera de forma
sistemática los 150 casos diarios, de modo que mejorar el AP presenta rendimientos
decrecientes.

---

## 17:00 a 18:00. Conclusión (A)

El sistema reduce el costo de 5,40 a 1,53 UM por transacción frente a aprobar
todo, una mejora del 71,7 %. Ese resultado proviene del modelo y de la política
económica.

El olvido por ventanas fijas no aporta mejora sobre IEEE-CIS con este protocolo.
La ventana elegida en desarrollo resulta un 6,4 % más cara que el estático y la
más corta un 17,3 % más cara. Solo la referencia expansiva mejora al estático, en
un 2,1 %.

La recomendación operativa es W60 con la advertencia de que no supera al estático
en el periodo evaluado. Un despliegue razonable mantendría el modelo estático con
monitoreo activo.

Este trabajo constituye un backtest retrospectivo y no demuestra que el sistema
funcione en el futuro.

---

# Hoja de respuestas

### El resultado es negativo. ¿El proyecto falló?

No. El objetivo era medir si el olvido compensa bajo restricciones reales, no
demostrar que compensa. El protocolo fue prerregistrado, la ventana se eligió en
desarrollo y el resultado se reporta como salió. Un resultado negativo obtenido
con rigor aporta más que uno positivo obtenido ajustando umbrales después de ver
el test.

### ¿Cómo saben que no hay fuga temporal?

Mediante 137 pruebas automáticas. Tres resultan decisivas. Alterar las filas
futuras no modifica medianas ni vocabulario del preprocesamiento. Invertir todas
las etiquetas inmaduras no cambia el predictor. Permutar los identificadores de
eventos con el mismo timestamp deja las features idénticas.

### ¿Por qué W30 rinde peor si es la más fresca?

Porque entrena con 9 días, 1 291 fraudes, frente a los 69 días y 8 269 fraudes de
W90. Bajo covariables estables, con un AUC de S1 de 0,654, el costo en varianza
supera el beneficio de frescura. El desarrollo ya lo detectó y por eso W30 no fue
la ventana elegida.

### ¿No estarán sobreajustando al elegir la ventana?

La ventana se eligió sobre la reserva de política, que pertenece a desarrollo, y
quedó congelada mediante hash antes de abrir el test. Si la configuración cambiara
entre la selección y el test, la corrida se detiene.

### ADWIN detecta 33 cambios. ¿No contradice que el olvido no sirva?

No. ADWIN observa deriva en el error, que existe. El olvido por ventanas fijas es
una respuesta posible a esa deriva y resulta que no la corrige en este dataset. La
deriva se concentra en P(y|X) mientras P(X) permanece estable, y una ventana corta
sobre P(X) estable solo reduce el tamaño de muestra.

### Los costos son supuestos. ¿Qué validez tienen las conclusiones?

Los supuestos están declarados y se somete el resultado a tres escenarios
económicos. El ordenamiento entre estrategias no cambia. Lo que no se afirma es un
ahorro causal sobre pagos reales.

### ¿Por qué no usan el veredicto del analista para entrenar?

Porque no constituye la verdad, sino una opinión con recall de 0,90. Además solo
se conoce cuando la etiqueta ya maduró, de modo que usarlo antes sería fuga. Se
calcula en el evaluador y no modifica ninguna decisión emitida.

### ¿Qué ocurre con las transacciones bloqueadas?

Constituye la limitación más seria, conocida como selective labels. El benchmark
usa información completa porque el dataset la contiene. En producción las
etiquetas disponibles estarían sesgadas por las decisiones previas del sistema. La
corrección mediante masked label quedó fuera del alcance.

### ¿Por qué Platt y no calibración isotónica?

La cola de calibración es de 7 días. La isotónica requiere más datos para no
sobreajustar escalones y produciría una función inestable entre versiones. Platt
tiene dos parámetros y degrada de forma predecible. La elección quedó congelada
antes del test.

### ¿Funcionaría en producción?

Con tres salvedades. El cupo está garantizado solo para un orquestador secuencial.
La latencia medida es local y no representa una región cloud. Se trata de un
backtest retrospectivo, y la selección previa del dataset ya examinó periodos
tardíos.

### ¿Qué harían distinto con más tiempo?

Evaluar cadencias distintas de 15 días, retrasos distintos de 30 y ventanas
mayores que 90. El resultado sugiere que la deriva de IEEE-CIS opera a una escala
temporal mayor que la del protocolo evaluado.
