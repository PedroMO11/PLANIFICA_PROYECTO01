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
| 9:00 a 12:30 | Experimento central y magnitud del drift | D |
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
AUC de {AUC_DOMINIO}, por debajo del umbral de alerta de 0,75. La validación
adversarial por feature lo confirma columna a columna: de 425 features, la mediana
queda en 0,5053 y la más alta del dataset original es `C9` con 0,6002. Ninguna
alcanza el umbral de 0,70 que la competencia usaba para descartar features
inestables. Las covariables cambian poco durante los 182 días, y ese hecho explica
el resultado del experimento central.

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

{TABLA_FAMILIAS}

La familia se elige por costo y no por AP. Son criterios distintos.

La calibración admite una demostración cuantitativa. La configuración ganadora de
la logística usa `class_weight=balanced` e infla los puntajes hasta un ECE de
{ECE_CRUDO_PEOR}. Platt lo corrige a {ECE_CALIBRADO_PEOR}, una mejora por un factor
de {FACTOR_ECE}. Sin esa corrección, `p × monto` produciría un exceso de bloqueos
que el AP no reflejaría.

Conviene dedicar un minuto a la política, porque es la pieza que más mueve el
costo. La acción sale del mínimo de los tres costos esperados, no de umbrales sobre
`p`. El motivo es que el punto de indiferencia entre aprobar y
bloquear es `p* = c_FP / (monto + c_FP)` y depende del monto, de modo que ningún
corte fijo puede ser óptimo. La diferencia medida sobre la reserva de desarrollo es
de {COSTO_UMBRAL} a {COSTO_ARGMIN} UM/tx, y la comparación va en contra de la regla
económica, porque los umbrales se eligen minimizando sobre esa misma ventana.

`c_FP` tampoco se fija a mano. El costo de un falso positivo no es observable, pero
la fracción de legítimas rechazadas sí lo es y es lo que la operación restringe. Se
declara el objetivo del 1 % y se busca el menor `c_FP` que lo cumple, que resulta
{C_FP} y alcanza un {BLOQUEO_OBTENIDO} %. El orden de las
estrategias no depende de ese valor.

---

## 9:00 a 12:30. Experimento central (D)

Mostrar `reports/figures/drift_adaptacion.png`. Siete estrategias, los mismos
eventos, la misma semilla y la misma política. Tres varían el volumen de
entrenamiento y tres mantienen 46 días de fit moviendo solo el punto de corte.

{TABLA_ADAPTACION_CORTA}

El mensaje central es que el olvido no compensa en este dataset.

Conviene explicar tres puntos con cuidado.

El primero es que la ventana se eligió en desarrollo, sin observar el test.
{TEXTO_SELECCION_CORTA}

El segundo es que el experimento separa volumen de frescura por diseño, y esa
separación cambia la conclusión. Variar solo el ancho de ventana confunde las dos
cosas, porque una ventana más corta entrena con menos datos y con datos más
recientes a la vez. El bloque de antigüedad variable mantiene los 46 días de fit y
solo mueve el corte. {TEXTO_VOLUMEN_CORTO}

Lo que produce un efecto medible es la antigüedad, no el volumen. Y aun así el
efecto es pequeño: ninguna estrategia se distingue del modelo estático de forma
estable.

El tercero es la explicación. Las covariables son estables, con un AUC de S1 de
{AUC_DOMINIO}, mientras que ADWIN sí registra deriva en el error. Existe deriva,
pero reentrenar con ventanas cortas no la corrige.

Conviene mencionar también que ninguna estrategia alcanzó el FPR máximo del 1 % ni
la precisión mínima del 80 % en test con los umbrales diagnósticos congelados. El
umbral no se ajustó porque hacerlo convertiría un objetivo incumplido en un
resultado ajustado a posteriori.

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
emite la misma acción que el cálculo offline, alcanza un p95 de {P95_HTTP} ms sobre
HTTP frente a un objetivo de 300 y responde 503 sin paquete montado. Sobre mil
peticiones devuelve {ACCIONES_HTTP}. Al equipo le queda la publicación en el
registro y el despliegue, no la construcción.

Pub/Sub, BigQuery y Cloud Scheduler figuran en el diagrama como diseño, sin
recursos creados.

---

## 15:00 a 17:00. Infografía y límites (A y E)

Mostrar `reports/infografia.pdf`.

{TEXTO_EQUIDAD} Conviene precisar que se trata de disparidad operativa sobre
variables de negocio y no de una auditoría demográfica, porque IEEE-CIS no contiene
atributos protegidos verificables.

{TEXTO_GATE_SOCIAL}

La restricción dominante es la capacidad. La demanda de revisión supera de forma
sistemática los 150 casos diarios, de modo que mejorar el AP presenta rendimientos
decrecientes.

---

## 17:00 a 18:00. Conclusión (A)

{TEXTO_CIERRE}

Este trabajo constituye un backtest retrospectivo y no demuestra que el sistema
funcione en el futuro.

---

# Hoja de respuestas

### ¿El proyecto falló al no encontrar mejora con las ventanas deslizantes?

No, y conviene precisar qué quedó medido. El trabajo cuantifica el drift de
IEEE-CIS y separa sus dos componentes: la distribución de entrada es estable y la
relación entre features y fraude cambia, con un costo de {EFECTO_FRESCURA} UM/tx
por cada 30 días de antigüedad. También mide que esa magnitud queda por debajo de
lo que cuesta entrenar con menos muestra.

Lo que no se confirmó es la hipótesis concreta que el enunciado plantea, que el
olvido por ventanas fijas reduzca el costo. El protocolo fue prerregistrado y la
ventana se eligió en desarrollo, de modo que ese resultado se reporta como salió en
lugar de ajustar umbrales después de ver el test.

### ¿Cómo saben que no hay fuga temporal?

Mediante 145 pruebas automáticas. Tres resultan decisivas. Alterar las filas
futuras no modifica medianas ni vocabulario del preprocesamiento. Invertir todas
las etiquetas inmaduras no cambia el predictor. Permutar los identificadores de
eventos con el mismo timestamp deja las features idénticas.

### ¿Por qué W30 rinde peor si es la más fresca?

Porque entrena con 16 días frente a los 76 de W90, y no porque la frescura
perjudique. El experimento separa las dos cosas: el bloque de antigüedad variable
mantiene 46 días de fit en las tres estrategias y solo mueve el punto de corte.
{TEXTO_VOLUMEN_CORTO}

### ¿No estarán sobreajustando al elegir la ventana?

La ventana se eligió sobre una reserva de desarrollo y quedó congelada mediante
hash antes de abrir el test. El sello incluye la economía calibrada, porque `c_FP`
se deriva de los datos de desarrollo y sin sellarlo sería posible recalibrarlo
después de ver el test. Si la configuración cambiara entre la selección y el test,
la corrida se detiene.

### ¿Por qué la política no usa umbrales?

Porque bajo este modelo de costos ningún umbral fijo sobre `p` puede ser óptimo.
El punto de indiferencia entre aprobar y bloquear es `p* = c_FP / (monto + c_FP)`
y depende del monto, de modo que un corte fijo bloquea de más en los montos bajos
y de menos en los altos. La diferencia medida es de {COSTO_UMBRAL} a
{COSTO_ARGMIN} UM/tx, con las mismas revisiones y menos bloqueo de legítimas. Los
umbrales se siguen calculando, pero solo para cuantificar esa diferencia y para
dar un corte binario a las métricas de diagnóstico.

### ¿De dónde sale el valor de `c_FP`?

De un objetivo operativo, no de una intuición. El costo de un falso positivo no es
observable, pero la fracción de transacciones legítimas rechazadas sí lo es y es
lo que la operación restringe. Se declara el objetivo del 1 % y se busca el menor
`c_FP` que lo cumple, sobre datos de desarrollo. El valor resultante es el precio
sombra de esa restricción, y alcanza un {BLOQUEO_OBTENIDO} % de bloqueo.

### ADWIN detecta 33 cambios. ¿No contradice que el olvido no sirva?

No. ADWIN observa deriva en el error, que existe. El olvido por ventanas fijas es
una respuesta posible a esa deriva y resulta que no la corrige en este dataset. La
deriva se concentra en P(y|X) mientras P(X) permanece estable, y una ventana corta
sobre P(X) estable solo reduce el tamaño de muestra.

### Hay trabajo publicado sobre IEEE-CIS que sí encuentra que adaptarse sirve. ¿Se contradicen?

No, y conviene tener la referencia a mano. Topal, Bozanta, Erer y Başar presentaron
en el Canadian AI 2025 una replicación sobre este mismo dataset. Comparan un modelo
congelado tras los primeros 16 días contra uno reentrenado a diario sobre los
últimos 23, y obtienen un recall a FPR del 5 % de 0,03 frente a 0,37.

Esa comparación mantiene el volumen casi constante y mueve solo la antigüedad, que
es exactamente lo que aquí hace el bloque de antigüedad variable, y da el mismo
signo. La diferencia de magnitud se explica porque su modelo congelado acumula
cinco meses de antigüedad y el de aquí retrocede 30 días.

Este trabajo agrega la distinción entre reentrenar y descartar histórico. Lo que ellos
llaman reentrenamiento usa todos los datos disponibles. Aquí la estrategia
expansiva, que reentrena sin descartar, es la más barata de las siete, y ninguna
ventana deslizante la supera. La evidencia sostiene reentrenar con frecuencia y no
sostiene descartar histórico.

### Los costos son supuestos. ¿Qué validez tienen las conclusiones?

Los supuestos están declarados y se somete el resultado a tres escenarios
económicos, expresados como múltiplos del `c_FP` calibrado. El ordenamiento entre
estrategias no cambia, y se comprobó además en todo el rango de `c_FP` entre 5 y
100. Lo que no se afirma es un ahorro causal sobre pagos reales.

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

También convendría revisar la capacidad. Con 150 revisiones diarias frente a una
demanda muy superior, la cola gobierna una fracción pequeña de las decisiones y el
sistema se comporta en la práctica como un clasificador binario con un presupuesto
marginal de revisión.
