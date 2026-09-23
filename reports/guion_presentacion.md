# Guion de exposición: 18 minutos y 5 a 10 de preguntas

El docente puede pedir a cualquier integrante que explique cualquier parte del
informe, la infografía o el código. El reparto asigna turnos de exposición; todos
deben conocer el sistema completo.

## Reparto y tiempos

| Tiempo | Bloque | Expone |
|---|---|---|
| 0:00 a 2:00 | Problema, objetivos y resultado principal | A |
| 2:00 a 4:30 | Datos, integración y estabilidad de P(X) | B |
| 4:30 a 6:30 | Protocolo temporal y controles de fuga | B y C |
| 6:30 a 9:00 | Modelos, calibración y política económica | C |
| 9:00 a 12:30 | Experimento de adaptación y magnitud del drift | D |
| 12:30 a 15:00 | Demo del replay y propuesta de despliegue | E |
| 15:00 a 17:00 | Infografía, equidad operativa y riesgos | A y E |
| 17:00 a 18:00 | Conclusión | A |

Con cuatro integrantes, D asume también el despliegue y A revisa ese bloque.

---

## 0:00 a 2:00. Problema (A)

El sistema decide entre aprobar, revisar y bloquear. Cuatro condiciones definen el
problema: la etiqueta llega 30 días después, la revisión humana admite 150 casos
diarios, los dos tipos de error tienen costos distintos y los datos cambian con el
tiempo.

Los objetivos O1 a O5 del avance se mantienen y cada uno tiene un criterio
verificable en la sección 1 del informe.

Conviene dar el resultado principal desde el inicio. El costo baja de 5,40 a 1,98 UM
por transacción frente a aprobar todo. IEEE-CIS tiene un concept drift medible pero
pequeño, y la adaptación que mejor responde a él es reentrenar cada 15 días con todo
el histórico (E15). Las ventanas deslizantes se implementaron y evaluaron, y
ninguna supera de forma estable al modelo estático.

## 2:00 a 4:30. Datos (B)

Dos fuentes integradas por left join uno a uno, con el número de filas invariante y
ningún identificador huérfano. La identidad cubre el 24,4 % de las transacciones y
su ausencia se conserva en `has_identity`.

`TransactionDT` es un desfase en segundos. La hora derivada es relativa y se usa solo
como ciclo de 24 horas.

Mostrar la figura 2 del informe (`reports/figures/eda_temporal.png`).

La prevalencia semanal varía entre 2,1 % y 5,1 %, y las covariables se mantienen
estables. El domain classifier obtiene un AUC de 0,551, con umbral de alerta en 0,75.
La validación adversarial por columna coincide: la mediana de las 425 columnas es
0,505 y la columna original más alta, `C9`, llega a 0,600.

Si preguntan por el 0,78 a 0,95 del avance: aquel benchmark comparaba quintos del
periodo con todas las columnas, incluidas `D1`–`D15` sin normalizar, que miden días
transcurridos. El monitor actual usa un panel acotado y excluye las variables que
crecen con el calendario. Los conteos expansivos propios subían el AUC a 0,654 y por
eso quedaron fuera del predictor.

## 4:30 a 6:30. Protocolo temporal (B y C)

Dibujar la línea temporal.

```
0        60   69   76   83   90        120  135  150  165  182
|---------|----|----|----|---------|----|----|----|----|
 desarrollo   fit  cal  pol   H   warmup   B1   B2   B3   B4
```

En desarrollo hay tres reservas de 7 días: calibración, política y validación. Ahí se
eligen una sola vez la familia, la política, los umbrales y la ventana.

En cada actualización del test, cada versión usa tres roles: predictor, calibrador y
validación de promoción, con dos reservas de 7 días. La política ya está congelada y
no necesita reserva propia; liberarla le da a W30 16 días de fit en lugar de 9. Las
reservas son iguales para todas las estrategias, de modo que una diferencia de costo
se puede atribuir a la ventana.

El repositorio tiene 150 pruebas automáticas. La más directa invierte todas las
etiquetas aún inmaduras y comprueba que el predictor no cambia.

## 6:30 a 9:00. Modelos y decisión (C)

| Familia | Config | AP | Costo con regla económica | Costo con mejor umbral fijo |
|---|---|---|---|---|
| LightGBM | `lgbm_31` | 0,6214 | 1,5895 | 1,9696 |
| Random Forest | `rf_200_20` | 0,5391 | 1,8728 | 2,2746 |
| Logística | `lr_c1_bal` | 0,4029 | 2,1910 | 2,8712 |

La familia se elige por costo. La logística es la referencia lineal, Random Forest
agrega interacciones por bagging y LightGBM maneja de forma nativa cientos de
columnas dispersas y con faltantes.

La calibración se justifica con un número. La logística con
`class_weight=balanced` tiene un ECE de 0,285 y Platt lo baja a 0,0044. Como la
política multiplica `p` por el monto, necesita probabilidades bien calibradas.

La acción sale del mínimo de los tres costos esperados. El punto de indiferencia
entre aprobar y bloquear es `p* = c_FP / (monto + c_FP)` y depende del monto, así que
un corte fijo sobre `p` bloquea de más en montos bajos y de menos en montos altos. En
la reserva de desarrollo, la mejor regla de dos umbrales cuesta 1,9696 UM/tx y la
regla económica 1,5895.

`c_FP` se deriva del objetivo de bloquear a menos del 1 % de las legítimas: el menor
valor que lo cumple es 25 UM, con 0,89 %. `λ` es el precio sombra del cupo y resulta
7,86 UM.

## 9:00 a 12:30. Experimento de adaptación (D)

Mostrar la figura 4 del informe (`reports/figures/drift_adaptacion.png`). Siete
estrategias sobre los mismos 175 998 eventos, con la misma semilla y la misma
política.

| Estrategia | Días de fit | AP | F1 | Costo UM/tx | Δ frente a S0 |
|---|---|---|---|---|---|
| E15 (recomendada) | 76 a 121 | 0,4965 | 0,488 | 1,984 | −0,108 |
| W90 | 76 | 0,4901 | 0,481 | 2,057 | −0,035 |
| S0 | 69 | 0,4828 | 0,480 | 2,092 | referencia |
| W60 | 46 | 0,4802 | 0,480 | 2,109 | 0,017 |
| R46_antiguo | 46 | 0,4681 | 0,472 | 2,117 | 0,025 |
| W30 | 16 | 0,4701 | 0,471 | 2,183 | 0,091 |
| R46_medio | 46 | 0,4571 | 0,461 | 2,215 | 0,123 |

Tres puntos a explicar.

La ventana se eligió en desarrollo sin mirar el test: W60 con 2,2527 UM/tx, W90 con
2,2708 y W30 con 2,3501.

El diseño separa volumen y frescura. W30, W60 y W90 cambian los días de fit con el
mismo corte. W60, R46_medio y R46_antiguo mantienen 46 días y mueven el corte. Con el
volumen fijo, retrasar el corte 30 días encarece 0,1059 UM/tx, y es la única
comparación que no cruza el cero en ninguna de las 18 combinaciones de semilla y
bloque. Pasar de 76 a 16 días de fit cuesta 0,1257, con una combinación que cruza el
cero.

La lectura es que el drift está en P(y|X) y es pequeño. Las covariables son estables,
ADWIN registra deriva del error y el efecto de frescura es robusto, pero su tamaño es
menor que lo que se pierde al entrenar con menos datos. Por eso E15, que reentrena
sin descartar histórico, es la más barata.

Sobre los umbrales diagnósticos: S0 y R46_antiguo cumplen FPR ≤ 1 % y precisión ≥ 80 %
en test. Las estrategias que se reentrenan aplican el mismo umbral a versiones nuevas
y quedan con FPR entre 1,06 % y 1,18 %. El umbral no se reajustó después de ver el
test.

## 12:30 a 15:00. Demo y despliegue (E)

Ejecutar en vivo, con capturas de respaldo.

```bash
python -m fraud_adaptive replay --package models/W60_T165 --max-events 5000 --fixtures
```

Mostrar que aparecen las tres acciones sobre datos reales, que el cupo llega a 150 de
150 y que la prueba de idempotencia pasa. La demo usa el paquete W60 porque valida la
mecánica del servicio; cualquier paquete versionado, incluido el de E15, se sirve de
la misma forma.

El servicio calcula la decisión y el replay la confirma y reserva el cupo. Si el cupo
viviera en el servicio, con varias réplicas haría falta estado distribuido, que queda
como diseño futuro.

La imagen Docker está construida y verificada: HEALTHCHECK en verde, la misma acción
que el cálculo offline, p95 de 90,1 ms sobre HTTP con objetivo de 300 ms, y 503 sin
paquete montado. En mil peticiones devuelve 951 aprobar, 38 revisar y 11 bloquear.
Queda como paso manual publicar la imagen y crear el servicio en GCP.

## 15:00 a 17:00. Infografía, equidad y riesgos (A y E)

Mostrar `reports/infografia.pdf` y recorrer las seis etapas y los dos relojes.

Equidad operativa. En validación, la brecha máxima de bloqueo de legítimas entre
segmentos es 0,70 pp, bajo el umbral de 2 pp del gate. En test, con W60, 2 de 14
segmentos con soporte lo superan: `DeviceType` desktop con 2,14 pp y mobile con
2,20 pp. IEEE-CIS no tiene atributos protegidos, así que se trata de disparidad sobre
variables de negocio.

Las 22 evaluaciones de gates de la corrida recomendaron promover. Cada evaluación
queda registrada con su motivo.

Riesgos: las transacciones bloqueadas no revelan su desenlace en operación real
(selective labels), una versión nueva puede mover el nivel real de los umbrales, y el
sistema resuelve solo cerca del 95 % de los casos, por lo que la apelación y la
aprobación humana de versiones son obligatorias.

## 17:00 a 18:00. Conclusión (A)

El costo simulado baja de 5,40 a 1,98 UM por transacción frente a aprobar todo, una
mejora de 63,2 % que viene del modelo calibrado y de la política económica.

El concept drift de IEEE-CIS existe y cuesta 0,106 UM/tx por cada 30 días de
antigüedad del modelo. Con un retraso de etiqueta de 30 días, la mejor respuesta es
reentrenar cada 15 días con todo el histórico, con monitoreo continuo y promoción
aprobada por una persona. El paso siguiente es evaluar retrasos y cadencias menores.

---

# Preguntas probables

### Si las ventanas deslizantes no mejoran, ¿el sistema resuelve el drift?

Sí, con la estrategia que el experimento señala. El enunciado lista el
reentrenamiento periódico entre las estrategias de adaptación. E15 reentrena cada 15
días y es la más barata de las siete, 5,2 % por debajo del estático. Las ventanas
deslizantes se evaluaron como pide el enunciado y el resultado muestra por qué
descartar datos no conviene aquí: el drift medido es menor que el costo de perder
volumen. La ventaja de E15 sobre el estático tiene un intervalo que toca el cero,
y el informe lo declara.

### ¿Cómo saben que no hay fuga temporal?

Por 150 pruebas automáticas. Alterar filas futuras no modifica medianas ni vocabulario
del preprocesamiento, invertir las etiquetas inmaduras no cambia el predictor,
permutar eventos con el mismo timestamp deja las features idénticas, y los roles de
cada versión no comparten ningún `TransactionID`.

### ¿Por qué W30 rinde peor si es la más fresca?

Porque entrena con 16 días y 2 191 fraudes, frente a 76 días y 9 169 fraudes de W90.
El bloque de antigüedad variable muestra que la frescura sí importa cuando el volumen
se mantiene, pero su efecto es menor que el del volumen.

### ¿No sobreajustaron al elegir la ventana?

La ventana se eligió en la reserva de validación de desarrollo y quedó sellada con el
hash `7d4c9fac0af4d15f` antes de abrir el test. El sello incluye `c_FP`, que se deriva
de datos de desarrollo. Si la configuración cambia, la corrida se detiene.

### ¿Por qué la política no usa umbrales sobre `p`?

Porque el punto de indiferencia depende del monto. La diferencia medida es de 1,9696
a 1,5895 UM/tx. Los umbrales se calculan igual, para cuantificar esa diferencia y para
las métricas diagnósticas como F1 y recall a FPR 1 %.

### ADWIN detecta 49 cambios. ¿Por qué no se reentrena cuando detecta?

ADWIN observa el error con etiquetas maduras, así que cada detección llega 30 días
después del evento. Además, las detecciones caen en los mismos días para las versiones
recién reentrenadas y para el estático, lo que apunta a periodos difíciles comunes.
Reentrenar en esos momentos no los habría evitado. La cadencia fija de 15 días se
eligió midiendo: cada 30 o 45 días capta solo el 29 % y el 37 % del beneficio.

### Topal et al. encuentran caídas de 90 % sobre el mismo dataset. ¿Se contradicen?

Su caída se concentra en los boosters, con recall de 0,03 y 0,04 a FPR 5 %, por debajo
de lo que daría un clasificador aleatorio; Random Forest cae 24 % y la logística 3 %.
Aquí LightGBM congelado 136 días obtiene 11 % menos recall que el reentrenado, del
orden de sus familias estables. Ambos trabajos coinciden en que reentrenar conviene.

### Los costos son supuestos. ¿Qué validez tienen las conclusiones?

Los parámetros están declarados y `c_FP` se deriva de un objetivo operativo. Con las
acciones de W60 congeladas, el costo va de 1,92 a 2,49 UM/tx entre los escenarios
bajo y alto, y sube a 2,34 si el analista acierta menos. Esa sensibilidad se calculó
para W60; el orden entre estrategias bajo otros costos no se evaluó. El informe no
afirma un ahorro sobre pagos reales.

### ¿Por qué no usan el veredicto del analista para entrenar?

Porque es una opinión con recall de 0,90 que solo se conoce cuando la etiqueta ya
maduró. Se calcula en el evaluador y no modifica decisiones ya emitidas.

### ¿Qué pasa con las transacciones bloqueadas?

Es la limitación más seria. El backtest usa información completa porque el dataset la
contiene. En producción las etiquetas quedarían sesgadas por las decisiones previas
del sistema; la corrección por etiquetas enmascaradas queda como trabajo futuro.

### ¿Por qué Platt y no calibración isotónica?

La cola de calibración es de 7 días. Con dos parámetros, Platt es estable entre
versiones; la isotónica necesita más datos para no sobreajustar escalones.

### ¿Qué harían con más tiempo?

Evaluar retrasos de etiqueta menores que 30 días y cadencias menores que 15, que es
donde las ventanas deslizantes podrían aportar, repetir el experimento con más
semillas de entrenamiento y corregir el sesgo de selective labels.
