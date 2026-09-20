# Decisiones de diseño

El plan de implementación fue redactado por un modelo de lenguaje. Siete de sus
decisiones se midieron y resultaron subóptimas. Este documento fija la decisión
vigente en cada caso y la medición que la sustenta. Lo que aquí se describe es lo
que el código hace, no una propuesta.

Las mediciones provienen de IEEE-CIS completo, 590 540 transacciones, 394 columnas
y 182 días.

---

## 1. Dos claves de entidad, no una

El dataset está anonimizado y no trae identificador de cliente. El panel construye
dos proxies de tarjeta y los usa a la vez.

| Proxy | Definición | Entidades | Eventos por entidad | Singletons |
|---|---|---|---|---|
| `card_proxy` | `card1` a `card6` más `addr1` | 43 018 | 13,73 | 40,4 % |
| `cliente_proxy` | `card1 + addr1 + D1n`, con `D1n` el día de alta | 217 850 | 2,71 | 57,5 % |

La clave fina reproduce el UID que usó la solución ganadora de la competencia.
`D1` mide los días transcurridos desde que la tarjeta empezó a usarse, de modo que
restarlo del día absoluto identifica la fecha de alta y separa tarjetas distintas
del mismo emisor.

La decisión de usar ambas se tomó midiendo. Se entrenó LightGBM únicamente con
features de historial sobre un test temporal, día 122 en adelante.

| Historial | AP |
|---|---|
| Solo `card_proxy` | 0,1668 |
| Solo `cliente_proxy` | 0,1671 |
| Ambos | 0,1829 |

Por separado valen lo mismo. Juntas rinden un 9,6 % relativo más, porque la gruesa
aporta soporte estadístico a las ventanas de 24 h y 7 d mientras la fina aporta
resolución por tarjeta. No son redundantes y el panel emite los dos historiales.

`D1n` se deriva de `TransactionDT` y no de la columna `dia`. Esta última se mide
desde el primer evento del frame, de modo que una misma tarjeta recibiría claves
distintas según el rango que se procese. El día absoluto es estable entre llamadas.

---

## 2. Los conteos expansivos quedan fuera del predictor

Un conteo que acumula desde el primer evento crece de forma monótona con el
calendario y funciona como sustituto del día absoluto, que el protocolo prohíbe.

| Feature | Media en `[0,60)` | Media en `[150,182)` | Correlación con el día |
|---|---|---|---|
| `card_cnt_expansivo` | 138,78 | 707,43 | +0,2276 |
| `card_cnt_7d` | estable | estable | −0,009 |

La competencia usaba adversarial validation por feature y descartaba las que
superaban AUC 0,70. Aplicado al panel completo de 425 columnas, solo dos superan
ese umbral y ambas son conteos expansivos propios, con 0,7066 y 0,7006. La mediana
del panel es 0,5053 y la feature original más alta es `C9` con 0,6002.

Las columnas se siguen calculando porque sirven de diagnóstico. `FORBIDDEN_PREDICTOR_SUFFIXES`
las excluye del panel y `assert_no_forbidden_features` falla si alguna se cuela.

El resultado tiene una consecuencia de fondo. Las covariables de IEEE-CIS son
estables en el tiempo, lo que explica por qué las ventanas deslizantes no compran
nada. El drift que el dataset presenta no es de covariables.

---

## 3. El experimento separa volumen de frescura

El plan definía las estrategias variando solo el ancho de ventana `W`, de modo que
al cambiar `W` cambiaban a la vez cuántos días entrena el modelo y cuán reciente es
su información. Con ese diseño una diferencia de costo no es atribuible a ninguno
de los dos factores.

La configuración vigente añade un bloque de antigüedad variable con volumen
constante.

| Bloque | Estrategias | Qué varía | Qué se mantiene |
|---|---|---|---|
| Volumen variable | W30, W60, W90 | días de fit | punto de corte |
| Antigüedad variable | W60, R46_medio, R46_antiguo | punto de corte | 46 días de fit |

`R46_medio` usa el tipo `lagged`, que toma 46 días terminando 30 días antes del
corte. `R46_antiguo` toma los primeros 46 días del histórico. Las tres comparten
tamaño de training set, de modo que la diferencia entre ellas solo puede venir de
la antigüedad.

`build_version_roles` rechaza de forma explícita una combinación de `lag_days` y
`fit_days` que no quepa en el corte, en lugar de truncar la ventana en silencio.

---

## 4. Dos reservas por versión, no tres

El plan reservaba `[c-14, c-7)` para elegir umbrales en cada actualización. Esa
elección ocurre una sola vez, en el primer corte, de modo que en los cortes
posteriores la reserva quedaba excluida del fit sin cumplir función alguna.

La reserva de política desaparece. El predictor termina en `c-14` y usa `W-14` días
efectivos.

| Ventana | Días de fit antes | Días de fit ahora |
|---|---|---|
| W30 | 9 | 16 |
| W60 | 39 | 46 |
| W90 | 69 | 76 |

La penalización recaía de forma desproporcionada sobre las ventanas cortas, que son
las que el experimento pretende evaluar. W30 entrenaba con un 44 % menos de datos
de los disponibles.

---

## 5. La acción se elige por argmin económico

El plan decidía con dos umbrales globales sobre `p`. Esa regla no puede ser óptima
bajo el modelo de costos declarado, porque el punto de indiferencia entre aprobar y
bloquear es

```
p* = c_FP / (monto + c_FP)
```

y por lo tanto depende del monto. Un corte fijo bloquea de más en los montos bajos
y de menos en los altos.

La regla vigente evalúa los tres costos esperados caso por caso y toma el mínimo,
sujeto al cupo diario. Medido sobre los cuatro bloques de test con la misma
economía y el mismo cupo:

| Regla | Costo por transacción | Revisiones | Bloqueo de legítimas | Monto de fraude aprobado |
|---|---|---|---|---|
| Dos umbrales fijos | 1,6262 | 9 300 | 12,24 % | 172 929 |
| Argmin económico | 1,5054 | 9 300 | 11,95 % | 154 091 |

El argmin domina en todos los ejes. La propiedad es estructural y no depende del
dataset. `test_argmin_nunca_es_peor_que_el_mejor_umbral_fijo` la fija como
invariante.

Los umbrales se siguen eligiendo en desarrollo, pero ya no deciden. Cumplen dos
funciones: cuantificar en el informe lo que cuesta ignorar el monto, y dar un corte
binario a las métricas de diagnóstico.

La regla induce un umbral por monto. `implied_thresholds` los calcula para poder
describir la política en términos de `p` sin cambiar la decisión.

Un efecto colateral. Con monto cero `E[aprobar] = 0` es siempre el mínimo, de modo
que el caso ya no consume cupo de revisión. Con la regla anterior caía en la zona
gris y sí lo consumía.

---

## 6. `c_FP` se deriva de un objetivo operativo

El plan fijaba `c_FP = 5 UM` sin justificarlo contra la distribución de montos.
Con el monto mediano de 68 UM basta una probabilidad del 6,8 % para que bloquear
resulte más barato que aprobar, y el sistema termina bloqueando al 11,95 % de las
transacciones legítimas. Ninguna operación de pagos acepta rechazar a uno de cada
nueve clientes legítimos.

El parámetro no es observable, pero la cantidad que restringe sí lo es. La relación
se invierte: se declara el objetivo de bloqueo de legítimas y se busca el menor
`c_FP` que lo cumple. El valor resultante es el precio sombra de esa restricción.

El objetivo es el 1 %, que ya estaba declarado en `diagnostic_targets.fpr_target` y
que la política anterior ignoraba.

La calibración corre una sola vez sobre la reserva de política de desarrollo, usa
como referencia la familia de mayor AP y no la de menor costo, porque el costo
depende de `c_FP` y elegir por costo antes de calibrarlo sería circular. El valor
queda sellado en el prerregistro junto con la familia y los hiperparámetros.

El orden de las estrategias no depende de esta elección. Reevaluado con la regla
argmin sobre toda la escala:

| `c_FP` | Orden de menor a mayor costo |
|---|---|
| 5 | E15 < W90 < S0 < W60 < W30 |
| 25 | E15 < W90 < W60 < S0 < W30 |
| 50 | E15 < W90 < W60 < S0 < W30 |
| 75 | E15 < W90 < W60 < S0 < W30 |
| 100 | E15 < S0 < W90 < W60 < W30 |

La estrategia con más datos gana siempre y la de menos datos pierde siempre. La
conclusión central es invariante al parámetro económico.

Los escenarios de sensibilidad se declaran como múltiplos del valor calibrado, no
como valores absolutos. Las acciones se calculan una sola vez con la economía
congelada y no se recalculan por escenario, porque con la regla argmin la economía
es la política y recalcularlas respondería a qué habría hecho un sistema distinto.

---

## 7. La economía viaja con el paquete

`c_FP` pasó de ser una constante del config a un parámetro derivado de los datos de
desarrollo. En consecuencia el modelo de costos se serializa en el manifiesto del
paquete desplegable y el servicio lo lee de ahí.

`policy_from_dict` y `cost_model_from_dict` son los únicos constructores. Antes
cada consumidor reconstruía la política campo por campo, de modo que uno podía
olvidar un campo y decidir distinto del backtest sin que nada fallara.
`cost_model_from_dict` rechaza de forma explícita un diccionario sin `c_fp`.

---

## 8. Una diferencia se declara concluyente solo si resiste semilla y bloque

Un intervalo de confianza obtenido con una semilla y un tamaño de bloque no basta.
`robustness_check` repite el bootstrap por bloques con 6 semillas y 3 tamaños y
cuenta en cuántas de las 18 combinaciones el intervalo cruza el cero.

El control es barato y ya evitó una afirmación incorrecta. La versión anterior del
informe declaraba W60 significativamente más cara que S0 con un intervalo que
excluía el cero. Repetido, el intervalo cruzaba el cero en 12 de 18 combinaciones.

---

## 9. Verificaciones que sostienen lo anterior

**Métricas.** El Average Precision reportado coincide con
`sklearn.metrics.average_precision_score` con una diferencia de 5,55e−17. El costo
por transacción recalculado a mano coincide con el reportado.

**Sorteo del analista simulado.** Sobre 200 000 identificadores la media es 0,50006
y la desviación 0,28866. Kolmogorov-Smirnov contra la uniforme da D = 0,00031 con
p = 1,0. La autocorrelación de orden 1 es 0,039.

**Causalidad.** Las pruebas alteran las filas futuras, invierten las etiquetas
inmaduras y permutan los identificadores de eventos con el mismo timestamp,
comprobando en cada caso que la salida no cambia bit a bit.

**Reproducibilidad.** Dos corridas independientes de la cadena completa produjeron
resultados idénticos en los 18 fits de tuning, en el hash de prerregistro y en el
costo de las cinco estrategias. `fraud-adaptive verify` lo comprueba a demanda.

**Gates de promoción.** El gate social discriminó en lugar de bloquear de forma
sistemática. En `T = 135` bloqueó a las cinco estrategias con brechas de 2,76 a
4,69 puntos porcentuales. En `T = 150` y `T = 165` las brechas caen a un rango de
0,18 a 1,22 y las promociones proceden.

---

## 10. Lo que el experimento no responde

La capacidad de revisión es de 150 casos diarios frente a una demanda superior a
1 000, de modo que la cola gobierna una fracción pequeña de las decisiones. El
sistema se comporta en la práctica como un clasificador binario con un presupuesto
marginal de revisión. Esto acota qué mide el experimento y no invalida el resultado.

El plan declara E15 como referencia no desplegable. Es la estrategia de menor costo
y es perfectamente desplegable, porque consiste en reentrenar sobre todo el
histórico cada 15 días. La exclusión responde a que el enunciado prioriza ventanas
deslizantes, no a un argumento técnico. El informe la reporta como desplegable.
