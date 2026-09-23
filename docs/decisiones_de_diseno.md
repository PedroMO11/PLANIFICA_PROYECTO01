# Decisiones de diseño

Cada sección describe una decisión del sistema y la medición que la sustenta. Las
cifras provienen de la corrida `v3` sobre IEEE-CIS completo: 590 540 transacciones,
394 columnas y 182 días.

## 1. Dos claves de entidad

El dataset no trae identificador de cliente. El panel construye dos proxies de
tarjeta y usa ambos.

| Proxy | Definición | Entidades | Eventos por entidad | Singletons |
|---|---|---|---|---|
| `card_proxy` | `card1` a `card6` más `addr1` | 43 018 | 13,73 | 40,4 % |
| `cliente_proxy` | `card1 + addr1 + D1n`, con `D1n` el día de alta | 217 850 | 2,71 | 57,5 % |

La clave fina sigue el identificador que usó la solución ganadora de la
competencia. `D1` mide los días desde que la tarjeta empezó a usarse, así que
restarlo del día absoluto aproxima la fecha de alta y separa tarjetas del mismo
emisor.

Para decidir se entrenó LightGBM solo con features de historial y se evaluó desde el
día 122:

| Historial | AP |
|---|---|
| Solo `card_proxy` | 0,1668 |
| Solo `cliente_proxy` | 0,1671 |
| Ambos | 0,1829 |

Por separado rinden lo mismo y juntas un 9,6 % más: la clave gruesa aporta soporte
estadístico a las ventanas de 24 horas y 7 días, y la fina aporta resolución por
tarjeta.

`D1n` se deriva de `TransactionDT`. La columna `dia` se mide desde el primer evento
del frame procesado, de modo que usarla daría claves distintas a una misma tarjeta
según el rango que se procese.

## 2. Los conteos expansivos quedan fuera del predictor

Un conteo que acumula desde el primer evento crece con el calendario y actúa como
sustituto del día absoluto, que el protocolo excluye.

| Feature | Media en `[0,60)` | Media en `[150,182)` | Correlación con el día |
|---|---|---|---|
| `card_cnt_expansivo` | 138,78 | 707,43 | +0,2276 |
| `card_cnt_7d` | estable | estable | −0,009 |

La validación adversarial por feature, que en la competencia descartaba las columnas
con AUC superior a 0,70, marca solo dos de las 425 columnas del panel: ambos conteos
expansivos, con 0,7066 y 0,7006. La mediana del panel es 0,5053 y la columna
original más alta es `C9`, con 0,6002.

Las columnas se siguen calculando como diagnóstico. `FORBIDDEN_PREDICTOR_SUFFIXES`
las excluye del panel y `assert_no_forbidden_features` falla si alguna entra al
predictor.

Con los conteos expansivos dentro, el domain classifier obtenía un AUC de 0,654; sin
ellos obtiene 0,551. Esa diferencia era drift de covariables producido por features
propias.

## 3. El experimento separa volumen y frescura

Variar solo el ancho `W` cambia a la vez cuántos días entrena el modelo y cuán
reciente es su información. La configuración agrega un bloque que mantiene el
volumen y mueve el corte.

| Bloque | Estrategias | Qué varía | Qué se mantiene |
|---|---|---|---|
| Volumen variable | W30, W60, W90 | 16, 46 y 76 días de fit | Corte en `c−14` |
| Antigüedad variable | W60, R46_medio, R46_antiguo | Punto de corte | 46 días de fit |

`R46_medio` es de tipo `lagged`: 46 días que terminan 30 días antes del corte `c`, es
decir 16 días antes que los de W60, cuyo fit ya termina en `c−14`. `R46_antiguo` toma
los primeros 46 días del histórico y no se reentrena. Al compartir el tamaño del
conjunto de entrenamiento, la diferencia entre W60 y R46_medio solo puede venir de la
antigüedad; la de R46_antiguo combina antigüedad y ausencia de reentrenamiento. `build_version_roles` rechaza una combinación de `lag_days` y `fit_days`
que no quepa antes del corte.

| Comparación | Qué aísla | Efecto UM/tx | Cruzan cero | Veredicto |
|---|---|---|---|---|
| R46_medio frente a W60 | Frescura, corte 16 días más antiguo | +0,1059 | 0 de 18 | Concluyente |
| R46_antiguo frente a W60 | Frescura, corte al inicio del histórico | +0,0084 | 18 de 18 | No concluyente |
| W30 frente a W90 | Volumen y antigüedad, 16 frente a 76 días | +0,1257 | 1 de 18 | No concluyente |
| W30 frente a W60 | Volumen y antigüedad, 16 frente a 46 días | +0,0742 | 4 de 18 | No concluyente |
| W60 frente a W90 | Volumen y antigüedad, 46 frente a 76 días | +0,0514 | 18 de 18 | No concluyente |

La única comparación que resiste el control de robustez es la de frescura: con 46
días de fit, retrasar el corte 16 días encarece la decisión. Variando solo `W`, la
correlación entre días de fit y costo era de −0,9916 y sugería que únicamente
importaba el volumen; con el diseño controlado baja a −0,7412.

El efecto de la antigüedad no es monótono. El modelo entrenado con los 46 primeros
días del histórico no se distingue de la ventana más fresca. Caracterizar esa curva
requeriría cortes intermedios.

Topal, Bozanta, Erer y Başar comparan sobre este dataset modelos congelados tras los
primeros 16 días con modelos reentrenados a diario.[^1] La caída de recall a FPR 5 %
se concentra en los boosters (0,03 y 0,04 frente a 0,37 y 0,35); Random Forest cae
24 % y la logística 3 %. Aquí, LightGBM congelado 136 días obtiene 11 % menos recall
a FPR 1 % que el reentrenado. En ambos trabajos reentrenar conviene; aquí, además,
reentrenar sin descartar histórico (E15) es más barato que cualquier ventana
deslizante.

[^1]: «Handling Concept Drift in Fraud Detection: A Replication Study», 38th
Canadian Conference on Artificial Intelligence, 2025.

## 4. Dos reservas por versión

La política y sus umbrales se eligen una vez en desarrollo. Reservar `[c−14, c−7)`
para la política en cada actualización restaba días al predictor sin cumplir
función, así que el predictor termina en `c−14` y usa `W − 14` días.

| Ventana | Días de fit con tres reservas | Días de fit con dos |
|---|---|---|
| W30 | 9 | 16 |
| W60 | 39 | 46 |
| W90 | 69 | 76 |

La reserva adicional afectaba sobre todo a las ventanas cortas: W30 entrenaba con
44 % menos datos.

## 5. La acción se elige por costo esperado

El punto de indiferencia entre aprobar y bloquear es

```
p* = c_FP / (monto + c_FP)
```

y depende del monto, así que dos umbrales globales sobre `p` no pueden ser óptimos
bajo este modelo de costos. La regla evalúa los tres costos esperados para cada
caso. Sobre la reserva de política de desarrollo, con la misma economía y el mismo
cupo:

| Familia | Costo con la regla económica | Costo con el mejor umbral fijo |
|---|---|---|
| LightGBM `lgbm_31` | 1,5895 | 1,9696 |
| Random Forest `rf_200_20` | 1,8728 | 2,2746 |
| Logística `lr_c1_bal` | 2,1910 | 2,8712 |

Los umbrales se eligieron minimizando el costo en esa misma reserva, y aun así
cuestan entre 19 % y 24 % más. Se siguen calculando para medir esa diferencia y para
las métricas diagnósticas. `implied_thresholds` describe la regla en términos de `p`:

| Monto | Revisar desde | Bloquear desde |
|---|---|---|
| 25 UM | `p` ≥ 0,4072 | `p` ≥ 0,5791 |
| 59 UM (mediana) | `p` ≥ 0,1747 | `p` ≥ 0,5143 |
| 250 UM | `p` ≥ 0,0415 | `p` ≥ 0,3159 |

Con monto cero, `E[aprobar] = 0` es siempre el mínimo y el caso no consume cupo.

## 6. La revisión paga el precio de la plaza que ocupa

Elegir la acción de menor costo esperado sin considerar el cupo pedía 704 revisiones
diarias para 150 plazas, que se llenaban por orden de llegada. La revisión se propone
solo si su ahorro supera el precio de la plaza:

```
min(E[aprobar], E[bloquear]) − E[revisar] > λ
```

`λ` es el multiplicador de Lagrange de la restricción de capacidad. Se calibra por
bisección, sin etiquetas, como el menor valor que ajusta la demanda diaria al cupo,
y la deja en 150,0. Ese valor queda a 0,7 % del que minimiza el costo observado. El
valor calibrado es 7,86 UM.

Sin precio sombra, la regla económica costaba 2,3090 UM/tx y el mejor umbral fijo
2,2339, porque el umbral elegido por rejilla racionaba el cupo de forma implícita.
Con el precio sombra la regla económica es la más barata.

## 7. `c_FP` se deriva de un objetivo operativo

Con `c_FP = 5 UM` el sistema bloqueaba al 11,17 % de las transacciones legítimas. El
costo de un falso positivo no es observable, pero la tasa de bloqueo de legítimas sí,
así que se fija el objetivo y se busca el menor `c_FP` que lo cumple. El objetivo es
1 %, el mismo de `diagnostic_targets.fpr_target`; el valor calibrado es 25 UM, con
0,89 %.

La calibración usa la reserva de política de desarrollo y la familia de mayor AP.
Elegir la familia por costo antes de calibrar `c_FP` sería circular, porque el costo
depende de él.

`c_FP` fija la escala de los costos y con ella el ahorro de revisar, y `λ` cambia
cuántos casos terminan bloqueados. Por eso se calibran alternando hasta que ambos se
estabilizan, en tres pasadas, y quedan sellados en el prerregistro.

## 8. La economía viaja con el paquete

Como `c_FP` y `λ` se derivan de los datos de desarrollo, el modelo de costos se
serializa en el manifiesto del paquete y el servicio lo lee de ahí.
`policy_from_dict` y `cost_model_from_dict` son los únicos constructores, de modo que
el servicio, el replay y el backtest reconstruyen la misma política.
`cost_model_from_dict` rechaza un diccionario sin `c_fp`.

## 9. Criterio para declarar una diferencia concluyente

`robustness_check` repite el bootstrap por bloques con 6 semillas y 3 tamaños de
bloque y cuenta en cuántas de las 18 combinaciones el intervalo cruza el cero. Una
diferencia es concluyente si no lo cruza en ninguna. El criterio se aplica también a
las comparaciones de volumen y frescura.

Con este control, la diferencia entre W60 y S0 y el efecto del volumen entre W30 y
W90 quedan como no concluyentes, aunque la corrida base sugería lo contrario.

## 10. Verificaciones

El Average Precision reportado coincide con `sklearn.metrics.average_precision_score`
con una diferencia de 5,55e−17, y el costo por transacción recalculado a mano
coincide con el reportado.

El sorteo del analista simulado, sobre 200 000 identificadores, tiene media 0,50006 y
desviación 0,28866; la prueba de Kolmogorov-Smirnov contra la uniforme da
D = 0,00031 con p = 1,0, y la autocorrelación de orden 1 es 0,039.

Las pruebas de causalidad alteran filas futuras, invierten etiquetas inmaduras y
permutan eventos con el mismo timestamp, y comprueban que la salida no cambia.

`test_backtest_end_to_end.py` recorre `run_backtest` con las siete estrategias y
verifica que la economía calibrada llega al manifiesto de cada paquete.

`test_paridad_entre_el_servicio_y_el_calculo_offline` compara la acción emitida y los
tres costos esperados del servicio con el cálculo offline, y otra prueba verifica
que montos de 1 UM y de 5 000 UM con la misma probabilidad reciben acciones
distintas.

Dos corridas independientes producen los mismos 18 fits de tuning, el mismo hash de
prerregistro y el mismo costo por estrategia. `fraud-adaptive verify --against` lo
comprueba y `reports/verificacion_reproducibilidad.json` guarda la comparación.

## 11. Preguntas abiertas

La capacidad sigue siendo la restricción dominante: con 150 plazas diarias el sistema
resuelve de forma automática cerca del 95 % de las transacciones, y mejorar el AP
tiene rendimientos decrecientes.

La forma de la curva de antigüedad queda sin caracterizar, y el protocolo no alcanza
retrasos de etiqueta menores que 30 días ni cadencias menores que 15, que es donde
las ventanas deslizantes podrían aportar.
