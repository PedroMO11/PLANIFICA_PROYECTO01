# Decisiones de diseño

El plan de implementación fue redactado por un modelo de lenguaje. Ocho de sus
decisiones se midieron y resultaron subóptimas. Este documento fija la decisión
vigente en cada caso y la medición que la sustenta. Lo que aquí se describe es lo
que el código hace, no una propuesta.

Las mediciones provienen de IEEE-CIS completo, 590 540 transacciones, 394 columnas
y 182 días. La corrida es `v3`.

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

Las columnas se siguen calculando porque sirven de diagnóstico.
`FORBIDDEN_PREDICTOR_SUFFIXES` las excluye del panel y
`assert_no_forbidden_features` falla si alguna se cuela.

El efecto sobre la señal de drift es directo. Con los conteos expansivos dentro, el
domain classifier obtenía un AUC de 0,654. Sin ellos obtiene 0,551, mucho más cerca
del azar. Buena parte del drift de covariables que el sistema creía observar lo
producían features propias, no el dataset.

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
| Volumen variable | W30, W60, W90 | 16, 46 y 76 días de fit | corte en `c-14` |
| Antigüedad variable | W60, R46_medio, R46_antiguo | punto de corte | 46 días de fit |

`R46_medio` usa el tipo `lagged`, que toma 46 días terminando 30 días antes del
corte. `R46_antiguo` toma los primeros 46 días del histórico. Las tres comparten
tamaño de training set, de modo que la diferencia entre ellas solo puede venir de
la antigüedad.

`build_version_roles` rechaza de forma explícita una combinación de `lag_days` y
`fit_days` que no quepa en el corte, en lugar de truncar la ventana en silencio.

**La separación cambia la conclusión, y en el sentido contrario al esperado.**

| Comparación | Qué aísla | Efecto | Cruzan cero | Veredicto |
|---|---|---|---|---|
| R46_medio frente a W60 | Frescura, corte 30 días más antiguo | +0,1059 | 0 de 18 | Concluyente |
| R46_antiguo frente a W60 | Frescura, corte al inicio del histórico | +0,0084 | 18 de 18 | No concluyente |
| W30 frente a W90 | Volumen, 16 frente a 76 días | +0,1257 | 1 de 18 | No concluyente |
| W30 frente a W60 | Volumen, 16 frente a 46 días | +0,0742 | 4 de 18 | No concluyente |
| W60 frente a W90 | Volumen, 46 frente a 76 días | +0,0514 | 18 de 18 | No concluyente |

La única comparación que resiste el control de robustez es la de frescura.
Mantener el volumen en 46 días y retroceder el corte 30 días encarece la decisión.
El diseño confundido llevaba a la conclusión opuesta, que el volumen dominaba y la
frescura no producía efecto medible.

**Contraste con la literatura.** Topal, Bozanta, Erer y Başar comparan sobre este
mismo dataset modelos congelados tras los primeros 16 días contra modelos
reentrenados a diario sobre los últimos 23, con un recall a FPR del 5 % de 0,03
frente a 0,37, y concluyen que existe concept drift.[^1] Su comparación mantiene el
volumen casi constante y mueve solo la antigüedad, que es lo que aquí hace el bloque
de antigüedad variable, y da el mismo signo. La magnitud difiere porque su modelo
congelado acumula cinco meses de antigüedad y el de aquí retrocede 30 días.

El diseño de aquí permite además distinguir entre reentrenar y descartar
histórico, distinción que el suyo no necesita hacer. Su método de reentrenamiento usa todos los datos
disponibles, y aquí la estrategia expansiva, que reentrena sin descartar, es la más
barata de las siete. La evidencia sostiene reentrenar con frecuencia y no sostiene
descartar histórico.

[^1]: *Handling Concept Drift in Fraud Detection: A Replication Study*, 38th
Canadian Conference on Artificial Intelligence, Calgary, 2025.

El efecto de la frescura no es monótono. Retroceder 30 días encarece, pero tomar
los 46 primeros días del histórico no se distingue de la ventana más fresca. La
correlación entre días de fit y costo cae de −0,9916 en el diseño confundido a
−0,7412 en el controlado.

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

## 5. La acción sale del costo esperado, no de umbrales

El plan decidía con dos umbrales globales sobre `p`. Esa regla no puede ser óptima
bajo el modelo de costos declarado, porque el punto de indiferencia entre aprobar y
bloquear es

```
p* = c_FP / (monto + c_FP)
```

y por lo tanto depende del monto. Un corte fijo bloquea de más en los montos bajos
y de menos en los altos.

La regla vigente evalúa los tres costos esperados caso por caso. Medido sobre la
reserva de desarrollo, con la misma economía y el mismo cupo:

| Familia | Costo con la regla económica | Costo con el mejor umbral fijo |
|---|---|---|
| LightGBM `lgbm_31` | **1,5895** | 1,9696 |
| Random Forest `rf_200_20` | 1,8728 | 2,2746 |
| Logística `lr_c1_bal` | 2,1910 | 2,8712 |

La comparación es conservadora en contra de la regla económica, porque los umbrales
se eligen minimizando el costo sobre esa misma ventana mientras la regla económica
no se ajusta a ella. Aun así pierde por entre un 19 % y un 24 %.

Los umbrales se siguen eligiendo en desarrollo, pero ya no deciden. Cumplen dos
funciones: cuantificar lo que cuesta ignorar el monto, y dar un corte binario a las
métricas de diagnóstico.

La regla induce un umbral por monto. `implied_thresholds` los calcula para poder
describir la política en términos de `p` sin cambiar la decisión.

| Monto | Revisar desde | Bloquear desde |
|---|---|---|
| 25 UM | `p` ≥ 0,4072 | `p` ≥ 0,5791 |
| 59 UM (mediana) | `p` ≥ 0,1747 | `p` ≥ 0,5143 |
| 250 UM | `p` ≥ 0,0415 | `p` ≥ 0,3159 |

Un efecto colateral. Con monto cero `E[aprobar] = 0` es siempre el mínimo, de modo
que el caso ya no consume cupo de revisión. Con la regla anterior caía en la zona
gris y sí lo consumía.

---

## 6. La revisión paga el precio de la plaza que ocupa

Elegir la acción de menor costo esperado resuelve el problema sin restricción, no
el que el sistema tiene. Con un cupo duro de 150 revisiones diarias, esa regla
propone 704 revisiones por día. El cupo se llena con los casos que llegan primero y
los que más ahorrarían quedan fuera.

La revisión solo se propone si supera el precio de la plaza que ocupa:

```
min(E[aprobar], E[bloquear]) - E[revisar] > lambda
```

`lambda` es el multiplicador de Lagrange de la restricción de capacidad. Se calibra
por bisección como el menor valor que deja la demanda diaria dentro del cupo, lo
que la sitúa en 150,0 exactas frente a las 704 sin precio. El criterio es la
capacidad y no el costo observado, de modo que la calibración no usa etiquetas. El
`lambda` que iguala demanda y cupo queda a un 0,7 % del que minimiza el costo, de
manera que el criterio sin etiquetas no pierde casi nada.

El valor calibrado es 7,86 UM.

El efecto sobre la comparación con los umbrales fijos es decisivo. Sin precio
sombra la regla económica **perdía** contra el mejor umbral fijo, con 2,3090 frente
a 2,2339 UM/tx. La razón es que un umbral ajustado por rejilla raciona el cupo de
forma implícita al elegir una zona estrecha, de modo que batía a una regla que no
lo racionaba en absoluto. Con precio sombra gana con holgura.

---

## 7. `c_FP` se deriva de un objetivo operativo

El plan fijaba `c_FP = 5 UM` sin justificarlo contra la distribución de montos. Con
ese valor el sistema bloquea al 11,17 % de las transacciones legítimas. Ninguna
operación de pagos acepta rechazar a uno de cada nueve clientes legítimos.

El parámetro no es observable, pero la cantidad que restringe sí lo es. La relación
se invierte: se declara el objetivo de bloqueo de legítimas y se busca el menor
`c_FP` que lo cumple. El valor resultante es el precio sombra de esa restricción.

El objetivo es el 1 %, que ya estaba declarado en `diagnostic_targets.fpr_target` y
que la política anterior ignoraba. El valor calibrado es 25 UM, que alcanza un
0,89 %.

La calibración corre sobre la reserva de política de desarrollo y usa como
referencia la familia de mayor AP y no la de menor costo, porque el costo depende de
`c_FP` y elegir por costo antes de calibrarlo sería circular.

`c_FP` y `lambda` se condicionan mutuamente: `c_FP` fija la escala de los costos,
que determina cuánto ahorra revisar, y racionar el cupo cambia cuántos casos
terminan bloqueados. Se calibran alternando hasta que ambos se estabilizan, lo que
ocurre en tres pasadas. Ambos quedan sellados en el prerregistro.

---

## 8. La economía viaja con el paquete

`c_FP` y `lambda` pasaron de ser constantes del config a parámetros derivados de los
datos de desarrollo. En consecuencia el modelo de costos se serializa en el
manifiesto del paquete desplegable y el servicio lo lee de ahí.

`policy_from_dict` y `cost_model_from_dict` son los únicos constructores. Antes cada
consumidor reconstruía la política campo por campo, de modo que uno podía olvidar un
campo y decidir distinto del backtest sin que nada fallara. `cost_model_from_dict`
rechaza de forma explícita un diccionario sin `c_fp`.

---

## 9. Una diferencia se declara concluyente solo si resiste semilla y bloque

Un intervalo de confianza obtenido con una semilla y un tamaño de bloque no basta.
`robustness_check` repite el bootstrap por bloques con 6 semillas y 3 tamaños y
cuenta en cuántas de las 18 combinaciones el intervalo cruza el cero.

El criterio se aplica también a las comparaciones controladas de volumen y de
frescura, no solo a las que van contra la referencia. Declarar concluyente una
comparación controlada a partir de un solo intervalo sería aplicar un criterio más
laxo justo donde se apoya la conclusión.

El control ya evitó dos afirmaciones incorrectas. La primera, que W60 fuese
significativamente más cara que S0. La segunda, que el volumen produjese un efecto
concluyente: con el diseño controlado, W30 frente a W90 cruza el cero en 1 de las 18
combinaciones y no alcanza el criterio.

---

## 10. Verificaciones que sostienen lo anterior

**Métricas.** El Average Precision reportado coincide con
`sklearn.metrics.average_precision_score` con una diferencia de 5,55e−17. El costo
por transacción recalculado a mano coincide con el reportado.

**Sorteo del analista simulado.** Sobre 200 000 identificadores la media es 0,50006
y la desviación 0,28866. Kolmogorov-Smirnov contra la uniforme da D = 0,00031 con
p = 1,0. La autocorrelación de orden 1 es 0,039.

**Causalidad.** Las pruebas alteran las filas futuras, invierten las etiquetas
inmaduras y permutan los identificadores de eventos con el mismo timestamp,
comprobando en cada caso que la salida no cambia bit a bit.

**Cadena completa.** `test_backtest_end_to_end.py` recorre `run_backtest` con las
siete estrategias y comprueba que la economía calibrada llega al manifiesto de cada
paquete persistido. Esa prueba faltaba, y su ausencia dejó pasar un cambio de firma
que habría roto la corrida real tras cuarenta minutos de tuning.

**Paridad entre servicio y backtest.** `test_paridad_entre_el_servicio_y_el_calculo_offline`
comparaba solo probabilidades. Eso dejó pasar que el servicio HTTP, el replay y los
fixtures de contrato propusieran la acción con `Policy.zone`, que ignora el monto y
el precio del cupo: con los umbrales de referencia en 0 y 1, las tres rutas mandaban
todas las transacciones a revisión y el cupo las recortaba después, de modo que el
resultado parecía razonable. La prueba compara ahora la acción emitida y los tres
costos esperados, y una prueba adicional comprueba que un monto de 1 UM y uno de
5000 no reciben la misma acción.

**Reproducibilidad.** Dos corridas independientes de la cadena completa producen
resultados idénticos en los 18 fits de tuning, en el hash de prerregistro y en el
costo de las estrategias. `fraud-adaptive verify --against` lo comprueba a demanda y
`reports/verificacion_reproducibilidad.json` guarda la comparación.

---

## 11. Lo que el experimento no responde

El plan declara E15 como referencia no desplegable. Es la estrategia de menor costo
y es perfectamente desplegable, porque consiste en reentrenar sobre todo el
histórico cada 15 días. La exclusión responde a que el enunciado prioriza ventanas
deslizantes, no a un argumento técnico. El informe la reporta como desplegable.

La capacidad sigue siendo la restricción dominante aunque el precio sombra la
administre mejor. Con 150 plazas diarias, el sistema resuelve de forma automática la
gran mayoría de las transacciones y la calidad del modelo tiene rendimientos
decrecientes.

El efecto no monótono de la antigüedad queda sin explicar. Retroceder el corte 30
días encarece de forma concluyente, pero tomar los 46 primeros días del histórico no
se distingue de la ventana más fresca. Una revisión posterior debería añadir cortes
intermedios para caracterizar la forma de esa curva antes de afirmar nada sobre ella.
