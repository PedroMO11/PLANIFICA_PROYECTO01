# Sistema adaptativo de decisión ante fraude transaccional

**Curso** Planificación y Toma de Decisiones en IA, UTEC
**Dataset** IEEE-CIS Fraud Detection, partición `train`, 590 540 transacciones en 182 días
**Semilla** 42. **Retraso de etiqueta** L = 30 días. **Cadencia** 15 días

> **Resultado principal.** El olvido por ventanas fijas no reduce el costo sobre
> IEEE-CIS con este protocolo. La ventana de 9 días de fit resulta un 17,3 % más
> cara que el modelo estático. El factor que explica el ordenamiento es el volumen
> de entrenamiento, con una correlación de −0,9916 entre días de fit y costo. A
> volumen igual la frescura no produce efecto medible.

---

## Página 1. El problema de decisión

### Predecir no basta

El sistema decide, para cada transacción, entre aprobar, enviar a revisión o
bloquear. Cuatro restricciones definen qué significa un buen modelo en este
contexto.

1. La etiqueta confirmada llega 30 días después del evento. Ninguna métrica de
   error está disponible en el momento de decidir.
2. La capacidad humana de revisión es de 150 casos diarios. Constituye un techo,
   no un parámetro a optimizar.
3. Un falso negativo cuesta el monto de la transacción. Un falso positivo cuesta
   fricción con un cliente legítimo.
4. La distribución de los datos cambia con el tiempo.

### Objetivos y verificación

| Obj. | Enunciado | Criterio | Resultado obtenido |
|---|---|---|---|
| O1 | Menor costo que la política actual y que el modelo estático | Inferior a 5,40 y a 1,53 UM/tx | Parcial. Supera aprobar todo. No supera al estático |
| O2 | Limitar la caída de PR-AUC | Caída superior al 10 % genera alerta | El AP de S0 pasa de 0,526 a 0,509 entre bloques extremos |
| O3 | Respetar la capacidad de revisión | 150 casos diarios como máximo | Cumplido en 62 días sin excepción |
| O4 | Señales sin etiqueta que anticipen la caída | Medir el retraso de cada señal | KS/PSI y S1 el mismo día. ADWIN a 30 días |
| O5 | Limitar la disparidad entre segmentos | Brecha superior a 2 pp genera alerta | 7 de 14 segmentos superan el umbral |

### Familias de métricas

Las métricas técnicas comprenden Average Precision, skill normalizado por
prevalencia, F1 y balanced accuracy en un umbral declarado, recall a FPR máximo
del 1 %, recall a precisión mínima del 80 %, Brier, ECE y latencia p50, p95 y p99.

Las métricas de decisión comprenden costo observado simulado por transacción,
monto de fraude evitado, costo de falsos positivos, retorno por analista hora,
revisiones diarias y demanda excedente.

Las métricas sociales comprenden el bloqueo de clientes legítimos, distinto del
FPR del clasificador porque incorpora las legítimas que el analista bloquea por
error, la disparidad por segmento con soporte e intervalos, y la cobertura
automática.

El sistema distingue el cambio en P(X) del cambio en P(y|X). KS/PSI y el domain
classifier detectan el primero. Solo ADWIN sobre el error real observa el segundo,
con 30 días de retraso.

---

## Página 2. Arquitectura y modelo económico

![Resultados de adaptación](figures/drift_adaptacion.png)

*Figura 1. Comparación de las cinco estrategias sobre los mismos eventos. El
diagrama de los cinco componentes está en `docs/arquitectura.mmd`.*

### Componentes

La adquisición integra dos fuentes tabulares mediante left join uno a uno por
`TransactionID`, validado por invariancia del número de filas. El predictor
compara regresión logística, Random Forest y LightGBM, con el preprocessing
ajustado dentro de cada fit. La decisión aplica el mínimo de los tres costos
esperados, sujeto al cupo diario de revisión. La incertidumbre se trata con
calibración de Platt por versión e intervalos por bootstrap de bloques. La acción produce tres salidas
simuladas con cola priorizada y apelación documentada.

### Cálculo de la decisión

Para probabilidad calibrada `p` y monto `m`,

```
E[aprobar]  = p * m
E[bloquear] = (1 - p) * c_FP
E[revisar]  = c_R + p * (1 - r_H) * m + (1 - p) * f_H * c_FP
```

con `c_R = 1 UM`, `r_H = 0,90` y `f_H = 0,02`. La unidad monetaria es consistente
con `TransactionAmt` y no se convierte a moneda.

**La acción es el argmin de los tres costos.** No hay umbrales sobre `p` que
decidan. El punto de indiferencia entre aprobar y bloquear es
`p* = c_FP / (m + c_FP)`, de modo que depende del monto: un corte fijo bloquea de
más en los montos bajos y de menos en los altos. Medido sobre los cuatro bloques de
test con la misma economía y el mismo cupo, la regla de dos umbrales cuesta
1,9696 UM/tx frente a 1,5895 del argmin, con el mismo número de
revisiones y más bloqueo de legítimas.

La revisión no siempre resulta más barata. Con montos bajos el costo fijo `c_R`
supera la pérdida esperada y la política prefiere una acción automática. Esa es la
razón de que exista una tercera acción y no una decisión binaria.

**`c_FP` no se fija a mano.** No es observable, pero la fracción de transacciones
legítimas rechazadas sí lo es y es lo que la operación restringe. Se declara el
objetivo, un 1 %, y se busca el menor `c_FP` que lo cumple sobre la reserva de
política de desarrollo. El valor resultante, 25 UM, es el precio sombra de esa
restricción y queda sellado en el prerregistro. Un valor plano de 5 UM, que es el
que fijaba el plan, produce un 11,17 % de bloqueo de legítimas.

La calibración usa como referencia la familia de mayor AP y no la de menor costo,
porque el costo depende de `c_FP` y elegir por costo antes de calibrarlo sería
circular.

**La revisión compite por un cupo escaso.** Que sea la acción más barata no basta:
tiene que serlo por un margen que justifique ocupar una de las 150 plazas diarias.
Ese margen es el precio sombra del cupo, el multiplicador de Lagrange de la
restricción de capacidad. Se revisa solo si

```
min(E[aprobar], E[bloquear]) - E[revisar] > lambda
```

Sin ese precio, la regla propone 704 revisiones diarias frente a un cupo de 150.
El cupo se llena con los casos que llegan primero y los que más ahorrarían quedan
fuera. `lambda` se calibra por bisección como el menor valor que deja la demanda
diaria dentro del cupo, lo que la sitúa en 150,0 exactas. El criterio es la
capacidad y no el costo observado, de modo que la calibración no usa etiquetas. El
`lambda` que iguala demanda y cupo queda a un 0,7 % del que minimiza el costo.

El valor calibrado es 7,86 UM. Ambos parámetros se condicionan
mutuamente, porque `c_FP` fija la escala de los costos y racionar el cupo cambia
cuántos casos terminan bloqueados, de modo que se calibran alternando hasta que
ambos se estabilizan. Convergen en tres pasadas.

### Evidencia sobre la necesidad de calibrar

La aritmética anterior exige que `p` sea una probabilidad. La configuración
ganadora de la regresión logística usa `class_weight=balanced`, que infla los
puntajes de forma severa. Los datos reales cuantifican el efecto.

| Familia | Brier crudo | Brier calibrado | ECE crudo | ECE calibrado |
|---|---|---|---|---|
| Logística `lr_c1_bal` | 0,14988 | 0,03170 | 0,29388 | 0,00437 |
| Random Forest `rf_200_20` | 0,05440 | 0,02521 | 0,14777 | 0,00433 |
| LightGBM `lgbm_31` | 0,01991 | 0,01978 | 0,00348 | 0,00130 |

El ECE de la logística mejora en un factor de 67. Sin esa corrección, usar el
puntaje crudo como probabilidad produciría un exceso de bloqueos que el AP no
reflejaría.

### Cola y cupo

El cupo se reserva al admitir, de forma irrevocable y por `event_id`. El sistema
no ordena el día completo por `p * monto` para retener los 150 mejores casos,
porque esa operación requeriría conocer transacciones que aún no ocurrieron. La
prioridad ordena el servicio entre los casos ya admitidos. Agotado el cupo, el
caso recibe la más barata entre aprobar y bloquear.

### Supervisión humana

Ninguna alerta despliega por sí sola. El reentrenamiento y la promoción exigen
autorización humana registrada aunque todos los gates técnicos pasen. Sin versión
válida el servicio responde 503 y pausa, sin decidir el pago ni degradar a una
referencia no desplegable.

---

## Página 3. Datos, integración y causalidad

![EDA temporal](figures/eda_temporal.png)

*Figura 2. Prevalencia semanal con intervalo de Wilson, volumen, faltantes por
familia de columnas y cobertura de identidad.*

### Integración auditada

| Comprobación | Resultado |
|---|---|
| Transacciones | 590 540 filas, 394 columnas |
| Identidad | 144 233 filas, 41 columnas |
| Filas tras el join | 590 540, invariante |
| Cobertura de identidad | 24,42 % |
| Identificadores de identidad huérfanos | 0 |
| Prevalencia global | 3,50 %, 20 663 fraudes |
| Monto mediano | 68,77 |

La ausencia de identidad constituye información y se preserva en `has_identity`.
Eliminar esas filas sesgaría el panel hacia clientes con dispositivo identificado.

### Relojes derivados

`TransactionDT` es un delta en segundos desde un origen desconocido, no una fecha.
De él se derivan `dia`, `semana` y una `hora` relativa que funciona como ciclo de
24 horas y no identifica hora local ni día laboral.

El reloj de evento determina qué features son visibles. El reloj de disponibilidad,
definido como `available_at = dia + 30`, determina cuándo una etiqueta puede
entrar a un fit, a una métrica o a ADWIN.

### Features causales y calidad de los proxies

El sistema construye dos proxies de entidad. El proxy de tarjeta combina `card1` a
`card6` con `addr1`. El proxy de dispositivo combina `DeviceType` con
`DeviceInfo`. Para cada uno calcula conteos y montos en ventanas de 1 hora, 24
horas y 7 días, rezago del monto, tiempo desde el último evento y conteo expansivo.

| Proxy | Definición | Entidades | Eventos por entidad | Singletons |
|---|---|---|---|---|
| Emisor | `card1` a `card6` más `addr1` | 43 018 | 13,7 | 40,4 % |
| Cliente | `card1 + addr1 + D1n` | 217 850 | 2,7 | 57,5 % |
| Dispositivo | `DeviceType + DeviceInfo` | 1 943 | 74,2 | 26,0 % |

Las dos claves de tarjeta se usan a la vez. `D1` mide los días transcurridos desde
que la tarjeta empezó a usarse, de modo que restarlo del día absoluto identifica la
fecha de alta y separa tarjetas distintas del mismo emisor. Entrenando LightGBM
solo con features de historial sobre un test temporal, cada clave por separado
alcanza AP de 0,1668 y 0,1671, y juntas 0,1829. La gruesa aporta soporte
estadístico y la fina aporta resolución por tarjeta, de modo que no son redundantes.

Un proxy agrupa comportamiento y no identifica personas.

**Los conteos expansivos quedan fuera del predictor.** Acumulan desde el primer
evento, de modo que crecen de forma monótona con el calendario y funcionan como
sustituto del día absoluto, que el protocolo prohíbe. La media de
`card_cnt_expansivo` pasa de 138,78 en `[0,60)` a 707,43 en `[150,182)`, con una
correlación de +0,2276 con el día, mientras `card_cnt_7d` marca −0,009.

Una validación adversarial por feature, el método que usaba la competencia para
descartar columnas con AUC superior a 0,70, confirma el diagnóstico: de las 425
columnas del panel solo dos superan ese umbral y ambas son conteos expansivos, con
0,7066 y 0,7006. La mediana del panel es 0,5053 y la feature original más alta es
`C9` con 0,6002. Las covariables de IEEE-CIS son estables, lo que anticipa el
resultado del experimento central.

### Controles de leakage

Las 145 pruebas de `tests/` verifican los controles. Cuatro resultan decisivas.
Alterar las filas futuras no modifica medianas ni vocabulario del preprocessing.
Invertir las etiquetas aún inmaduras no cambia el predictor. Permutar los
identificadores de eventos con el mismo timestamp deja las features idénticas. Los
cuatro roles de cada versión no comparten ningún `TransactionID`.

---

## Página 4. Protocolo temporal

### Particiones

Todos los intervalos son semiabiertos y se expresan en días relativos.

```
dia:  0        30   45   60      69   76   83   90        120  135  150  165  182
      |---------|----|----|-------|----|----|----|---------|----|----|----|----|
      [ tuning fold 1 ]
      [ tuning fold 2      ]
      [     fit base / S0          ]
                                   [cal][pol][ H ]
                                                  [warmup X ]
                                                            [ B1][ B2][ B3][ B4 ]
```

En cada actualización `T` de {120, 135, 150, 165}, con `c = T - 30`, los roles se
distribuyen de la forma siguiente.

| Rol | Intervalo | Días efectivos |
|---|---|---|
| Predictor | `[c-W, c-14)` | 16 con W30, 46 con W60, 76 con W90 |
| Calibrador | `[c-14, c-7)` | 7 |
| Validación de promoción | `[c-7, c)` | 7 |

Hay dos reservas por versión, no tres. Los umbrales de referencia se eligen una
sola vez en desarrollo y quedan congelados, de modo que reservar una ventana de
política en cada actualización excluía siete días del fit sin cumplir función.
Liberarla beneficia sobre todo a las ventanas cortas, que son las que el
experimento evalúa: W30 pasa de 9 a 16 días de fit.

Las dos reservas son idénticas para todas las estrategias. Esa igualdad permite
atribuir una diferencia de costo al tamaño de ventana.

**El experimento separa volumen de frescura.** Variar solo `W` cambiaba a la vez
cuántos días entrena el modelo y cuán reciente es su información, de modo que
ninguna diferencia era atribuible. El diseño vigente añade un bloque de antigüedad
variable con volumen constante.

| Bloque | Estrategias | Qué varía | Qué se mantiene |
|---|---|---|---|
| Volumen variable | W30, W60, W90 | días de fit | corte en `c-14` |
| Antigüedad variable | W60, R46_medio, R46_antiguo | punto de corte | 46 días de fit |

### Distinción entre cutoff y tiempo del job

`cutoff = T - L` delimita qué días de evento pueden usarse. `job_time = T` indica
cuándo se ejecuta el ajuste y constituye el valor contra el que se compara
`available_at`. Con T igual a 120 y L igual a 30, el último día de evento con
etiqueta confirmada es el 89. Comparar contra `cutoff` en lugar de `job_time`
exigiría `dia < 60` y dejaría vacías las colas de calibración y validación.

### Orden de construcción

El orden es predictor, calibrador y validación. Cada paso consume scores
producidos por el anterior sobre datos que ese paso no vio. Invertir el orden haría
que el calibrador corrigiera un modelo distinto del que se despliega. La política
no consume una reserva propia, porque se congela en desarrollo y consiste en el
modelo de costos.

Si un rol no alcanza el soporte mínimo de 200 fraudes y 2 000 legítimas en el fit,
la versión se marca no válida. El sistema no amplía la ventana ni incorpora
etiquetas inmaduras para completar el soporte.

### Prerregistro

La familia, los hiperparámetros, la economía calibrada y la ventana desplegable se
congelan usando solo desarrollo. El hash `7d4c9fac0af4d15f` sella esa selección antes de
abrir el test. `c_FP` entra al sello porque pasó de ser una constante del config a
un parámetro derivado de los datos de desarrollo, de modo que sin sellarlo sería
posible recalibrarlo después de ver el test. Si la configuración cambiara entre la selección y el test, la
corrida se detiene.

El benchmark histórico que seleccionó IEEE-CIS ya examinó periodos tardíos del
dataset. Este trabajo continúa siendo un backtest retrospectivo y no una
validación prospectiva.

---

## Página 5. Modelos y selección en desarrollo

![Rendimiento del modelo estático](figures/rendimiento_estatico.png)

*Figura 3. Evolución de AP, costo y Brier del sistema estático en el periodo de
test, con la tasa base superpuesta.*

### Comparación de familias

La selección aplica el criterio de costo, no el de AP. El proceso ejecuta 18 fits
de tuning y tres ajustes finales.

| Familia | Config | AP | Costo UM/tx con la regla económica | Costo con el mejor umbral fijo |
|---|---|---|---|---|
| LightGBM | `lgbm_31` | **0,6214** | **1,5895** | 1,9696 |
| Random Forest | `rf_200_20` | 0,5391 | 1,8728 | 2,2746 |
| Logística | `lr_c1_bal` | 0,4029 | 2,1910 | 2,8712 |

Las referencias simuladas sobre la reserva de política alcanzan 4,5937 UM/tx
al aprobar todo y 24,1365 UM/tx al bloquear todo. LightGBM queda
congelada como familia adaptativa.

La columna de umbral fijo es el costo que alcanzaría el mejor par de cortes
globales sobre `p`, elegido por minimización sobre la misma reserva. La diferencia
con la columna de argmin mide lo que cuesta ignorar el monto.

La regla económica induce un umbral distinto por monto. Con la economía calibrada:

| Monto | Revisar desde | Bloquear desde |
|---|---|---|
| 25,00 UM | `p` ≥ 0,4072 | `p` ≥ 0,5791 |
| 59,00 UM | `p` ≥ 0,1747 | `p` ≥ 0,5143 |
| 250,00 UM | `p` ≥ 0,0415 | `p` ≥ 0,3159 |

### Selección de la ventana desplegable

El protocolo exige elegir la ventana con datos de desarrollo. La evaluación se
realiza sobre la reserva de validación, posterior al predictor y al calibrador de
cada paquete.

| Estrategia | Días de fit | Fraudes en el fit | Costo UM/tx en desarrollo |
|---|---|---|---|
| **W60** | 46 | 5 768 | **2,2527** |
| W90 | 76 | 9 169 | 2,2708 |
| W30 | 16 | 2 191 | 2,3501 |

W60 y W90 quedan separadas por un 0,80 %, dentro del margen de desempate del 1 %
que favorece la ventana menor. W30 queda un 4,33 % por encima. El desarrollo
anticipa que la ventana más corta es la peor de las tres.

W60 constituye la recomendación operativa y fue elegida sin observar el test.

---

## Página 6. Resultado del experimento central

Siete estrategias evaluadas sobre los mismos eventos, la misma semilla, la misma
política y la misma familia. 175 998 eventos por estrategia en cuatro bloques.
Las tres primeras varían el volumen de entrenamiento y las tres últimas lo
mantienen constante en 46 días variando solo la antigüedad.

| Estrategia | Días de fit | AP | Brier | Costo UM/tx con IC 95 % | Δ frente a S0 | Bloqueo legítimas | Versiones |
|---|---|---|---|---|---|---|---|
| E15 *(ref.)* | 76 a 121 | 0,4965 | 0,0231 | 1,9840 [1,821 y 2,155] | −0,108 [−0,205 y 0,002] | 1,26 % | 4 |
| W90 | 76 | 0,4901 | 0,0234 | 2,0573 [1,909 y 2,208] | −0,035 [−0,128 y 0,061] | 1,36 % | 4 |
| S0 *(ref.)* | 69 | 0,4828 | 0,0233 | 2,0921 [1,911 y 2,256] | referencia | 0,87 % | 1 |
| **W60** | 46 | 0,4802 | 0,0237 | 2,1087 [1,917 y 2,330] | 0,017 [−0,114 y 0,165] | 1,38 % | 4 |
| R46_antiguo *(diag.)* | 46 | 0,4681 | 0,0236 | 2,1171 [1,943 y 2,246] | 0,025 [−0,059 y 0,090] | 0,98 % | 1 |
| W30 | 16 | 0,4701 | 0,0238 | 2,1830 [1,943 y 2,421] | 0,091 [−0,061 y 0,280] | 1,27 % | 4 |
| R46_medio *(diag.)* | 46 | 0,4571 | 0,0242 | 2,2146 [2,033 y 2,382] | 0,123 [0,004 y 0,229] | 1,40 % | 4 |

*Intervalos por bootstrap pareado de bloques contiguos de 7 días con 200
remuestreos.*

### Qué diferencias son concluyentes

Un intervalo obtenido con una semilla fija puede excluir el cero por azar. Cada
diferencia se reevaluó con 6 semillas y 3 tamaños de bloque, 18 combinaciones en
total, contando en cuántas el intervalo cruza el cero.

| Diferencia | Efecto | Cruzan cero | Veredicto |
|---|---|---|---|
| E15 frente a S0 | −0,1081 | 8 de 18 | no concluyente |
| W90 frente a S0 | −0,0348 | 18 de 18 | no concluyente |
| W60 frente a S0 | 0,0166 | 18 de 18 | no concluyente |
| R46_antiguo frente a S0 | 0,0250 | 18 de 18 | no concluyente |
| W30 frente a S0 | 0,0909 | 18 de 18 | no concluyente |
| R46_medio frente a S0 | 0,1225 | 13 de 18 | no concluyente |

Ninguna de las 6 diferencias frente a S0 resiste el control. Todas cruzan el
cero en al menos una de las 18 combinaciones, de modo que el informe no declara
concluyente ninguna comparación contra el modelo estático.

### Volumen frente a frescura

El experimento separa los dos factores por diseño. El bloque de volumen variable
mueve los días de fit manteniendo el corte, y el bloque de antigüedad variable
mueve el corte manteniendo 46 días de fit.

| Comparación | Qué aísla | Efecto | Cruzan cero | Veredicto |
|---|---|---|---|---|
| R46_antiguo frente a W60 | Frescura, corte al inicio del histórico | 0,0084 | 18 de 18 | no concluyente |
| W60 frente a W90 | Volumen, 46 frente a 76 días | 0,0514 | 18 de 18 | no concluyente |
| W30 frente a W60 | Volumen, 16 frente a 46 días | 0,0742 | 4 de 18 | no concluyente |
| R46_medio frente a W60 | Frescura, corte 30 días más antiguo | 0,1059 | 0 de 18 | **concluyente** |
| W30 frente a W90 | Volumen, 16 frente a 76 días | 0,1257 | 1 de 18 | no concluyente |

Solo 1 de las 5 comparaciones controladas resiste el control de robustez, la de
frescura: mantener el volumen en 46 días y retroceder el corte 30 días encarece
la decisión en 0,1059 UM/tx, con las 18 combinaciones excluyendo el cero.

Mantener el corte y variar el volumen produce un efecto de magnitud parecida,
0,1257 entre 16 y 76 días de fit, pero menos estable: cruza el cero en 1 de las
18 combinaciones y no se declara concluyente.

El efecto de la frescura no es monótono. Retroceder el corte 30 días encarece,
pero tomar los 46 primeros días del histórico no se distingue de la ventana más
fresca, con 0,0084 y un intervalo que cruza el cero en las 18 combinaciones. La
correlación entre días de fit y costo sobre las estrategias de tamaño fijo es de
−0,7412, frente al −0,9916 que producía el diseño confundido.

Las señales de drift explican por qué los efectos son pequeños y por qué el que
sobrevive es el de frescura. El domain classifier obtiene un AUC de 0,551,
por debajo del umbral de alerta de 0,75, de modo que P(X) se mantiene estable
durante los 182 días. La validación adversarial por feature lo confirma de forma
independiente: ninguna columna original del dataset supera un AUC de 0,61 entre el
primer y el último bloque.

Sin deriva de covariables, deslizar la ventana no aporta distribuciones de entrada
nuevas, y por eso el efecto del volumen no se distingue del ruido. El efecto de la
frescura, que sí sobrevive, no puede venir entonces de P(X). Apunta a P(y|X), que es
donde ADWIN registra deriva. Es también la deriva que el protocolo no puede
identificar causalmente con datos observacionales.

### AP por bloque

El AP del sistema estático pasa de 0,5257 en B1 a 0,5144 en B4, sin degradación
monótona. El descenso común en B2, hasta 0,3907, afecta a las siete estrategias
por igual, incluidas las que se reentrenan, de modo que corresponde a un periodo
más difícil y no a un modelo que caduca. El detalle por bloque está en
`reports/tables/adaptacion.csv`.

### Objetivos diagnósticos no alcanzados

Los umbrales diagnósticos se eligieron en validación y se aplicaron congelados.
Ninguna estrategia alcanza el FPR máximo del 1 % ni la precisión mínima del 80 %
en test. El FPR obtenido queda entre 0,78 % y 1,18 % y la precisión entre 75,4 %
y 82,5 %.
El informe reporta el nivel obtenido y no ajusta el umbral, porque hacerlo
convertiría un objetivo incumplido en un resultado ajustado a posteriori.

### Señales de drift y gates

ADWIN registra 33 detecciones repartidas entre las cinco estrategias, todas con
exactamente 30 días de retraso. Existe deriva en el error real. KS/PSI y S2 generan
5 alertas sobre 373 registros, concentradas a partir del día 170, y el domain
classifier no supera su umbral. El error deriva mientras las covariables se
mantienen estables.

En T igual a 135 las cinco estrategias fueron bloqueadas por el gate social, con
brechas de entre 2,76 y 4,69 puntos porcentuales. En T igual a 150 y 165 las
brechas caen por debajo de 1,3 y las promociones proceden. W30 falló además el gate
de costo en T igual a 150. Los gates discriminaron.

---

## Página 7. Despliegue, operación y costo

### Entregado frente a diseñado

| Entregado y verificado localmente | Ejecuta el equipo de forma manual |
|---|---|
| Paquete versionado con hashes | Publicación de la imagen en Artifact Registry |
| Imagen Docker construida y probada | Creación del servicio en Cloud Run |
| Servicio HTTP con `/health` y `/predict` | Apuntar el replay al endpoint remoto |
| Replay con ledger SQLite transaccional | Promoción, rollback y cierre de recursos |
| Runbook y política de promoción | |

Pub/Sub, Firestore, BigQuery, Cloud Scheduler y Vertex AI Pipelines figuran en el
diagrama como diseño futuro. No existen recursos creados ni código de integración.

### Evidencia de la demo local

| Comprobación | Resultado |
|---|---|
| Tres acciones sobre datos reales | Presentes en el replay |
| Cupo respetado | 150 de 150, nunca excedido en 62 días |
| Idempotencia con 50 reenvíos | Aprobada, sin consumo adicional de cupo |
| Imagen Docker | `linux/amd64`, 1,06 GB, HEALTHCHECK en verde |
| Paridad contenedor frente a cálculo offline | 5 de 5 idénticas con tolerancia 1e−12 |
| Latencia p95 sobre HTTP al contenedor | 87,2 ms frente a un objetivo de 300 ms |
| Sin paquete montado | `/health` informa `model_unavailable` y `/predict` devuelve 503 |
| Cambio de versión y rollback | Score restaurado de forma exacta |

### Frecuencia, autonomía y gates

El reentrenamiento ocurre cada 15 días con la ventana W60 elegida en desarrollo.
El monitoreo aplica KS/PSI a diario, S1 por bloque y ADWIN al madurar cada
etiqueta. Cinco gates controlan la promoción. El primero verifica integridad y
bloquea la versión ante cualquier leakage. El segundo exige costo inferior al de
aprobar todo en el arranque. El tercero aplica no inferioridad con costo máximo de
1,01 veces el del champion, caída de AP no superior a 0,01 y aumento de Brier no
superior a 0,005. El cuarto limita el aumento del bloqueo de legítimas a 2 puntos
porcentuales. El quinto exige autorización humana registrada.

El rollback técnico responde a errores HTTP superiores al 1 % o a p95 por encima de
300 ms en dos lotes consecutivos, y opera de forma inmediata bajo regla
preautorizada. El deterioro de negocio solo puede afirmarse con etiquetas maduras,
30 días después, y exige revisión humana.

### Costo de cómputo

| Trabajo | Tareas | Minutos |
|---|---|---|
| Tuning | 18 | 17,3 |
| Ajustes finales y de adaptación | 6 | 9,9 |
| Selección de ventana | 3 | 0,8 |
| Backtest completo | 1 | 3,2 |
| Datos y features | 5 | 0,3 |
| Total | 33 | 31,5 de 480 |

El perfil reducido de recursos previsto en el plan no resultó necesario.

### Escala, costo e integración

El informe no estima montos porque dependen de la cuenta, la región y el tráfico.
Los factores de costo son el consumo de CPU y memoria por petición junto con el
cold start en serving, el tamaño de la ventana y el ancho del panel en
reentrenamiento, y el número de versiones retenidas junto con el volumen de logs en
almacenamiento.

Tres límites quedan declarados. El cupo está garantizado únicamente para un
orquestador secuencial. La latencia medida es local y no representa una región
cloud. El prototipo asume identidad simultánea y features precomputadas.

---

## Página 8. Límites y conclusión

### Afirmaciones que no se sostienen

| Afirmación | Motivo |
|---|---|
| El olvido mejora el costo | No ocurre sobre IEEE-CIS. Ninguna ventana deslizante bate al estático |
| Existe concept drift demostrado | ADWIN detecta deriva en el error. La comparación temporal no identifica causalmente el cambio en P(y\|X) |
| La ventana elegida es la óptima | Es la mejor de tres evaluadas en desarrollo, con una cadencia y una semilla |
| El sistema es justo | La disparidad medida es operativa sobre variables de negocio, sin atributos protegidos verificables |
| El sistema funciona en producción | El cupo está garantizado solo para un orquestador secuencial |
| Existe un ahorro económico medido | Los costos son simulados bajo supuestos declarados |
| `c_FP` representa el costo real de un falso positivo | Es el precio sombra del objetivo de bloqueo declarado, no una medición de negocio |

### Por qué el olvido no compensa en este dataset

Las señales medidas ofrecen una explicación coherente y convergente.

El domain classifier alcanza un AUC de 0,551, por debajo del umbral de
0,75, lo que indica estabilidad de P(X) durante los 182 días. La validación
adversarial por feature, aplicada columna a columna entre el primer y el último
bloque, da una mediana de 0,5053 sobre 425 features y un máximo de 0,6002 en `C9`.
Ninguna columna original del dataset alcanza el umbral de 0,70 que la competencia
usaba para descartar features inestables.

En ese régimen la ventana corta pierde información sin ganar frescura útil.

El soporte refuerza la explicación. W30 entrena con 2 191 fraudes frente a los 9 169 de W90. El proxy de emisor
presenta un 40 % de singletons y el de cliente un 57 %, de modo que las features
de historial aportan menos de lo que aportarían con entidades recurrentes.

El resultado no invalida el diseño adaptativo. Indica que la cadencia de 15 días
con L igual a 30 no encuentra, en este dataset, una deriva suficiente para
justificar el descarte de datos.

### Selective labels y restricción dominante

El núcleo emplea información completa, de modo que toda etiqueta madura a los 30
días incluso si la acción simulada fue bloquear. En operación real una transacción
bloqueada no revela su desenlace y las etiquetas disponibles quedarían sesgadas por
las decisiones previas del sistema. La corrección mediante masked label quedó fuera
del alcance.

La demanda de revisión supera el cupo de 150 casos diarios de forma sistemática. El
cuello de botella es la capacidad humana y no la calidad del modelo. Mejorar el AP
presenta rendimientos decrecientes mientras la mayor parte de los casos que la
política enviaría a revisión reciban una acción automática por falta de capacidad.

Los riesgos residuales cuentan con controles entregados. Los scores descalibrados
se corrigen con Platt por versión sobre una cola separada. La oscilación de
versiones se limita con cadencia y cooldown de 15 días sin retuning por alertas.
Las etiquetas tardías se gobiernan con `available_at` y predicciones inmutables.
La ausencia de modelo válido produce 503 y pausa. El retuning tras observar el test
queda bloqueado por el hash de prerregistro.

### Conclusión

El sistema reduce el costo observado de 5,40 a 1,98 UM por transacción frente a
aprobar todo, una mejora del 63,2 %. Ese resultado proviene del modelo y de la
política económica, no de la adaptación.

El olvido por ventanas fijas no aporta mejora sobre IEEE-CIS con este protocolo.
La estrategia de menor costo es E15, con 1,9840 UM/tx, que es la que usa todo el
histórico disponible. La de mayor costo es R46_medio, con 2,2146. Ninguna de las
6 diferencias frente a S0 resiste el control. Todas cruzan el cero en al menos
una de las 18 combinaciones, de modo que el informe no declara concluyente
ninguna comparación contra el modelo estático.

Las comparaciones controladas sí encuentran señal. Mantener el volumen constante
y retroceder el corte 30 días encarece la decisión en 0,1059 UM/tx, con las 18
combinaciones de semilla y bloque excluyendo el cero. La frescura produce por
tanto un efecto medible, al contrario de lo que sugería el diseño que confundía
volumen con antigüedad. El efecto del volumen es de magnitud parecida, 0,1257
entre 16 y 76 días de fit, pero no resiste el mismo control.

La lectura conjunta es que ambos factores importan poco. El rango completo de
las siete estrategias va de 1,98 a 2,21 UM/tx, un 11,6 %, y ninguna se distingue
del modelo estático de forma estable.

Ninguna estrategia alcanza los objetivos diagnósticos de FPR máximo del 1 % ni
de precisión mínima del 80 %. La capacidad de revisión resulta más limitante que
la calidad del modelo.

La recomendación operativa es conservar el modelo con el mayor volumen de
entrenamiento disponible y monitoreo activo. W60 fue la ventana elegida en
desarrollo entre las tres deslizantes, pero el protocolo no ofrece evidencia de
que el olvido aporte valor sobre este dataset. Una revisión posterior debería
evaluar cadencias y retrasos distintos antes de descartar el enfoque adaptativo,
porque la deriva de IEEE-CIS parece operar a una escala temporal mayor que la
del protocolo evaluado.

Este trabajo constituye un backtest retrospectivo. Un backtest no demuestra que el
sistema funcione en el futuro.

---

### Trazabilidad

Cada cifra es trazable a un `run_id` y a un artefacto con hash en
`reports/indice_evidencia.csv`.

Los antecedentes son `propuesta_proyecto1_final.md`, `concept_drift_findings.md` y
`concept_drift_benchmark_instructions.md`. El benchmark que seleccionó IEEE-CIS se
cita como evidencia consistente con drift, no como prueba causal, y sus
limitaciones de preprocessing quedan declaradas.

Las siete decisiones en que este sistema se aparta del plan de implementación
están en `docs/decisiones_de_diseno.md`, cada una con la medición que la sustenta.

El protocolo está en `docs/protocolo_experimental.md`, el contrato en
`docs/contrato_sistema.md` y la reproducción en `docs/reproducibilidad.md`. El
despliegue está en `deploy/gcp_runbook.md` y `deploy/promocion_rollback.md`. La
evidencia ejecutable consta de tres notebooks en `notebooks/` y 145 pruebas en
`tests/`.
