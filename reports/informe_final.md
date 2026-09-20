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
ajustado dentro de cada fit. La decisión aplica una política de costo esperado con
dos umbrales y cupo. La incertidumbre se trata con calibración de Platt por
versión e intervalos por bootstrap de bloques. La acción produce tres salidas
simuladas con cola priorizada y apelación documentada.

### Cálculo de la decisión

Para probabilidad calibrada `p` y monto `m`,

```
E[aprobar]  = p * m
E[bloquear] = (1 - p) * c_FP
E[revisar]  = c_R + p * (1 - r_H) * m + (1 - p) * f_H * c_FP
```

con `c_FP = 5 UM`, `c_R = 1 UM`, `r_H = 0,90` y `f_H = 0,02`. La unidad monetaria
es consistente con `TransactionAmt` y no se convierte a moneda.

La revisión no siempre resulta más barata. Con montos bajos el costo fijo `c_R`
supera la pérdida esperada y la política prefiere una acción automática. Esa es la
razón de que exista una zona gris en lugar de un umbral único.

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

| Proxy | Entidades | Eventos por entidad | Fracción de singletons |
|---|---|---|---|
| Tarjeta | 43 018 | 13,7 | 40,4 % |
| Dispositivo | 1 943 | 74,2 | 26,0 % |

El 40 % de singletons en el proxy de tarjeta limita el soporte de las features de
historial. Un proxy agrupa comportamiento y no identifica personas.

### Controles de leakage

Las 137 pruebas de `tests/` verifican los controles. Cuatro resultan decisivas.
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
| Predictor | `[c-W, c-21)` | 9 con W30, 39 con W60, 69 con W90 |
| Calibrador | `[c-21, c-14)` | 7 |
| Reserva de política | `[c-14, c-7)` | 7 |
| Validación de promoción | `[c-7, c)` | 7 |

Las tres reservas son idénticas para todas las estrategias. Esa igualdad permite
atribuir una diferencia de costo al tamaño de ventana.

### Distinción entre cutoff y tiempo del job

`cutoff = T - L` delimita qué días de evento pueden usarse. `job_time = T` indica
cuándo se ejecuta el ajuste y constituye el valor contra el que se compara
`available_at`. Con T igual a 120 y L igual a 30, el último día de evento con
etiqueta confirmada es el 89. Comparar contra `cutoff` en lugar de `job_time`
exigiría `dia < 60` y dejaría vacías las colas de calibración, política y
validación.

### Orden de construcción

El orden es predictor, calibrador, política y validación. Cada paso consume scores
producidos por el anterior sobre datos que ese paso no vio. Invertir el orden haría
que el calibrador corrigiera un modelo distinto del que se despliega.

Si un rol no alcanza el soporte mínimo de 200 fraudes y 2 000 legítimas en el fit,
la versión se marca no válida. El sistema no amplía la ventana ni incorpora
etiquetas inmaduras para completar el soporte.

### Prerregistro

La familia, los hiperparámetros, los umbrales y la ventana desplegable se congelan
usando solo desarrollo. El hash `6029a7722645b957` sella esa selección antes de
abrir el test. Si la configuración cambiara entre la selección y el test, la
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

| Familia | Config | AP | Costo UM/tx | Umbrales |
|---|---|---|---|---|
| LightGBM | `lgbm_31` | **0,6105** | **1,1963** | 0,0067 y 0,9041 |
| Random Forest | `rf_200_20` | 0,5404 | 1,4957 | 0,0000 y 0,2408 |
| Logística | `lr_c1_bal` | 0,3986 | 1,5325 | 0,0158 y 0,2058 |

Las referencias simuladas sobre la reserva de política alcanzan 4,5937 UM/tx al
aprobar todo y 4,8273 UM/tx al bloquear todo. LightGBM queda congelado como
familia adaptativa.

Los umbrales no se fijan en 0,5. Resultan de minimizar el costo observado sobre
`[76,83)` recorriendo pares de cuantiles del score.

### Selección de la ventana desplegable

El protocolo exige elegir la ventana con datos de desarrollo. La evaluación se
realiza sobre la reserva de política, posterior al predictor y al calibrador de
cada paquete.

| Estrategia | Días de fit | Fraudes en el fit | Costo UM/tx en desarrollo |
|---|---|---|---|
| **W60** | 39 | 4 868 | **1,1909** |
| W90 | 69 | 8 269 | 1,1963 |
| W30 | 9 | 1 291 | 1,4743 |

W60 y W90 quedan separadas por un 0,45 %, dentro del margen de desempate del 1 %,
que favorece la ventana menor. W30 queda un 23,8 % por encima. El desarrollo
anticipa que la ventana más corta es la peor opción.

W60 constituye la recomendación operativa y fue elegida sin observar el test.

---

## Página 6. Resultado del experimento central

Cinco estrategias evaluadas sobre los mismos eventos, la misma semilla, la misma
política y la misma familia. 175 998 eventos por estrategia en cuatro bloques.

| Estrategia | AP | Skill | Brier | Costo UM/tx con IC 95 % | Δcosto frente a S0 con IC 95 % | Bloqueo legítimas | Versiones |
|---|---|---|---|---|---|---|---|
| E15 *(ref.)* | 0,4870 | 0,4687 | 0,0238 | **1,4961** [1,402 y 1,604] | **−0,032** [−0,058 y −0,009] | 11,51 % | 3 |
| W90 | 0,4775 | 0,4589 | 0,0241 | 1,5230 [1,443 y 1,623] | −0,005 [−0,035 y +0,026] | 11,87 % | 3 |
| S0 *(ref.)* | 0,4799 | 0,4614 | 0,0239 | 1,5282 [1,440 y 1,622] | referencia | 10,79 % | 1 |
| **W60** *(elegida)* | 0,4636 | 0,4445 | 0,0245 | 1,6262 [1,494 y 1,782] | +0,098 [+0,009 y +0,196] | 12,24 % | 3 |
| W30 | 0,4267 | 0,4062 | 0,0252 | **1,7924** [1,655 y 1,925] | **+0,264** [+0,174 y +0,367] | 12,82 % | 2 |

*Costo de aprobar todo en test 5,3978 UM/tx. Intervalos por bootstrap pareado de
bloques contiguos de 7 días con 200 remuestreos. S0 y E15 son referencias no
desplegables.*

### Qué diferencias son concluyentes

Un intervalo obtenido con una semilla fija puede excluir el cero por azar. Cada
diferencia se reevaluó con 6 semillas y 3 tamaños de bloque, 18 combinaciones en
total, contando en cuántas el intervalo cruza el cero.

| Diferencia | Efecto | Cruzan cero | Veredicto |
|---|---|---|---|
| W30 frente a S0 | +0,2642 | 0 de 18 | Concluyente |
| E15 frente a S0 | −0,0322 | 0 de 18 | Concluyente |
| W60 frente a S0 | +0,0980 | 12 de 18 | **No concluyente** |
| W90 frente a S0 | −0,0052 | 18 de 18 | Sin efecto |

Solo dos afirmaciones resisten. La ventana de 9 días de fit es peor que el modelo
estático. La referencia expansiva es mejor, en un 2,1 %. La diferencia de W60 no se
distingue del cero de forma estable y el informe no la declara concluyente.

### El experimento confunde volumen con frescura

Cada ventana W reserva 21 días para calibración, política y validación, de modo
que el predictor entrena con W menos 21 días. Variar W cambia a la vez cuánto
entrena el modelo y cuán reciente es su información. Las dos comparaciones
controladas que el propio diseño permite separan ambos factores.

| Comparación | Qué aísla | Efecto | IC 95 % |
|---|---|---|---|
| W90 frente a S0, ambos 69 días de fit | Frescura a volumen igual | −0,0052 | [−0,037 y +0,026] cruza cero |
| W30 frente a W90, ambas deslizantes | Volumen a frescura igual | +0,2695 | [+0,186 y +0,364] |

La correlación entre días de fit y costo sobre las cuatro estrategias de tamaño
fijo es de −0,9916. El factor que explica el ordenamiento es el volumen de
entrenamiento, no la frescura.

El resultado es coherente con las señales. El domain classifier obtiene un AUC de
0,654, por debajo del umbral de alerta de 0,75, de modo que P(X) se mantiene
estable durante los 182 días. Sin deriva apreciable en las covariables, deslizar la
ventana no aporta información nueva y reducirla solo resta muestra.

### AP por bloque

El AP del sistema estático pasa de 0,5255 en B1 a 0,5092 en B4, sin degradación
monótona. El descenso común en B2, hasta 0,3882, afecta a las cinco estrategias por
igual, incluidas las que se reentrenan, de modo que corresponde a un periodo más
difícil y no a un modelo que caduca. El detalle por bloque está en
`reports/tables/adaptacion.csv`.

### Objetivos diagnósticos no alcanzados

Los umbrales se eligieron en validación y se aplicaron congelados. Ninguna
estrategia alcanza el FPR máximo del 1 % ni la precisión mínima del 80 % en test.
El FPR obtenido queda entre 1,34 % y 1,45 % y la precisión entre 70,3 % y 72,7 %.
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
| El olvido mejora el costo | No ocurre sobre IEEE-CIS. W60 resulta un 6,4 % más cara que el estático |
| Existe concept drift demostrado | ADWIN detecta deriva en el error. La comparación temporal no identifica causalmente el cambio en P(y\|X) |
| W60 es la ventana óptima | Es la mejor de tres evaluadas en desarrollo, con una cadencia y una semilla |
| El sistema es justo | La disparidad medida es operativa sobre variables de negocio, sin atributos protegidos verificables |
| El sistema funciona en producción | El cupo está garantizado solo para un orquestador secuencial |
| Existe un ahorro económico medido | Los costos son simulados bajo supuestos declarados |

### Por qué el olvido no compensa en este dataset

Las señales medidas ofrecen una explicación coherente. El domain classifier alcanza
un AUC de 0,654, por debajo del umbral de 0,75, lo que indica estabilidad de P(X)
durante los 182 días. Las alertas de KS/PSI aparecen solo a partir del día 170. En
ese régimen la ventana corta pierde información sin ganar frescura útil.

El soporte refuerza la explicación. W30 entrena con 1 291 fraudes frente a los
8 269 de W90. El proxy de tarjeta presenta un 40 % de singletons, de modo que las
features de historial aportan menos de lo que aportarían con entidades recurrentes.

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
presenta rendimientos decrecientes mientras la mayor parte de la zona gris reciba
una acción automática por falta de capacidad.

Los riesgos residuales cuentan con controles entregados. Los scores descalibrados
se corrigen con Platt por versión sobre una cola separada. La oscilación de
versiones se limita con cadencia y cooldown de 15 días sin retuning por alertas.
Las etiquetas tardías se gobiernan con `available_at` y predicciones inmutables.
La ausencia de modelo válido produce 503 y pausa. El retuning tras observar el test
queda bloqueado por el hash de prerregistro.

### Conclusión

El sistema reduce el costo observado de 5,40 a 1,53 UM por transacción frente a
aprobar todo, una mejora del 71,7 %. Ese resultado proviene del modelo y de la
política económica, no de la adaptación.

El olvido por ventanas fijas no aporta mejora sobre IEEE-CIS con este protocolo.
Dos afirmaciones resisten el control de robustez. La ventana de 9 días de fit
resulta un 17,3 % más cara que el modelo estático. La referencia expansiva, que
usa todo el histórico, resulta un 2,1 % más barata. Las diferencias de W60 y W90
frente al estático no se distinguen del cero de forma estable.

El factor que explica el ordenamiento es el volumen de entrenamiento, con una
correlación de −0,9916 entre días de fit y costo. A volumen igual la frescura no
produce efecto medible. El diseño del protocolo confunde ambos factores porque cada
ventana reserva 21 días, de modo que reducir W reduce también el training set.

Ninguna estrategia alcanza los objetivos diagnósticos de FPR máximo del 1 % ni de
precisión mínima del 80 % con los umbrales congelados. La capacidad de revisión
resulta más limitante que la calidad del modelo.

La recomendación operativa es conservar el modelo estático con monitoreo activo.
W60 fue la ventana elegida en desarrollo, pero no mejora al estático en el periodo
evaluado y el protocolo no ofrece evidencia de que el olvido aporte valor sobre
este dataset. Una revisión posterior debería evaluar cadencias y retrasos distintos
antes de descartar el enfoque adaptativo, porque la deriva de IEEE-CIS parece
operar a una escala temporal mayor que la del protocolo evaluado.

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

La auditoría crítica del plan y de la implementación está en
`docs/auditoria_critica.md`. Documenta un defecto corregido y cinco decisiones
de diseño que acotan el alcance de las conclusiones.

El protocolo está en `docs/protocolo_experimental.md`, el contrato en
`docs/contrato_sistema.md` y la reproducción en `docs/reproducibilidad.md`. El
despliegue está en `deploy/gcp_runbook.md` y `deploy/promocion_rollback.md`. La
evidencia ejecutable consta de tres notebooks en `notebooks/` y 137 pruebas en
`tests/`.
