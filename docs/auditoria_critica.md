# Auditoría crítica de la implementación y del plan

El plan de implementación fue redactado por un modelo de lenguaje. Este documento
verifica que la implementación hace lo que afirma y examina si las decisiones de
diseño del plan fueron las adecuadas.

Fecha de la auditoría: 20 de septiembre de 2026. Corrida auditada: `ieee_real`
sobre IEEE-CIS, 590 540 transacciones.

---

## 1. Resumen

De 13 comprobaciones, 7 confirman que la implementación es correcta, 1 identificó
un defecto ya corregido y 5 señalan decisiones del plan que merecen reserva.

| Hallazgo | Tipo | Estado |
|---|---|---|
| H1. Estrategia equivocada en los cortes sociales | Defecto de implementación | Corregido |
| H2. El diseño confunde volumen de entrenamiento con frescura | Diseño del plan | Documentado y medido |
| H3. Una afirmación del informe no era robusta | Interpretación | Corregida |
| H4. La reserva de política desperdicia 7 días por actualización | Diseño del plan | Documentado |
| H5. Los parámetros económicos producen un punto de operación implausible | Diseño del plan | Documentado |
| H6. La zona gris opera con 90 % de overflow | Diseño del plan | Documentado |
| H7 a H13. Métricas, causalidad, gates, sorteo del analista | Verificación | Correctos |

---

## 2. Defecto encontrado y corregido

### H1. Los cortes sociales se calculaban sobre la estrategia equivocada

`_preferred_strategy` devolvía siempre W30, la ventana de menor tamaño, en lugar
de la ventana elegida en desarrollo. El informe describía por tanto la equidad
operativa y la sensibilidad económica de W30 mientras la recomendación era W60.

El defecto pasó desapercibido sobre el dataset sustituto porque allí W30 era
también la ventana elegida. Con IEEE-CIS la elección cambió a W60 y la
inconsistencia quedó expuesta.

Tras la corrección los segmentos con alerta de brecha pasan de 6 a 7 y la tabla de
sensibilidad cambia sus conteos de bloqueos.

---

## 3. Decisiones del plan que merecen reserva

### H2. El experimento confunde volumen de entrenamiento con frescura

Este es el hallazgo de mayor alcance.

El plan define cada ventana W de forma que incluya el predictor más tres reservas
de 7 días. El predictor entrena por tanto con `W - 21` días. Variar W modifica a la
vez cuánto entrena el modelo y cuán reciente es su información, de modo que el
experimento no permite atribuir una diferencia de costo a ninguno de los dos
factores por separado.

El propio diseño contiene dos comparaciones controladas que sí los separan.

| Comparación | Qué aísla | Efecto | IC 95 % | Veredicto |
|---|---|---|---|---|
| W90 frente a S0, ambos con 69 días de fit | Frescura a volumen igual | −0,0052 | [−0,037 y +0,026] | Sin efecto |
| W30 frente a W90, ambas deslizantes | Volumen a frescura igual | +0,2695 | [+0,186 y +0,364] | Efecto grande |

La correlación entre días de fit y costo sobre las cuatro estrategias de tamaño
fijo es de −0,9916.

La conclusión correcta no es que el olvido no compense, sino que el volumen de
entrenamiento domina y que la frescura no produce efecto medible a volumen igual.
El informe se corrigió para reflejarlo.

**Cómo debió diseñarse.** Un experimento que aislara la frescura mantendría
constante el número de días de fit y variaría solo el punto de corte. Por ejemplo,
tres estrategias con 39 días de fit tomados de `[c-60, c-21)`, `[c-81, c-42)` y
`[0, 39)`. El plan no lo contempla.

### H4. La reserva de política se desperdicia después del arranque

El plan reserva `[c-14, c-7)` para elegir umbrales. Esa elección ocurre una sola
vez, en el primer corte. En los cortes posteriores la reserva queda excluida del
fit y no se usa para nada.

| Corte | Predictor de W30 | Reserva de política | Utilizada |
|---|---|---|---|
| T = 120 | `[60,69)`, 9 días | `[76,83)` | Sí |
| T = 135 | `[75,84)`, 9 días | `[91,98)` | No |
| T = 150 | `[90,99)`, 9 días | `[106,113)` | No |
| T = 165 | `[105,114)`, 9 días | `[121,128)` | No |

El plan justifica la reserva permanente porque mantenerla evita cambiar la W
efectiva entre cortes. El argumento es válido para la comparabilidad, pero el costo
es alto. W30 entrena con 9 días en lugar de 16, un 44 % menos de datos, y H2
demuestra que el volumen es el factor dominante.

La decisión perjudica de forma desproporcionada a las ventanas cortas, que son
precisamente las que el experimento pretende evaluar.

### H5. Los parámetros económicos producen un punto de operación implausible

El plan fija `c_FP = 5 UM` sin justificarlo contra la distribución de montos. La
política bloquea cuando `(1-p) * c_FP < p * monto`, es decir cuando
`p > c_FP / (c_FP + monto)`.

| Monto | Umbral de bloqueo |
|---|---|
| 20 | p > 0,2000 |
| 68 (mediana) | p > 0,0678 |
| 200 | p > 0,0244 |
| 1 000 | p > 0,0050 |

Con el monto mediano basta una probabilidad del 6,8 % para que bloquear resulte más
barato que aprobar. El resultado es que el sistema bloquea al 10,79 % de los
clientes legítimos.

Ningún negocio de pagos aceptaría bloquear a uno de cada nueve clientes legítimos.
El costo real de un falso positivo incluye abandono del cliente, carga de soporte y
reputación, que no se reducen a 5 unidades de la moneda de la transacción.

El escenario alto de la sensibilidad, con `c_FP = 10`, reduce los bloqueos de
24 427 a 15 819 y sigue siendo agresivo. Un análisis serio exploraría valores de
`c_FP` del orden del monto mediano o superiores.

### H6. La zona gris opera con 90 % de overflow

El plan fija de forma independiente la capacidad de revisión en 150 casos diarios
y los umbrales por minimización de costo. La interacción entre ambos no se examinó.

| Estrategia | Demanda de revisión | Admitidas | Proporción |
|---|---|---|---|
| S0 | 89 602 | 9 300 | 10,4 % |
| W60 | 97 798 | 9 300 | 9,5 % |
| W30 | 106 411 | 9 300 | 8,7 % |

Más del 90 % de los casos que la política envía a revisión terminan resueltos por
la acción automática más barata. La maquinaria de dos umbrales y cola priorizada
gobierna menos del 10 % de las decisiones que pretende gobernar.

La composición del costo lo confirma. El 60 a 63 % corresponde a fraude aprobado,
el 34 a 36 % a falsos positivos y solo el 3 % al costo de las revisiones.

El sistema se comporta, en la práctica, como un clasificador binario con un
presupuesto marginal de revisión. Esto no invalida los resultados, pero sí acota
qué está midiendo realmente el experimento.

### H3. Una afirmación del informe no era robusta

La versión anterior del informe declaraba que W60 resulta significativamente más
cara que S0, con un intervalo de [+0,009 y +0,196] que excluye el cero.

Ese intervalo se obtuvo con una semilla y un tamaño de bloque concretos. Repetido
con 6 semillas y 3 tamaños de bloque, el intervalo cruza el cero en 12 de 18
combinaciones.

| Diferencia | Efecto | Cruzan cero | Veredicto |
|---|---|---|---|
| W30 frente a S0 | +0,2642 | 0 de 18 | Concluyente |
| E15 frente a S0 | −0,0322 | 0 de 18 | Concluyente |
| W60 frente a S0 | +0,0980 | 12 de 18 | No concluyente |
| W90 frente a S0 | −0,0052 | 18 de 18 | Sin efecto |

El informe se corrigió y ahora declara concluyentes únicamente las dos primeras.
`reports/tables/robustez.csv` contiene el detalle y `robustness_check` lo regenera.

---

## 4. Comprobaciones que resultaron correctas

### H7. Las métricas coinciden con una implementación independiente

El Average Precision reportado coincide con `sklearn.metrics.average_precision_score`
con una diferencia de 5,55e−17 para S0 y para W60. El costo por transacción
recalculado a mano coincide con el reportado.

### H8. El sorteo del analista simulado es uniforme

Sobre 200 000 identificadores, la media es 0,50006 frente a 0,5 esperado y la
desviación 0,28866 frente a 0,2887. La prueba de Kolmogorov-Smirnov contra la
uniforme da D = 0,00031 con p = 1,0. La autocorrelación de orden 1 es 0,039.

### H9. Los gates de promoción funcionaron correctamente

En T = 135 las cinco estrategias fueron bloqueadas por el gate social. Las brechas
medidas estaban entre 2,76 y 4,69 puntos porcentuales, todas por encima del umbral
de 2. En T = 150 y T = 165 las brechas caen a un rango de 0,18 a 1,22 y las
promociones proceden. El gate discriminó, no bloqueó de forma sistemática.

W30 falló además el gate de no inferioridad de costo en T = 150, de modo que operó
con un modelo antiguo durante buena parte del test. El comportamiento es el
esperado.

### H10. La reproducibilidad es exacta

Dos corridas independientes de la cadena completa produjeron resultados idénticos
bit a bit en los 18 fits de tuning, en el hash de prerregistro y en el costo de las
cinco estrategias. `fraud-adaptive verify` lo comprueba.

### H11. Los controles de causalidad funcionan

Las 137 pruebas cubren los mecanismos de leakage. Las verificaciones decisivas
alteran las filas futuras, invierten las etiquetas inmaduras y permutan los
identificadores de eventos con el mismo timestamp, comprobando en cada caso que la
salida no cambia.

### H12. La corrección de `job_time` era necesaria

La implementación inicial comparaba la madurez de la etiqueta contra `cutoff` en
lugar de contra `job_time`. Con ese criterio las colas de calibración, política y
validación quedaban vacías. El plan especifica el criterio correcto en el texto,
de modo que el defecto era de implementación y no de diseño.

### H13. La selección de ventana en desarrollo faltaba

El plan la exige en F3 paso 5 y en C21. No estaba implementada. Sobre el sustituto
no se notó porque W30 ganaba en desarrollo y en test. Con IEEE-CIS la elección
cambia a W60 y la ausencia habría dejado el trabajo sin recomendación defendible.
Se implementó en `select_deployable_window`.

---

## 5. Qué habría convenido hacer distinto

Las siguientes observaciones son sobre el plan, no sobre la ejecución.

**Separar volumen de frescura desde el diseño.** Es la carencia más importante.
Fijar el número de días de fit y variar solo el punto de corte habría permitido
responder la pregunta que el trabajo dice responder.

**Liberar la reserva de política tras el arranque.** Los 7 días reservados en cada
corte posterior no cumplen ninguna función y penalizan a las ventanas cortas.

**Justificar `c_FP` contra la distribución de montos.** El valor de 5 UM produce un
régimen de bloqueo que ningún negocio aceptaría. Convenía derivarlo de un objetivo
operativo, por ejemplo una tasa máxima de bloqueo de legítimas.

**Comprobar la interacción entre capacidad y umbrales.** Con 150 revisiones diarias
frente a una demanda superior a 1 000, la zona gris resulta casi inerte.

**Incluir el control de robustez desde el principio.** Un intervalo con una sola
semilla no basta para declarar una diferencia concluyente. El control es barato y
habría evitado una afirmación incorrecta en el informe.

**Revisar el estatus de E15.** El plan la declara referencia no desplegable, pero
es la estrategia con mejor costo y es perfectamente desplegable, porque consiste en
reentrenar sobre todo el histórico cada 15 días. La exclusión responde a que el
enunciado prioriza ventanas deslizantes, no a un argumento técnico.

---

## 6. Qué se sostiene tras la auditoría

La infraestructura resulta sólida. El protocolo temporal, los controles de leakage,
la calibración, los gates y la reproducibilidad funcionan como se afirma y están
verificados.

El resultado principal se sostiene con la precisión corregida. El olvido por
ventanas fijas no mejora el costo sobre IEEE-CIS. Lo que el experimento demuestra
de forma concluyente es que reducir el training set perjudica. Lo que no logra
medir es un efecto de la frescura, porque a volumen igual la diferencia no se
distingue del cero.

Las reservas de esta auditoría no invalidan el trabajo. Acotan qué preguntas
responde y cuáles quedan abiertas.
