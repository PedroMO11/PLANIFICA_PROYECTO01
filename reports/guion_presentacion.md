# Guion de exposición — 18 minutos + 5–10 de preguntas

**Escrito para:** los cinco integrantes del equipo que expondrán.

**Regla que conviene tener presente todo el rato:** el docente puede pedirle a
cualquiera que explique cualquier componente. El reparto de abajo es de *turnos*,
no de conocimiento. Todos deberían poder defender las cinco piezas.

---

## Reparto y tiempos

| Tiempo | Bloque | Expone |
|---|---|---|
| 0:00–2:00 | El problema de decisión y los compromisos del avance | **A** |
| 2:00–4:30 | Datos, integración y drift observado | **B** |
| 4:30–6:30 | Cortes temporales y por qué no hay fuga | **B/C** |
| 6:30–9:00 | Tres modelos, calibración y costo de decisión | **C** |
| 9:00–12:30 | Ventanas de olvido: el experimento central | **D** |
| 12:30–15:00 | Demo del replay y propuesta de despliegue | **E** (o **D** si son cuatro) |
| 15:00–17:00 | Infografía, equidad operativa y límites | **A/E** |
| 17:00–18:00 | Conclusión cuantitativa y validación futura | **A** |

*Variante para cuatro: D asume el bloque de despliegue con revisión de A.*

---

## 0:00–2:00 · El problema (A)

**Abrir con la frase que ordena todo lo demás:**

> «Un modelo entrenado hoy deja de servir mañana, y no se entera hasta un mes
> después. Nuestro proyecto no es un clasificador de fraude: es un sistema que
> decide bajo esa restricción.»

Tres acciones por transacción — aprobar, revisar, bloquear — bajo cuatro
restricciones: la etiqueta llega a 30 días, la capacidad humana es de 150
revisiones/día, los errores cuestan cosas distintas, y la distribución cambia.

Conectar con el avance: los objetivos O1–O5 se mantienen y aquí se convierten en
criterios verificables.

**Aviso obligatorio, sin esconderlo:** los resultados provienen de un dataset
sustituto sintético; IEEE-CIS exige un token individual de Kaggle. Los comandos son
idénticos con los datos reales.

---

## 2:00–4:30 · Datos y drift (B)

- Dos fuentes, left join uno-a-uno **con el número de filas invariante**.
- La ausencia de identidad (24 % de cobertura) **es información**, no un faltante
  a imputar. Descartar esas filas sesgaría el panel.
- `TransactionDT` es un delta en segundos, **no una fecha**. La hora derivada es
  relativa: no identifica hora local ni día laboral.

**Mostrar** `reports/figures/eda_temporal.png`.

**La observación que vale el bloque:** la prevalencia se mantiene plana en ~3,5 %
durante los seis meses. Lo que cambia no es *cuánto* fraude hay, sino *qué lo
predice*. Por eso no basta con vigilar la tasa de fraude.

---

## 4:30–6:30 · Cortes temporales (B/C)

**El bloque más fácil de perder y el que más rigor demuestra.** Dibujar la línea
temporal en pizarra:

```
0        60   69   76   83   90        120  135  150  165  182
|---------|----|----|----|---------|----|----|----|----|
 desarrollo   fit  cal  pol   H   warmup   B1   B2   B3   B4
```

Cuatro roles por versión, **disjuntos**: predictor → calibrador → política →
validación. Las tres reservas de 7 días son **idénticas** para todas las
estrategias: eso es lo que hace que una diferencia de costo sea atribuible al
tamaño de ventana y no a otra cosa.

**Si preguntan cómo saben que no hay fuga:** no es una afirmación, son 127 pruebas.
La decisiva: se invierten todas las etiquetas que aún no han madurado y se comprueba
que el predictor no cambia **ni un bit**.

---

## 6:30–9:00 · Modelos y decisión (C)

| Familia | AP | Costo (UM/tx) |
|---|---|---|
| LightGBM | 0,5083 | **0,5945** |
| Logística | 0,4547 | 0,6606 |
| Random Forest | 0,2746 | 1,0135 |

**La familia se elige por costo, no por AP.** Son criterios distintos y conviene
decirlo en voz alta.

**Por qué calibrar no es un adorno:** la política compara `p·monto` con
`(1−p)·c_FP`. Esa aritmética solo tiene sentido si `p` es una probabilidad. Un
modelo con `scale_pos_weight` produce puntajes inflados; usarlos como probabilidad
haría bloquear de más **sin que el AP lo notara**. El ECE baja de 0,0046 a 0,0010.

**Los umbrales no son 0,5**: salen de minimizar el costo, y son (0,012 · 0,774).

---

## 9:00–12:30 · El experimento central (D)

**Mostrar** `reports/figures/drift_adaptacion.png`. Cinco estrategias, los mismos
eventos, la misma semilla, la misma política.

| | AP | Costo UM/tx | Δ vs S0 [IC 95 %] |
|---|---|---|---|
| **W30** | 0,2955 | **1,2833** | **−0,676** [−0,999 · −0,424] |
| W60 | 0,2503 | 1,4341 | −0,526 |
| W90 | 0,2247 | 1,4899 | −0,470 |
| E15 | 0,2130 | 1,5477 | −0,412 |
| S0 | 0,1445 | 1,9596 | — |

**El mensaje:** bajo drift, la frescura vale más que el volumen. W30 entrena con
solo 9 días y gana a W90, que usa 69. El orden es monótono, y el intervalo pareado
no cruza el cero.

**Decir lo que no funcionó, sin que lo tengan que preguntar:** ninguna estrategia
alcanzó el FPR ≤ 1 % ni la precisión ≥ 80 % en test con los umbrales congelados.
W30 obtuvo 1,85 % de FPR y 48,9 % de precisión. No se corrigió moviendo el umbral:
eso convertiría un objetivo incumplido en un resultado ajustado a posteriori.

**Los dos relojes, empíricamente:** KS/PSI generó 29 alertas disponibles el mismo
día; ADWIN detectó 2 cambios en S0 y ninguno en las estrategias que se reentrenan,
y su evidencia llega 30 días después. Esa distancia es la respuesta a O4.

---

## 12:30–15:00 · Demo y despliegue (E)

**Ejecutar en vivo** (tener capturas de respaldo):

```bash
python -m fraud_adaptive replay --package models/W30_T165 --max-events 5000 --fixtures
```

Señalar: las tres acciones aparecen en datos reales (4 294 / 406 / 300), el cupo se
respeta (150/150), la idempotencia pasa y la latencia p95 es de 66 ms.

**La distinción que conviene explicar bien:** el servicio *calcula* la decisión; el
replay la *confirma* y reserva el cupo. Un cliente arbitrario podría mentir sobre
el cupo, así que la demo exige un único orquestador confiable. Hacer el cupo
autoritativo en el servicio requeriría estado distribuido, que es el diseño futuro
con Firestore.

**La imagen Docker está construida y probada**, no solo escrita: arranca en verde,
devuelve las mismas probabilidades que el cálculo offline (5/5 idénticas), p95 de
87 ms sobre HTTP, y sin paquete montado responde 503 en vez de decidir el pago.
Lo que queda para el equipo es el `push` al registro y el `deploy`, no el build.

Pub/Sub, BigQuery y Cloud Scheduler están en el diagrama como diseño, **sin
recursos creados**.

---

## 15:00–17:00 · Infografía y límites (A/E)

**Mostrar** `reports/infografia.pdf`: las seis estaciones y el centro — *X es
observable hoy; y se confirma 30 días después*.

Equidad operativa: 13 segmentos con soporte, 0 alertas de brecha. **Decir que no es
una auditoría demográfica**: IEEE-CIS no tiene atributos protegidos verificables.

**El hallazgo operativo más útil:** la demanda de revisión es de 600–1 300
casos/día frente a un cupo de 150. El cuello de botella dominante es la capacidad,
no el modelo. Mejorar el AP tiene rendimientos decrecientes mientras el 90 % de la
zona gris caiga a una acción automática por falta de gente.

---

## 17:00–18:00 · Conclusión (A)

> «Bajo el drift inyectado, olvidar compensa y compensa más cuanto más corta es la
> ventana: W30 reduce el costo un 34,5 % frente al estático y un 64,3 % frente a
> aprobar todo, con un intervalo que no cruza el cero.
>
> Al mismo tiempo, ninguna estrategia alcanzó los objetivos de FPR ni de precisión
> en test, y la capacidad de revisión resultó más limitante que la calidad del
> modelo.
>
> Esto es un backtest, no una validación prospectiva. Un backtest no prueba que
> mañana funcione.»

---

# Hoja de respuestas

Preguntas probables, con la respuesta corta primero.

### «¿Cómo saben que no hay fuga temporal?»

No lo afirmamos: lo probamos. 127 pruebas automáticas. Tres decisivas: (1) alterar
las filas futuras no mueve las medianas ni el vocabulario del preprocesamiento;
(2) invertir todas las etiquetas inmaduras no cambia el predictor ni un bit;
(3) permutar los IDs de eventos con el mismo timestamp deja las features idénticas.

### «¿Por qué W30 gana si entrena con solo 9 días?»

Porque bajo drift la frescura vale más que el volumen. El vector que genera el
fraude rota con el tiempo; un modelo de 69 días promedia regímenes que ya no
existen. El límite es el soporte: W30 tiene 959 fraudes en su fit, por encima del
mínimo de 200. Con menos datos, se marcaría no válida.

### «¿No estarán sobreajustando al elegir W?»

La ventana se elige en `[76,83)`, que es desarrollo, y se congela con un hash antes
de abrir el test. Si alguien cambiara la configuración después, la corrida aborta.
Que W30 también gane en el test es una confirmación, no el criterio.

### «Los costos son inventados, ¿no?»

Sí, y lo decimos en el informe. Son supuestos declarados (c_FP=5, c_R=1). Por eso
hay una sensibilidad de tres escenarios: el costo va de 0,73 a 1,70 UM/tx y **el
ordenamiento entre estrategias no cambia**. Lo que no afirmamos es un ahorro causal.

### «¿Por qué no usan el veredicto del analista para entrenar?»

Porque no es la verdad, es una opinión con recall 0,90. Y porque solo lo conocemos
cuando la etiqueta ya maduró: usarlo antes sería fuga. Se calcula en el evaluador y
nunca cambia una decisión ya emitida.

### «¿Qué pasa con las transacciones que bloquearon? Nunca sabrán si eran fraude.»

Correcto, y es la limitación más seria: *selective labels*. En el benchmark usamos
full-information —toda etiqueta madura aunque la acción fuera bloquear— porque el
dataset la tiene. En producción no sería así, y las etiquetas disponibles estarían
sesgadas por las propias decisiones del sistema. La corrección con *masked-label*
quedó fuera del alcance.

### «¿Por qué 150 revisiones y no más?»

Es la restricción dada, no un parámetro. Y resultó ser el cuello de botella
dominante: la demanda es de 600 a 1 300 casos/día. Ampliar el equipo humano mejoraría
más el costo que mejorar el modelo.

### «¿Esto funcionaría en producción?»

Con tres salvedades: el cupo solo está garantizado para un orquestador secuencial;
la latencia es local y no representa una región cloud; y es un backtest
retrospectivo. Además, la selección previa del dataset ya había mirado periodos
tardíos, y lo declaramos.

### «¿Por qué Platt y no isotónica?»

La cola de calibración son 7 días. La isotónica necesita más datos para no
sobreajustar escalones, y con 50–300 fraudes daría una función escalonada inestable
entre versiones. Platt tiene dos parámetros y degrada de forma predecible. La
elección se congeló antes del test para que no fuera un grado de libertad.

### «El benchmark anterior probaba que hay concept drift, ¿no?»

Es evidencia **consistente** con drift, no una prueba causal. Además, su
preprocesamiento ajustaba categorías sobre toda la ventana, lo cual es una
limitación que arrastramos y declaramos en vez de repararla retroactivamente.

### «¿Y si los datos reales dan otro resultado?»

Es posible y sería un resultado legítimo. La contribución es el método: el
protocolo temporal, los controles de fuga, la política económica y los gates. Los
comandos son idénticos: basta colocar los CSV reales y ejecutar con `--rebuild`.
