# Contrato del sistema: objetivos, métricas y política de decisión

Este documento fija qué mide el sistema, con qué criterio decide y cuáles son los
límites de sus conclusiones. Los valores se cargan desde `configs/`; cambiar uno
modifica el hash de prerregistro y obliga a repetir la corrida.

## 1. Objetivos O1–O5

Los objetivos provienen de la propuesta del avance
(`docs/antecedentes/propuesta_avance.md`) y aquí se expresan como criterios
verificables.

| Obj. | Enunciado | Métrica | Criterio | Resultado en la corrida v3 |
|---|---|---|---|---|
| O1 | Menor costo que la política actual y que el modelo estático | Costo observado simulado, UM/tx | Inferior al de aprobar todo y al de S0 | E15: 1,984 frente a 5,398 y 2,092. La ventaja sobre S0 tiene un intervalo que incluye el cero |
| O2 | Limitar la caída de AP | AP por bloque | Una caída relativa superior a 10 % genera alerta diagnóstica | AP de S0: 0,526 en B1 y 0,514 en B4 |
| O3 | Respetar la capacidad de revisión | Revisiones admitidas por día | Como máximo 150 | Se cumple en las siete estrategias |
| O4 | Señales sin etiqueta que anticipen una caída confirmada | Retraso entre el día del evento y el día de disponibilidad | KS/PSI y S1 el mismo día; ADWIN a 30 días | Sin cambio en P(X); ADWIN detecta deriva del error a 30 días |
| O5 | Limitar la disparidad del bloqueo de legítimas | Bloqueo de legítimas por segmento | Brecha superior a 2 pp con al menos 1 000 legítimas | 2 de 14 segmentos con soporte superan el umbral (W60) |

El costo observado es simulado con los parámetros de la sección 3, y «aprobar todo»
es una referencia aritmética. IEEE-CIS no contiene atributos protegidos, así que
los segmentos son variables de negocio (dominio de correo, tipo de dispositivo,
cuartil de monto, `addr1` y `addr2`) y O5 mide disparidad operativa.

## 2. Métricas

### Técnicas

| Métrica | Definición | Observación |
|---|---|---|
| AP | `average_precision_score`, suma de (Rₙ − Rₙ₋₁)·Pₙ | Sin interpolación trapezoidal, que cambia el valor en datos desbalanceados |
| Skill | (AP − prevalencia) / (1 − prevalencia) | Separa un cambio de tasa base de un cambio de calidad |
| F1 y balanced accuracy | En el umbral de FPR 1 % fijado en validación | La política no tiene un corte sobre `p`, así que se usa ese umbral declarado |
| Recall a FPR ≤ 1 % | Umbral elegido en validación y aplicado sin reajuste | Se reporta el FPR obtenido en test |
| Recall a precisión ≥ 80 % | Igual, con restricción de precisión | Si ningún umbral alcanza la precisión se reporta N/A con el motivo |
| Brier y ECE | Antes y después de calibrar, por versión | El ECE depende de los bins y se usa como complemento |
| Latencia p50, p95 y p99 | 1 000 peticiones HTTP tras 100 de calentamiento | Medición local |

### De decisión

```
C_obs = [ Σ monto·1(fraude finalmente aprobado)
        + c_FP · N(legítimas finalmente bloqueadas)
        + c_R  · N(revisadas) ] / N
```

Se reportan además el monto de fraude evitado, las revisiones por día, la demanda
que excede el cupo y el uso del cupo.

### Sociales

Tasa de bloqueo de clientes legítimos, que suma los bloqueos automáticos y los
errores del analista; disparidad de esa tasa por segmento con intervalo y soporte
mínimo; y cobertura automática. Las apelaciones no son medibles en IEEE-CIS y
quedan especificadas para una operación real.

## 3. Modelo económico

Para una probabilidad calibrada `p` y un monto `m`:

```
E[aprobar]  = p · m
E[bloquear] = (1 − p) · c_FP
E[revisar]  = c_R + p·(1 − r_H)·m + (1 − p)·f_H·c_FP
```

| Parámetro | Valor | Origen |
|---|---|---|
| `c_FP` | 25 UM | Menor valor que cumple el objetivo de bloqueo de legítimas en desarrollo |
| `c_R` | 1 UM | Supuesto |
| `r_H` | 0,90 | Recall del analista simulado |
| `f_H` | 0,02 | Tasa de bloqueo erróneo del analista simulado |
| Objetivo de bloqueo de legítimas | 1 % | Restricción operativa declarada |
| Capacidad | 150 revisiones/día | Dos analistas, 8 horas y 10 casos por hora dan 160; se dejan 10 de holgura |

La unidad monetaria es la de `TransactionAmt` y no se convierte a otra moneda.

El costo de un falso positivo no es observable, pero la fracción de legítimas
rechazadas sí lo es. Por eso `c_FP` se calibra como el menor valor que cumple el
objetivo del 1 % sobre la reserva de política de desarrollo `[76, 83)`, con la
familia de mayor AP como referencia. Con un valor fijo de 5 UM se bloqueaba el
11,17 % de las legítimas. El valor calibrado queda sellado en el prerregistro y viaja
en el manifiesto del paquete.

La sensibilidad económica recalcula el costo de predicciones guardadas con las
acciones congeladas: `c_FP` a 0,5×, 1× y 2× del calibrado con `c_R` de 0,5, 1 y 2, y
un escenario de analista con `r_H = 0,80` y `f_H = 0,05`.

Con montos bajos el costo fijo de revisar supera la pérdida esperada y la política
prefiere una acción automática; por eso la revisión es una tercera acción y la
decisión no es binaria.

## 4. Política de tres acciones y cola

La acción se decide caso por caso con los costos esperados y el cupo:

```
accion = revisar                      si min(E[aprobar], E[bloquear]) − E[revisar] > λ
         la más barata entre          en otro caso
         aprobar y bloquear
```

`λ` es el precio sombra del cupo, el multiplicador de Lagrange de la restricción de
capacidad. Sin él la regla pedía 704 revisiones diarias para 150 plazas y el cupo se
llenaba por orden de llegada. Se calibra por bisección, sin etiquetas, como el menor
valor que ajusta la demanda al cupo, y resulta 7,86 UM.

El punto de indiferencia entre aprobar y bloquear es `p* = c_FP / (m + c_FP)`, de
modo que un corte global sobre `p` bloquea de más en montos bajos y de menos en
montos altos. Sobre la reserva de política de desarrollo, el mejor par de umbrales
cuesta 1,9696 UM/tx y la regla económica 1,5895, aunque los umbrales se eligieron
minimizando el costo en esa misma reserva. Los umbrales se siguen calculando para
medir esa diferencia y para las métricas diagnósticas; `implied_thresholds` describe
en términos de `p` el umbral que la regla induce para cada monto.

El cupo se reserva al admitir cada caso, de forma irrevocable y por `event_id`.
Ordenar el día completo por `p × monto` exigiría conocer transacciones futuras; la
prioridad ordena solo el servicio entre los casos ya admitidos. Agotado el cupo, el
caso recibe la acción automática más barata. Un top-150 retrospectivo solo puede
aparecer como cota de referencia en una tabla aparte.

## 5. Autonomía y supervisión humana

| Automático | Requiere autorización humana |
|---|---|
| Adquisición, preparación y features | Cada reentrenamiento |
| Scoring y aplicación de la política | Cada promoción de versión |
| Ledger, cupo y registro de decisiones | Cualquier despliegue remoto |
| Métricas, monitores y alertas | Activación inicial del sistema |

Una señal de drift genera una recomendación dirigida a una persona. La corrida
offline opera con un manifiesto de autorización previa que enumera sus tareas, y sus
promociones quedan rotuladas como simuladas.

Si no hay ninguna versión válida, el servicio responde `model_unavailable` (HTTP
503) y la operación se pausa. El sistema no decide pagos por defecto ni recurre a un
modelo de respaldo.

## 6. Decisiones vedadas

El sistema no cierra ni suspende cuentas, no reporta a centrales de riesgo, no
bloquea sin canal de apelación, no se reentrena ni promueve versiones sin
autorización humana y no da explicaciones causales basadas en variables anónimas.
Todas las aprobaciones y bloqueos de esta entrega son simulados.
