# Contrato del sistema: objetivos, métricas y política de decisión

Documento de referencia para el equipo y para la revisión del curso. Fija qué mide
el sistema, con qué criterio decide y qué **no** puede afirmar. Los valores aquí
son los que el código carga desde `configs/`; cambiar uno invalida el hash de
prerregistro y obliga a repetir la corrida.

---

## 1. Matriz de objetivos O1–O5

Los objetivos vienen del avance (`propuesta_proyecto1_final.md`). Aquí se
convierten en criterios verificables con un artefacto que los sustenta.

| Obj. | Enunciado | Métrica operativa | Criterio | Evidencia |
|---|---|---|---|---|
| **O1** | Menor costo que la política actual y que el modelo estático | Costo observado simulado, UM/transacción | `costo_um_tx` < `costo_aprobar_todo` y < el de S0 | `reports/tables/adaptacion.csv` |
| **O2** | Limitar la caída de PR-AUC frente a una referencia reentrenada | Average Precision por bloque | Caída relativa de AP > 10 % genera alerta **diagnóstica** | `reports/figures/drift_adaptacion.png` |
| **O3** | Respetar la capacidad diaria de revisión | Revisiones admitidas por día | ≤ 150/día, límite duro | `capacidad.*.excedio_capacidad` = `false` |
| **O4** | Señales sin etiqueta que anticipen una caída confirmada | Retraso entre día del evento y día de disponibilidad | KS/PSI y S1 disponibles el mismo día; ADWIN a L=30 días | `runs/<run_id>/drift_log.csv` |
| **O5** | Limitar la disparidad de falsos positivos entre segmentos | Bloqueo de legítimas por grupo | Brecha > 2 pp con ≥ 1 000 legítimas ⇒ alerta | `reports/tables/segmentos.csv` |

**Lo que O1 no afirma.** El costo observado es *simulado* bajo supuestos
declarados (c_FP, c_R, r_H, f_H). No es un ahorro causal medido sobre pagos
reales, y "aprobar todo" es una referencia aritmética, no la política comercial de
un comercio observado.

**Lo que O5 no afirma.** IEEE-CIS no contiene atributos protegidos verificables.
Los segmentos son variables de negocio (dominio de correo, tipo de dispositivo,
cuartil de monto, `addr1`/`addr2`). Esto es **disparidad operativa**, no una
auditoría demográfica.

---

## 2. Las tres familias de métricas

### Técnicas

| Métrica | Definición operativa | Advertencia |
|---|---|---|
| **AP (PR-AUC)** | `average_precision_score`: Σ (Rₙ − Rₙ₋₁)·Pₙ, sin interpolación trapezoidal | Mezclar ambas convenciones cambia el valor varios puntos en datos desbalanceados |
| **Skill** | (AP − prevalencia) / (1 − prevalencia) | Separa un cambio de tasa base de un cambio de calidad |
| **F1 / balanced accuracy** | En un umbral **declarado** (τ_alto de la política) | Nunca en 0.5; el umbral lo fija el costo |
| **Recall @ FPR ≤ 1 %** | Umbral elegido en validación, aplicado congelado | Se reporta el **FPR realmente obtenido**, que puede incumplir el objetivo |
| **Recall @ precisión ≥ 80 %** | Igual, con restricción de precisión | Si ningún umbral la alcanza ⇒ **N/A con motivo**, nunca una cifra inventada |
| **Brier / ECE** | Antes y después de calibrar, por versión | ECE depende de los bins: es complemento, no métrica principal |
| **Latencia p50/p95/p99** | Warm, local, 1 000 peticiones tras 100 de calentamiento | Medición local; no representa una región cloud ni un SLA |

### De decisión

```
C_obs = [ Σ monto·1(fraude finalmente aprobado)
        + c_FP · N(legítimas finalmente bloqueadas)
        + c_R  · N(revisadas) ] / N
```

Se reportan además: monto de fraude evitado, costo de falsos positivos, retorno
por analista-hora, revisiones/día, demanda excedente y utilización del cupo.

### Sociales

Bloqueo de clientes legítimos (distinto del FPR del clasificador: incluye las
legítimas que el analista bloquea por error), disparidad por segmento con soporte
e intervalos, y cobertura automática. Las apelaciones **no** son medibles en
IEEE-CIS: quedan como especificación futura, no como métrica experimental.

---

## 3. Modelo económico de decisión

Para probabilidad calibrada `p` y monto `m`:

```
E[aprobar]  = p · m
E[bloquear] = (1 − p) · c_FP
E[revisar]  = c_R + p·(1 − r_H)·m + (1 − p)·f_H·c_FP
```

| Parámetro | Valor | Naturaleza |
|---|---|---|
| `c_FP` | 5 UM | Supuesto cerrado |
| `c_R` | 1 UM | Supuesto cerrado |
| `r_H` | 0.90 | Recall del analista simulado |
| `f_H` | 0.02 | Bloqueo erróneo del analista simulado |
| Capacidad | 150 revisiones/día | Límite duro (2 × 8 h × 10 casos/h = 160; 10 de holgura declarada) |

UM es consistente con `TransactionAmt`; **no se convierte a PEN ni USD**.

Sensibilidad económica sobre predicciones ya guardadas, sin reajustar nada:
`(1, 0.5)`, `(5, 1)`, `(10, 2)`. Estrés del analista: `r_H=0.80`, `f_H=0.05`.

**Revisar no siempre gana.** Con montos bajos, el costo fijo `c_R` supera la
pérdida esperada: la política prefiere una acción automática. Esa es la razón de
que exista una zona gris y no un umbral único.

---

## 4. Política de tres acciones y cola

Dos umbrales globales `(τ_bajo, τ_alto)` elegidos **por costo** sobre la reserva
`[76, 83)` y congelados antes del test. La rejilla recorre pares de cuantiles del
score, incluidos pares iguales (sin zona gris) y los bordes de aprobar/bloquear
todo. Desempates, en orden: menor bloqueo de legítimas, menor demanda de revisión,
zona gris más estrecha.

**Causalidad de la cola (C12).** El cupo se reserva al **admitir**, de forma
irrevocable y por `event_id`. No se ordena el día completo por `p × monto` para
quedarse con los 150 mejores: eso exigiría conocer transacciones que aún no
ocurrieron. La prioridad ordena el **servicio** entre los ya admitidos. Agotado el
cupo, el caso cae a la más barata entre aprobar y bloquear.

Un top-150 retrospectivo solo puede figurar como **cota oracle** en una tabla
aparte, nunca como política.

---

## 5. Autonomía y supervisión humana

| Automático dentro del experimento | Requiere autorización humana |
|---|---|
| Adquisición, preparación y features | Cada reentrenamiento |
| Scoring y aplicación de la política | Cada promoción de versión |
| Ledger, cupo y registro de decisiones | Cualquier despliegue remoto |
| Métricas, monitores y alertas | Activación inicial del sistema |

**No existe la cadena alerta → deploy.** Una señal genera, como máximo, una
recomendación dirigida a una persona. La corrida offline opera bajo un manifest de
autorización previa que enumera sus tareas, y sus promociones quedan rotuladas
`simuladas`: la aprobación instantánea de un script no representa el tiempo ni el
juicio de una revisión humana real.

**Contingencia (C21).** Si no hay ninguna versión válida, el sistema responde
`model_unavailable` (HTTP 503) y pausa. No decide el pago por defecto y **no**
degrada a S0 o E15, que son referencias no desplegables.

---

## 6. Límites de autonomía heredados del avance

El sistema no cierra cuentas, no mantiene listas negras, no bloquea sin
apelación, no emite explicaciones causales y no se reentrena de forma autónoma.
Toda aprobación o bloqueo en esta entrega es **simulado**: no hay pagos reales.
