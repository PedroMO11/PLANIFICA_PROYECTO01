# Política de promoción y rollback

**Escrito para:** el responsable que autoriza cambios de versión (rol A) y quien
ejecuta el despliegue (rol E).

Regla que gobierna todo el documento: **ninguna señal automática promueve ni
despliega**. El sistema produce evidencia y una recomendación; la decisión es de
una persona y queda registrada.

---

## 1. Dos tipos de cambio, dos reglas distintas

No se deben confundir. Uno se decide en minutos; el otro no puede decidirse antes
de 30 días.

| | **Rollback técnico** | **Cambio por deterioro de negocio** |
|---|---|---|
| Qué lo dispara | Errores HTTP o latencia | Costo observado, AP, Brier |
| Cuándo se observa | Al instante | Solo con etiquetas maduras (L=30 días) |
| Quién decide | Regla preautorizada, ejecución automática permitida | Revisión humana explícita, siempre |
| Acción | Volver al paquete anterior | Evaluar, documentar y decidir |

Motivo de la separación: un error 500 es un hecho verificable ahora. Una subida
del costo observado hoy habla de decisiones tomadas hace un mes, y actuar sobre
ella de inmediato sería reaccionar a ruido.

---

## 2. Gates de promoción

Se evalúan sobre **H = [c−7, c)**: el tramo más reciente ya maduro, posterior al
predictor, al calibrador y a la reserva de política.

### Gate 0 — Integridad (bloqueante)

- Los cuatro roles no comparten ningún `TransactionID`.
- H no fue visto por el ajuste, la calibración ni la política **del challenger ni
  del champion**.
- Soporte mínimo: fit ≥ 200 fraudes / 2 000 legítimas; calibración y H ≥ 50 / 500.
- Artefactos completos y esquema de features compatible.
- El cupo nunca superó 150/día en la evaluación.

Una falla aquí **bloquea la versión**. No se negocia.

### Gate 1 — Arranque (solo la primera activación)

Costo en H ≤ costo de "aprobar todo". Si no lo supera, no hay razón económica para
activar el sistema.

### Gate 2 — No inferioridad (promociones posteriores)

| Criterio | Umbral |
|---|---|
| Costo en H | ≤ 1.01 × costo del champion |
| ΔAP | ≥ −0.01 absoluto |
| ΔBrier | ≤ +0.005 |

Es una regla de **no inferioridad tolerante a ruido**, no de superioridad. Exigir
una mejora en cada ciclo llevaría a no actualizar nunca bajo ruido, o a ir
ajustando la tolerancia hasta que pase. El precio declarado: se admite hasta un 1 %
de costo mayor en validación.

### Gate 3 — Social

En segmentos con soporte suficiente (≥ 1 000 legítimas), el bloqueo de legítimas
no puede crecer más de 2 pp frente al champion sobre el mismo H. Si crece, se
genera alerta y **revisión humana explícita**.

### Gate 4 — Humano (siempre)

Aunque los cuatro anteriores pasen, la promoción exige autorización registrada
con identidad, fecha, versión, hash del paquete y la evidencia consultada.

---

## 3. Registro de una autorización

```json
{
  "tipo": "promocion",
  "version_nueva": "W30_T165",
  "version_anterior": "W30_T150",
  "package_hash": "<sha256>",
  "autorizado_por": "<nombre del responsable>",
  "fecha": "<ISO-8601>",
  "evidencia_consultada": [
    "runs/<run_id>/gates.csv",
    "reports/tables/adaptacion.csv",
    "reports/tables/segmentos.csv"
  ],
  "gates": {"integridad": true, "no_inferioridad": true, "social": true},
  "naturaleza": "real | simulada"
}
```

**`naturaleza` importa.** Las promociones dentro del benchmark son `simulada`:
solo habilitan el replay académico, nunca una activación remota. La aprobación
instantánea de un script no representa el tiempo ni el juicio de una revisión
humana real.

---

## 4. Rollback técnico

Regla preautorizada, evaluada por lotes de 100 peticiones:

| Condición | Umbral |
|---|---|
| Tasa de error HTTP | > 1 % |
| Latencia p95 | > 300 ms |
| Lotes consecutivos en falla | 2 |

Al cumplirse: pausar, alertar y volver al paquete anterior. Se conservan siempre
**dos** paquetes elegibles (activo y anterior), cada uno con su propio
preprocesador, calibrador, política y esquema. Un paquete nunca se mezcla con
piezas de otro.

Un error de esquema implica **rechazo de la activación**, no rollback: la versión
nunca llega a recibir tráfico.

---

## 5. Contingencia sin versión válida (C21)

Si ninguna versión de ventana deslizante es válida:

1. El servicio responde `model_unavailable` (HTTP 503).
2. El replay se pausa.
3. **No** se decide el pago por defecto ni se envían todos los casos a revisión.
4. **No** se degrada a S0 ni a E15: son referencias no desplegables por diseño.

Si ya existe un champion válido, conservarlo es la contingencia ante un ajuste o
una promoción fallidos. El rollback solo puede ir a otra ventana válida.

Un gate fallido puede impedir la demo. Eso se documenta como resultado, no se
resuelve inventando un éxito.

---

## 6. Cadencia y cooldown

- Reentrenamiento cada **15 días** de evento, con autorización por acto.
- Cooldown de promoción: **15 días**, salvo rollback técnico.
- Las alertas **no** disparan retuning: recomiendan revisión humana.

---

## 7. Qué no puede concluirse de la demo

- Que el sistema funcione en producción distribuida: el cupo solo está garantizado
  para un orquestador secuencial.
- Que la latencia medida represente una región cloud.
- Que exista un ahorro causal: los costos son simulados bajo supuestos declarados.
- Que el analista simulado represente a un analista real.
