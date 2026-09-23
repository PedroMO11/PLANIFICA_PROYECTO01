# Política de promoción y rollback

Guía para quien autoriza cambios de versión y para quien ejecuta el despliegue.

Las señales automáticas producen evidencia y una recomendación. Promover o desplegar
una versión es una decisión de una persona y queda registrada.

## 1. Dos tipos de cambio

| | Rollback técnico | Cambio por deterioro de negocio |
|---|---|---|
| Qué lo dispara | Errores HTTP o latencia | Costo observado, AP o Brier |
| Cuándo se observa | De inmediato | Con etiquetas maduras, 30 días después |
| Quién decide | Regla preautorizada; puede ejecutarse de forma automática | Revisión humana explícita |
| Acción | Volver al paquete anterior | Evaluar, documentar y decidir |

Un error HTTP 500 se verifica en el momento. Un aumento del costo observado hoy
refleja decisiones tomadas hace un mes, y reaccionar de inmediato a esa señal sería
reaccionar a ruido.

## 2. Gates de promoción

Se evalúan sobre H = `[c−7, c)`, el tramo maduro más reciente, posterior al
predictor y al calibrador.

### Integridad (bloqueante)

- Los roles de la versión no comparten ningún `TransactionID`.
- H no participó en el ajuste ni en la calibración de la versión candidata ni de la
  vigente.
- Soporte mínimo: fit con 200 fraudes y 2 000 legítimas; calibración y H con 50 y 500.
- Artefactos completos y esquema de features compatible.
- El cupo no superó 150 casos diarios en la evaluación.

Una falla de integridad bloquea la versión.

### Arranque (primera activación)

El costo en H debe ser inferior al de aprobar todo; si no lo es, activar el sistema
no tiene justificación económica.

### No inferioridad (promociones posteriores)

| Criterio | Umbral |
|---|---|
| Costo en H | ≤ 1,01 × costo de la versión vigente |
| ΔAP | ≥ −0,01 absoluto |
| ΔBrier | ≤ +0,005 |

Exigir una mejora estricta en cada ciclo llevaría a no actualizar casi nunca, o a
relajar la tolerancia hasta que pase. A cambio, se admite hasta 1 % más de costo en
validación.

### Social

En segmentos con al menos 1 000 legítimas, el bloqueo de legítimas no puede crecer
más de 2 pp frente a la versión vigente sobre el mismo H. Si crece, se genera una
alerta y la promoción pasa a revisión humana explícita.

### Autorización humana

Aunque los gates anteriores pasen, la promoción exige una autorización registrada con
responsable, fecha, versión, hash del paquete y evidencia consultada.

## 3. Registro de una autorización

```json
{
  "tipo": "promocion",
  "version_nueva": "E15_T165",
  "version_anterior": "E15_T150",
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

Las promociones del backtest se registran como `simulada` y solo habilitan el replay
académico. Una aprobación automática dentro de un script no reemplaza el tiempo ni el
criterio de una revisión humana.

## 4. Rollback técnico

Regla preautorizada, evaluada por lotes de 100 peticiones:

| Condición | Umbral |
|---|---|
| Tasa de error HTTP | > 1 % |
| Latencia p95 | > 300 ms |
| Lotes consecutivos en falla | 2 |

Cuando se cumple, el sistema se pausa, alerta y vuelve al paquete anterior. Se
conservan dos paquetes elegibles, el activo y el anterior, cada uno con su propio
preprocesador, calibrador, política y esquema; las piezas de paquetes distintos no se
mezclan. Un error de esquema rechaza la activación antes de que la versión reciba
tráfico.

## 5. Sin versión válida

Si ninguna versión es válida:

1. El servicio responde `model_unavailable` (HTTP 503).
2. El replay se pausa.
3. El sistema no decide pagos por defecto ni envía todos los casos a revisión.

Si ya hay una versión vigente válida, se conserva ante un ajuste o una promoción
fallidos. Un gate fallido que impida la demo se documenta como resultado.

## 6. Cadencia y cooldown

- Reentrenamiento cada 15 días de evento, con una autorización por cada uno.
- Cooldown de promoción de 15 días, salvo rollback técnico.
- Las alertas recomiendan revisión humana y no disparan reajustes.

## 7. Alcance de la demo

- El cupo está garantizado para un orquestador secuencial; la demo no prueba una
  operación distribuida.
- La latencia medida es local.
- Los costos son simulados con supuestos declarados.
- El analista es simulado.
