# Replay local: evidencia de la demo

Generado: 2026-09-20T18:02:29+00:00


## Configuracion

| Elemento | Valor |
|---|---|
| Paquete | models/W60_T165 |
| Version del modelo | W60_T165 |
| Modo | en_proceso |
| Eventos procesados | 5000 |
| Dias cubiertos | 2 |

## Acciones observadas en datos reales

| Accion | n |
|---|---|
| aprobar | 4678 |
| revisar | 201 |
| bloquear | 121 |

Tres acciones presentes en el replay natural: **si**


## Fixtures de contrato (casos sinteticos, fuera del dataset)

Ejercitan ramas que el replay sobre datos reales puede no producir.
Se evaluan con la misma politica y no forman parte de los resultados.


| Caso | p forzada | Monto | Accion obtenida | Esperado |
|---|---|---|---|---|
| aprobar_p_baja | 0.2036 | 25.00 | aprobar | aprobar |
| bloquear_p_alta | 0.5683 | 900.00 | bloquear | bloquear |
| revisar_zona_intermedia | 0.1342 | 400.00 | revisar | revisar |
| overflow_sin_cupo | 0.1342 | 400.00 | bloquear | automatica |
| monto_cero | 0.1342 | 0.00 | aprobar | aprobar |

## Cambio de version y rollback (prueba de contrato)

| Paso | Version | p de la fila de prueba |
|---|---|---|
| Activa al inicio | W60_T165 | 0.029034 |
| Tras activar otra | W90_T165 | 0.023002 |
| Tras el rollback | W60_T165 | 0.029034 |

Rollback correcto: **True** · score restaurado: **True**

> Cambio técnico de paquete. Una promoción de negocio requiere además pasar el gate sobre H y autorización humana registrada.

## Latencia local (warm)

| Percentil | ms |
|---|---|
| p50 | 46.20 |
| p95 | 74.59 |
| p99 | 87.00 |

Medicion local en proceso; la latencia sobre HTTP esta en `reports/latencia_http.json`.


## Idempotencia y cupo

| Comprobacion | Resultado |
|---|---|
| Eventos reenviados | 50 |
| Sin nueva reserva de cupo | 50 |
| Cupo estable tras reenvio | True |
| Prueba aprobada | True |
| Cupo maximo usado en un dia | 135 de 150 |
| Excedio capacidad | False |

## Limitaciones declaradas

- Un unico orquestador secuencial garantiza el cupo; no certifica concurrencia.
- La latencia medida es local y no representa una region cloud.
- Las acciones son simuladas: no hay pagos ni bloqueos reales.