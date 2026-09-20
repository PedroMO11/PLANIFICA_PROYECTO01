# Replay local: evidencia de la demo

Generado: 2026-09-20T09:41:43+00:00


## Configuracion

| Elemento | Valor |
|---|---|
| Paquete | models/W30_T165 |
| Version del modelo | W30_T165 |
| Modo | en_proceso |
| Eventos procesados | 5000 |
| Dias cubiertos | 2 |

## Acciones observadas en datos reales

| Accion | n |
|---|---|
| aprobar | 4294 |
| bloquear | 406 |
| revisar | 300 |

Tres acciones presentes en el replay natural: **si**


## Fixtures de contrato (casos sinteticos, NO son transacciones del dataset)

Se incluyen para ejercitar ramas que el replay natural puede no producir. 
No se movieron umbrales ni se presentan como resultados de fraude.


| Caso | p forzada | Monto | Accion obtenida | Esperado |
|---|---|---|---|---|
| aprobar_p_baja | 0.0060 | 25.00 | aprobar | aprobar |
| bloquear_p_alta | 0.8870 | 900.00 | bloquear | bloquear |
| revisar_zona_gris | 0.3930 | 400.00 | revisar | revisar |
| overflow_sin_cupo | 0.3930 | 400.00 | bloquear | automatica |
| monto_cero | 0.3930 | 0.00 | revisar | revisar |

## Cambio de version y rollback (prueba de contrato)

| Paso | Version | p de la fila de prueba |
|---|---|---|
| Activa al inicio | W30_T165 | 0.000055 |
| Tras activar otra | W90_T165 | 0.000102 |
| Tras el rollback | W30_T165 | 0.000055 |

Rollback correcto: **True** · score restaurado: **True**

> Cambio técnico de paquete. NO es una promoción de negocio aprobada: esa requiere pasar el gate sobre H y autorización humana registrada.

## Latencia local (warm)

| Percentil | ms |
|---|---|
| p50 | 45.60 |
| p95 | 66.76 |
| p99 | 71.87 |

Medicion LOCAL. No representa la latencia de una region cloud ni un SLA.


## Idempotencia y cupo

| Comprobacion | Resultado |
|---|---|
| Eventos reenviados | 50 |
| Sin nueva reserva de cupo | 50 |
| Cupo estable tras reenvio | True |
| Prueba aprobada | True |
| Cupo maximo usado en un dia | 150 de 150 |
| Excedio capacidad | False |

## Limitaciones declaradas

- Un unico orquestador secuencial garantiza el cupo; no certifica concurrencia.
- La latencia medida es local y no representa una region cloud.
- Las acciones son simuladas: no hay pagos ni bloqueos reales.