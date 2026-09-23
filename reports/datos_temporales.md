# Datos temporales: integracion, EDA y controles

> Origen de los datos: IEEE-CIS Fraud Detection, particion `train` unicamente.


Generado: 2026-09-20T18:01:34+00:00


## 1. Fuentes e integridad

| Elemento | Valor |
|---|---|
| Transacciones | 590540 filas, 394 columnas |
| Identidad | 144233 filas, 41 columnas |
| Prevalencia global | 0.0350 |
| Dias cubiertos | 182 |
| Cobertura de identidad | 0.244 |

### Auditoria del join

| Comprobacion | Resultado |
|---|---|
| filas_antes | 590540 |
| filas_despues | 590540 |
| filas_invariantes | True |
| ids_identidad | 144233 |
| ids_identidad_huerfanos | 0 |
| cobertura_identidad | 0.2442391709283029 |
| columnas_identidad | 40 |
| nota | La ausencia de identidad es informativa (has_identity), no se imputa ni se descarta la fila. |

El join es un left join uno a uno validado, con el numero de filas invariante.
La ausencia de identidad se conserva como `has_identity`, porque descartar esas filas
sesgaria el panel hacia los clientes con dispositivo identificado.


## 2. Relojes derivados

`TransactionDT` es un desfase en segundos desde un origen desconocido.
De el se derivan `dia`, `semana` y una `hora` relativa, que se usa solo como ciclo
de 24 horas y no identifica la hora local ni el dia laboral.


## 3. Perfil semanal

| Semana | n | Fraudes | Prevalencia | IC 95% Wilson | Monto mediano |
|---|---|---|---|---|---|
| 0 | 27596 | 804 | 0.0291 | [0.0272, 0.0312] | 68.95 |
| 1 | 28463 | 717 | 0.0252 | [0.0234, 0.0271] | 69.31 |
| 2 | 35701 | 869 | 0.0243 | [0.0228, 0.0260] | 82.95 |
| 3 | 34906 | 724 | 0.0207 | [0.0193, 0.0223] | 75.00 |
| 4 | 24539 | 889 | 0.0362 | [0.0340, 0.0386] | 65.00 |
| 5 | 20919 | 803 | 0.0384 | [0.0359, 0.0411] | 61.80 |
| 6 | 20564 | 823 | 0.0400 | [0.0374, 0.0428] | 65.00 |
| 7 | 19761 | 862 | 0.0436 | [0.0409, 0.0466] | 59.00 |
| 8 | 21432 | 927 | 0.0433 | [0.0406, 0.0461] | 63.13 |
| 9 | 22534 | 996 | 0.0442 | [0.0416, 0.0470] | 67.07 |
| 10 | 20953 | 880 | 0.0420 | [0.0394, 0.0448] | 59.00 |
| 11 | 20175 | 676 | 0.0335 | [0.0311, 0.0361] | 59.58 |
| 12 | 22391 | 912 | 0.0407 | [0.0382, 0.0434] | 77.95 |
| 13 | 27429 | 884 | 0.0322 | [0.0302, 0.0344] | 77.95 |
| 14 | 21269 | 718 | 0.0338 | [0.0314, 0.0363] | 68.50 |
| 15 | 20891 | 866 | 0.0415 | [0.0388, 0.0442] | 67.95 |
| 16 | 21078 | 1069 | 0.0507 | [0.0478, 0.0538] | 65.98 |
| 17 | 23575 | 833 | 0.0353 | [0.0331, 0.0378] | 76.95 |
| 18 | 19258 | 727 | 0.0378 | [0.0351, 0.0405] | 62.95 |
| 19 | 18572 | 611 | 0.0329 | [0.0304, 0.0356] | 59.51 |
| 20 | 18549 | 639 | 0.0344 | [0.0319, 0.0372] | 67.95 |
| 21 | 21443 | 660 | 0.0308 | [0.0286, 0.0332] | 72.95 |
| 22 | 19811 | 565 | 0.0285 | [0.0263, 0.0309] | 67.95 |
| 23 | 21038 | 729 | 0.0347 | [0.0323, 0.0372] | 67.95 |
| 24 | 19941 | 733 | 0.0368 | [0.0342, 0.0395] | 70.70 |
| 25 | 17752 | 747 | 0.0421 | [0.0392, 0.0451] | 68.50 |

## 4. Faltantes por familia de columnas

| Familia | Columnas | Faltante medio | Min | Max | >95% faltante |
|---|---|---|---|---|---|
| C | 14 | 0.000 | 0.000 | 0.000 | 0 |
| D | 15 | 0.582 | 0.002 | 0.934 | 0 |
| M | 9 | 0.499 | 0.287 | 0.593 | 0 |
| V | 339 | 0.430 | 0.000 | 0.861 | 0 |
| addr | 2 | 0.111 | 0.111 | 0.111 | 0 |
| card | 20 | 0.077 | 0.000 | 0.754 | 0 |
| id_ | 38 | 0.848 | 0.756 | 0.992 | 9 |
| otras | 52 | 0.376 | 0.000 | 0.936 | 0 |

Las columnas con mas de 95 % de faltantes se descartan dentro de cada fit, segun
el tramo de entrenamiento de la version: una columna vacia en ese tramo y poblada
despues no aporta a esa version.


## 5. Anomalias temporales

```json
{
  "dias_cubiertos": 182,
  "dias_faltantes": [],
  "n_dias_faltantes": 0,
  "volumen_diario_mediano": 3049.5,
  "dias_volumen_bajo": [],
  "dias_volumen_alto": [],
  "monto_p99": 1104.0,
  "n_montos_extremos": 2,
  "n_duplicados_exactos": 111677,
  "prevalencia_global": 0.03499000914417313
}
```


Estas comprobaciones se revisan antes de atribuir una alerta a concept drift,
porque un hueco de captura o un pico de duplicados tambien pueden producirla.


## 6. Proxies de entidad

| Proxy | Entidades | Cobertura | Eventos/entidad | Singletons |
|---|---|---|---|---|
| card | 43018 | 1.000 | 13.73 | 0.404 |
| cliente | 217850 | 1.000 | 2.71 | 0.575 |
| device | 1943 | 0.244 | 74.23 | 0.260 |

Un proxy agrupa comportamiento: dos clientes pueden compartir una clave y un
cliente puede aparecer con varias. Por eso se reporta su calidad y el informe habla
de proxies.


## 7. Causalidad de las features

Ventanas de historial: {'1h': 3600, '24h': 86400, '7d': 604800}. Grupos de empate temporal: 16741.


Las features se emiten antes de actualizar el estado, y los eventos con el mismo
`TransactionDT` leen todos el mismo pasado. Esto se verifica en
`tests/test_point_in_time_features.py`: permutar los IDs empatados deja las features
identicas, y un evento futuro de monto extremo no altera ninguna fila anterior.
