# Datos temporales: integracion, EDA y controles

> **Origen de los datos: SUSTITUTO SINTETICO.** Las cifras de este documento NO describen IEEE-CIS. Ver `ieee-fraud-detection/LEEME_DATOS_SUSTITUTOS.txt`.


Generado: 2026-09-20T09:41:33+00:00


## 1. Fuentes e integridad

| Elemento | Valor |
|---|---|
| Transacciones | 581972 filas, 394 columnas |
| Identidad | 139843 filas, 41 columnas |
| Prevalencia global | 0.0352 |
| Dias cubiertos | 182 |
| Cobertura de identidad | 0.240 |

### Auditoria del join

| Comprobacion | Resultado |
|---|---|
| filas_antes | 581972 |
| filas_despues | 581972 |
| filas_invariantes | True |
| ids_identidad | 139843 |
| ids_identidad_huerfanos | 0 |
| cobertura_identidad | 0.24029162915054333 |
| columnas_identidad | 40 |
| nota | La ausencia de identidad es informativa (has_identity), no se imputa ni se descarta la fila. |

El join es un LEFT JOIN uno-a-uno validado: el numero de filas es invariante. 
La ausencia de identidad se conserva como `has_identity`, porque descartar esas filas 
sesgaria el panel completo hacia los clientes con dispositivo identificado.


## 2. Relojes derivados

`TransactionDT` es un DELTA en segundos desde un origen desconocido, no una fecha. 
De el se derivan `dia`, `semana` y una `hora` RELATIVA. La hora se usa solo como ciclo 
de 24 h; no identifica hora local ni dia laboral, y el informe no afirma lo contrario.


## 3. Perfil semanal

| Semana | n | Fraudes | Prevalencia | IC 95% Wilson | Monto mediano |
|---|---|---|---|---|---|
| 0 | 22403 | 852 | 0.0380 | [0.0356, 0.0406] | 36.77 |
| 1 | 22492 | 833 | 0.0370 | [0.0346, 0.0396] | 36.51 |
| 2 | 22314 | 723 | 0.0324 | [0.0302, 0.0348] | 36.52 |
| 3 | 22372 | 738 | 0.0330 | [0.0307, 0.0354] | 36.57 |
| 4 | 22378 | 817 | 0.0365 | [0.0341, 0.0390] | 36.61 |
| 5 | 22380 | 841 | 0.0376 | [0.0352, 0.0402] | 36.50 |
| 6 | 22383 | 734 | 0.0328 | [0.0305, 0.0352] | 37.07 |
| 7 | 22255 | 754 | 0.0339 | [0.0316, 0.0363] | 36.58 |
| 8 | 22337 | 839 | 0.0376 | [0.0351, 0.0401] | 36.19 |
| 9 | 22468 | 840 | 0.0374 | [0.0350, 0.0399] | 36.60 |
| 10 | 22251 | 715 | 0.0321 | [0.0299, 0.0345] | 36.66 |
| 11 | 22364 | 700 | 0.0313 | [0.0291, 0.0337] | 37.79 |
| 12 | 22452 | 883 | 0.0393 | [0.0369, 0.0420] | 37.90 |
| 13 | 22189 | 808 | 0.0364 | [0.0340, 0.0390] | 38.20 |
| 14 | 22447 | 762 | 0.0339 | [0.0317, 0.0364] | 38.78 |
| 15 | 22488 | 736 | 0.0327 | [0.0305, 0.0351] | 40.41 |
| 16 | 22244 | 795 | 0.0357 | [0.0334, 0.0383] | 39.87 |
| 17 | 22152 | 856 | 0.0386 | [0.0362, 0.0413] | 40.69 |
| 18 | 22150 | 736 | 0.0332 | [0.0309, 0.0357] | 42.64 |
| 19 | 22806 | 713 | 0.0313 | [0.0291, 0.0336] | 43.20 |
| 20 | 22215 | 860 | 0.0387 | [0.0363, 0.0413] | 43.70 |
| 21 | 22681 | 829 | 0.0366 | [0.0342, 0.0391] | 44.17 |
| 22 | 22356 | 722 | 0.0323 | [0.0301, 0.0347] | 45.12 |
| 23 | 22493 | 732 | 0.0325 | [0.0303, 0.0349] | 45.67 |
| 24 | 22522 | 853 | 0.0379 | [0.0355, 0.0404] | 45.08 |
| 25 | 22380 | 835 | 0.0373 | [0.0349, 0.0399] | 45.48 |

## 4. Faltantes por familia de columnas

| Familia | Columnas | Faltante medio | Min | Max | >95% faltante |
|---|---|---|---|---|---|
| C | 14 | 0.000 | 0.000 | 0.000 | 0 |
| D | 15 | 0.490 | 0.104 | 0.875 | 0 |
| M | 9 | 0.300 | 0.181 | 0.421 | 0 |
| V | 339 | 0.429 | 0.020 | 0.762 | 0 |
| addr | 2 | 0.000 | 0.000 | 0.000 | 0 |
| card | 20 | 0.076 | 0.000 | 0.959 | 1 |
| id_ | 38 | 0.803 | 0.779 | 0.813 | 0 |
| otras | 36 | 0.428 | 0.000 | 0.930 | 0 |

Las columnas con mas de 95%% de faltantes se descartan DENTRO de cada fit, 
nunca globalmente: una columna puede estar vacia en el tramo de entrenamiento de una 
version y poblada despues, y esa version no pudo aprender de ella.


## 5. Anomalias temporales

```json
{
  "dias_cubiertos": 182,
  "dias_faltantes": [],
  "n_dias_faltantes": 0,
  "volumen_diario_mediano": 3187.5,
  "dias_volumen_bajo": [],
  "dias_volumen_alto": [],
  "monto_p99": 461.31929,
  "n_montos_extremos": 3,
  "n_duplicados_exactos": 2,
  "prevalencia_global": 0.03523537214848824
}
```


Estas comprobaciones se hacen ANTES de atribuir cualquier alerta a concept drift: 
un hueco de captura o un pico de duplicados explican mejor una senal que un cambio de 
comportamiento.


## 6. Proxies de entidad

| Proxy | Entidades | Cobertura | Eventos/entidad | Singletons |
|---|---|---|---|---|
| card | 10613 | 1.000 | 54.84 | 0.192 |
| device | 2294 | 0.240 | 60.96 | 0.303 |

Un proxy agrupa comportamiento, no identifica a una persona. Dos clientes pueden 
colisionar en la misma clave y un cliente puede aparecer con varias. Por eso se reporta 
su calidad en lugar de asumirla, y el informe habla de proxies y no de clientes.


## 7. Causalidad de las features

Ventanas de historial: {'1h': 3600, '24h': 86400, '7d': 604800}. Grupos de empate temporal: 12856.


Las features se emiten ANTES de actualizar el estado, y los eventos con el mismo 
`TransactionDT` leen todos el mismo pasado. Esto se verifica en 
`tests/test_point_in_time_features.py`: permutar los IDs empatados deja las features 
identicas, y un evento futuro de monto extremo no altera ninguna fila anterior.
