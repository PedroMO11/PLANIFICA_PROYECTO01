# Reproducción final: qué se verificó

Generado: 2026-09-20T15:12:13+00:00


> Resultados sobre IEEE-CIS (particion train).


## 1. Entorno de la corrida

| Elemento | Valor |
|---|---|
| Python | 3.12.11 |
| Plataforma | Windows-11-10.0.26200-SP0 |
| Procesador | AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD |
| CPUs disponibles | 8 (el código se limita a 4 hilos) |
| Commit | 8018b978dcb312c111eccfb3fe7add23aa2ea20d |
| Hash del código | 183a7374f33e2fbca04ba91bd8cde877 |
| Hash de configuración | 0ae2212bbb6d780d408158fd2d329aa5 |

### Versiones de los paquetes

| Paquete | Versión |
|---|---|
| lightgbm | 4.7.0 |
| matplotlib | 3.11.2 |
| numpy | 2.5.3 |
| pandas | 2.3.3 |
| pyarrow | 25.0.1 |
| river | 0.26.1 |
| scipy | 1.18.1 |
| sklearn | 1.9.1 |

## 2. Qué se ejecutó y qué costó

| Tipo de tarea | Tareas | Minutos |
|---|---|---|
| fit_tuning | 18 | 18.36 |
| fit_final | 6 | 9.84 |
| backtest | 1 | 3.21 |
| fit_seleccion_W | 3 | 0.36 |
| datos | 5 | 0.28 |
| fit_adaptivo | 2 | 0.26 |
| **Total** | 35 | **32.32 de 480** |

Tareas fallidas: **0**. Tareas que excedieron su límite: **0**.


Una tarea fallida o excedida igual consume presupuesto y queda registrada: 
un timeout nunca se presenta como un ajuste exitoso.


## 3. Verificaciones realizadas

| Verificación | Cómo | Resultado |
|---|---|---|
| Suite de pruebas | `python -m pytest` | 137 pruebas (136 pasan, 1 omitida) |
| Notebooks | Ejecutados de principio a fin con `nbclient` | 3 de 3 |
| Servicio HTTP | `uvicorn` + peticiones reales a `/health` y `/predict` | Verificado |
| Replay y ledger | 5 000 eventos, 50 reenvíos | Idempotencia aprobada |
| Cupo diario | 62 días de backtest | Nunca excedido |
| Detector ADWIN | Streams sintéticos con cambio conocido | Detecta en 1 055, 0 falsas alarmas |
| Límite de páginas | Recuento sobre el PDF generado | 8 de 8 |

## 3.b Reproducibilidad verificada entre dos corridas independientes

Se ejecutó la cadena completa dos veces con la misma semilla y los mismos datos, en corridas separadas (`principal` y `reproduccion`).


| Comparación | Resultado |
|---|---|
| Fits de tuning idénticos | **18 / 18** |
| Hash de prerregistro | Idéntico (`c6288e32141507c6`) |
| Costo observado · S0 | Idéntico (1.9596135397 UM/tx) |
| Costo observado · E15 | Idéntico (1.5476906192 UM/tx) |
| Costo observado · W30 | Idéntico (1.2833064702 UM/tx) |
| Costo observado · W60 | Idéntico (1.4340933193 UM/tx) |
| Costo observado · W90 | Idéntico (1.4899289531 UM/tx) |

Comprobable con: `python -m fraud_adaptive --run-id principal verify --against reproduccion`


> Los resultados coinciden pese a que el hash de codigo difiere: los cambios estan en modulos que no participan del entrenamiento. Para una comparacion estricta, ejecutar ambas corridas sin editar el codigo entre ellas.


## 4. Lo que NO pudo verificarse


## 5. Cómo repetirlo

```bash
uv venv --python 3.12
uv pip install -e ".[serving,dev]"
python -m fraud_adaptive data surrogate --scale 1.0   # o los CSV reales de Kaggle
python -m fraud_adaptive all
python -m fraud_adaptive deliver
```

Con los datos reales de IEEE-CIS los comandos son **idénticos**: la única 
diferencia es el archivo de entrada.


## 6. Qué no se garantiza

- **Igualdad bit a bit entre máquinas.** BLAS, versión de CPU y orden de 
reducción en punto flotante pueden diferir.

- **Variabilidad entre semillas.** El núcleo usa solo la semilla 42; los 
intervalos son descriptivos de esa corrida.
