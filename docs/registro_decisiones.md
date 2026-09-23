# Registro de decisiones

Ambigüedades encontradas entre el enunciado, la propuesta del avance y el material del
curso, con la decisión tomada y el lugar del código donde se verifica.

## Adaptación y protocolo temporal

| Ambigüedad | Decisión | Dónde se verifica |
|---|---|---|
| El avance contraponía lotes periódicos y modelo incremental; el enunciado prioriza el olvido con ventanas fijas | Se evalúan W30, W60 y W90 con cadencia de 15 días, junto a S0 estático y E15 expansivo. Los modelos incrementales y W14 quedan fuera | `configs/adaptation.yaml`, `test_adaptation_windows.py` |
| Qué estrategia recomendar si ninguna ventana supera al estático | Se recomienda E15, reentrenamiento periódico que el enunciado incluye entre las estrategias de adaptación, declarando que la recomendación surge del test | `reports/informe_final.md`, sección 5 |
| Si un modelo nuevo conserva el calibrador anterior | Cada versión ajusta su propio Platt después del predictor, sobre datos que este no vio | `adaptation.build_package`, `test_temporal_integrity.py` |
| Con L ≥ 60 la política usaría etiquetas no disponibles en T = 120 | Se evalúa L = 30; otros retrasos quedan como extensión | `configs/temporal.yaml` |
| El avance decía que F1 no se podía monitorear, pero con L = 30 sí es posible | F1 y costo se calculan solo con etiquetas maduras | `backtest.run_backtest` |
| El periodo 90–119 no puede alimentar métricas si la política usa etiquetas hasta el día 89 | El warmup solo calienta el historial de features; el sistema arranca en T = 120 | `configs/temporal.yaml` |
| Faltaba una validación madura posterior a la calibración | H = `[c−7, c)`, posterior al predictor y a Platt | `test_promotion_holdout.py` |
| Qué hacer si ninguna versión es válida | HTTP 503 y pausa, sin decisión por defecto | `serving.py`, `test_sin_paquete_valido_responde_503` |
| Elegir W con resultados de test contaminaría la evaluación | Folds en `[0,60)`, calibración `[69,76)`, política `[76,83)` y validación `[83,90)` | `docs/protocolo_experimental.md` |
| La selección del dataset ya había examinado periodos tardíos | Prerregistro con hash y declaración del trabajo como backtest retrospectivo | `pipeline.run_adaptation` |

## Datos y causalidad

| Ambigüedad | Decisión | Dónde se verifica |
|---|---|---|
| «Ventana móvil» se usaba tanto para el historial de features como para el entrenamiento | Se distinguen las ventanas de historial (1 h, 24 h, 7 d) de la ventana de fit del predictor | `features.HISTORY_WINDOWS` y `splits.VersionRoles` |
| El enunciado pide considerar fuentes multimodales y el dataset tiene dos tablas | Se integran transacciones e identidad; texto y sesión aparecen como adaptadores futuros en la arquitectura | `docs/arquitectura.dot` |
| Las guías del curso muestran holdout aleatorio | Particiones temporales y preprocesamiento por fit | `test_preprocesamiento_no_cambia_al_alterar_filas_futuras` |
| Cómo obtener los datos sin exponer credenciales | Descarga documentada, token fuera del repositorio y rechazo explícito de los `test_*` | `data.kaggle_download_instructions`, `assert_no_forbidden_files` |
| El benchmark de selección ajustaba categorías sobre toda la ventana | Se cita con esa limitación; el pipeline ajusta dentro de cada fit | `models.FeaturePipeline.fit` |

## Decisión, costos y equidad

| Ambigüedad | Decisión | Dónde se verifica |
|---|---|---|
| El avance fijaba recall a precisión fija; también se pedían FPR fijo y F1 | Se reportan recall a FPR 1 %, recall a precisión 80 %, F1, balanced accuracy, AP, Brier y latencia, con el nivel realmente obtenido | `metrics.recall_at_constraint` |
| Ordenar el día completo por `p × monto` exigiría ver el futuro | Admisión por llegada con reserva atómica por `event_id` | `test_admision_es_causal_no_top_k_del_dia` |
| La auditoría aleatoria se describía como fuente de etiquetas sin sesgo | Se usa información completa y el analista simulado no entrena el modelo; selective labels queda como limitación | `decision.simulate_outcomes` |
| La capacidad de 150 se justificaba con 2 × 8 × 10 = 160 | 150 efectivos con 10 de holgura declarada | `configs/decision.yaml` |
| «Hasta 120 días» se presentaba como plazo regulatorio universal | L = 30 se describe como supuesto de simulación | `docs/contrato_sistema.md` |
| El veredicto del analista derivado de `isFraud` antes de `available_at` usaría una etiqueta inmadura | El veredicto se calcula al madurar la etiqueta, en el evaluador | `test_el_veredicto_del_analista_no_altera_decisiones_previas` |

## Evidencia, cómputo y despliegue

| Ambigüedad | Decisión | Dónde se verifica |
|---|---|---|
| El benchmark llamaba «prueba» de concept drift a la brecha entre modelos | Se cita como evidencia consistente; ADWIN se valida con series sintéticas de cambio conocido | `drift.adwin_selftest` |
| El enunciado presenta el despliegue (3.4) antes del drift (3.5) | Se ejecuta primero el experimento de drift, porque la frecuencia de actualización depende de él | Orden de fases en `pipeline.py` |
| Cifras del benchmark con nombres ambiguos (sin corregir y corregidas) | Se conservan y se citan con nombres distintos | `docs/antecedentes/` |
| Una corrida nocturna frente a la aprobación humana obligatoria | Manifiesto de autorización previa y promociones rotuladas como simuladas | `runs/<run_id>/authorization.json` |
| Despliegue en la nube manual frente a cupo persistente | El endpoint no guarda el cupo; el replay es el único orquestador, con ledger transaccional | `test_replay_idempotency.py` |
| Mostrar las tres acciones y una alerta podía inducir a fabricar evidencia | Se usan acciones reales y fixtures rotulados para las ramas que no aparecen | `replay.contract_fixtures` |
| Repetir fits en los notebooks excedería el presupuesto | Los notebooks leen artefactos; el presupuesto se conserva al reanudar | `test_checkpoint_resume.py` |
| Tiempo y memoria desconocidos para tres familias | Tres configuraciones por familia; Random Forest con muestreo estratificado por día y clase | `models.stratified_day_subsample` |

## Ajustes surgidos al ejecutar

| Situación | Decisión | Motivo |
|---|---|---|
| La madurez se comparaba contra `cutoff = T − L` | Se compara contra `job_time = T` | Con el criterio anterior las colas de calibración y validación quedaban vacías |
| El rezago de monto dependía del orden de llegada entre eventos empatados | Orden canónico por `TransactionID` | Permutar eventos empatados debe dejar las features iguales |
| La latencia medida incluía solo el commit al ledger | Medición HTTP de una fila por petición, con 100 de calentamiento y 1 000 medidas | El replay puntúa el lote por adelantado y la cifra no representaba el servicio |
| Los umbrales globales ignoran el monto | Regla por mínimo costo esperado | 1,5895 frente a 1,9696 UM/tx en la reserva de política |
| `c_FP = 5` bloqueaba al 11,17 % de las legítimas | `c_FP` derivado del objetivo de 1 % | Resulta 25 UM, con 0,89 % |
| El cupo no intervenía en la regla de decisión | Precio sombra `λ` del cupo | La demanda pasó de 704 a 150 revisiones diarias |

## Consultas al docente

Cada consulta tiene un valor por defecto aplicado.

| Consulta | Valor aplicado |
|---|---|
| ¿Basta tratar la multimodalidad como adaptadores futuros junto a dos fuentes tabulares? | Sí; se documenta la consideración y su límite |
| ¿Las referencias cuentan dentro de las ocho páginas? | Sí; el PDF tiene 8 páginas con referencias |
| ¿Hay un formato exigido para la infografía o la presentación? | Infografía en SVG y PDF, y guion de 18 minutos |
