# Registro de decisiones resueltas (C1–C28)

> **Sobre el nombre del archivo.** Se conserva `decisiones_pendientes.md` como ruta
> de compatibilidad con el plan. Su contenido son decisiones **resueltas**: no hay
> preguntas técnicas abiertas ni aprobaciones pendientes.

Cada fila registra una contradicción o ambigüedad detectada entre el enunciado, el
avance y las guías del curso; qué se decidió; y **dónde se puede comprobar en el
código** que se implementó así.

---

## Adaptación y protocolo temporal

| ID | Ambigüedad | Decisión implementada | Dónde se verifica |
|---|---|---|---|
| **C1** | El avance contrapone lotes periódicos y modelo incremental; el enunciado prioriza olvido por ventanas fijas | W ∈ {30, 60, 90} con cadencia 15. S0 y E15 son **solo referencias no desplegables**. W14 e incremental quedan fuera | `configs/adaptation.yaml` · `test_adaptation_windows.py` |
| **C4** | No se especifica si un modelo nuevo conserva el calibrador anterior | Platt **propio por versión** en `[c−21, c−14)`, ajustado después del predictor. Método y umbrales congelados | `adaptation.build_package` · `test_temporal_integrity.py` |
| **C5** | L ≥ 60 usaría una política construida con etiquetas no disponibles en T=120 | Solo **L = 30**. Otros retrasos quedan como extensión sin ejecutar | `configs/temporal.yaml` |
| **C6** | El avance dice «no se puede monitorear F1» pero L principal es 30 días | F1 y costo se monitorean **solo al madurar** `y`. Al cerrar, se avanza únicamente el reloj de disponibilidad | `backtest.run_backtest` (cierre de madurez) |
| **C19** | El arranque 90–119 pretendía alimentar S2, pero la política usa etiquetas hasta el día 89 | Warmup **solo de estado X**: sin scores, sin acciones, sin modelo provisional. El sistema arranca en T=120 | `configs/temporal.yaml` · `backtest` empieza en `test_start` |
| **C20** | No quedaba un holdout maduro posterior a la calibración | H = `[c−7, c)`, posterior a predictor, Platt y reserva de política. Sin shadow deployment | `test_promotion_holdout.py` |
| **C21** | Faltaba definir la contingencia si ninguna ventana es válida | `model_unavailable` (HTTP 503) y pausa. **Nunca** se degrada a S0 o E15 | `serving.py` · `test_sin_paquete_valido_responde_503` |
| **C22** | Elegir W con resultados finales contaminaría la evaluación | Dos folds en `[0,60)`, calibración `[69,76)`, política `[76,83)`, validación `[83,90)`. El test nunca elige | `docs/protocolo_experimental.md` |
| **C17** | La selección previa del dataset ya examinó periodos tardíos | Prerregistro con hash verificado; la exposición exploratoria se **declara** como limitación | `pipeline.run_adaptation` aborta si cambia el hash |

---

## Datos y causalidad

| ID | Ambigüedad | Decisión implementada | Dónde se verifica |
|---|---|---|---|
| **C2** | «Ventana móvil» y «ventana expansiva» se usaban como equivalentes | Se distingue la ventana de **historial de features** (1 h, 24 h, 7 d) de la ventana que limita el **fit del predictor** | `features.HISTORY_WINDOWS` vs `splits.VersionRoles` |
| **C7** | Dos tablas tabulares frente a «fuentes multimodales» | Integración real de transacciones + identidad. Texto y señales de sesión se **dibujan** como adaptadores futuros, sin implementar | `docs/arquitectura.mmd` |
| **C9** | Las guías muestran holdout aleatorio e interpolación con futuro | Splits forward, preprocesamiento por fold, solo pasado. Nunca se traslada el holdout aleatorio del ejemplo | `test_preprocesamiento_no_cambia_al_alterar_filas_futuras` |
| **C10** | Las instrucciones decían que no hacía falta la API de Kaggle | Se documenta la descarga reproducible; credenciales **fuera del repo**. Los `test_*` se rechazan explícitamente | `data.kaggle_download_instructions` · `assert_no_forbidden_files` |
| **C16** | El benchmark histórico ajustaba categorías sobre toda la ventana | Se **cita con su limitación**, no se repara retroactivamente. Todo pipeline nuevo ajusta dentro del fit/fold | `models.FeaturePipeline.fit` |

---

## Decisión, costos y equidad

| ID | Ambigüedad | Decisión implementada | Dónde se verifica |
|---|---|---|---|
| **C8** | El avance fijaba recall a precisión fija; la solicitud pedía FPR fijo y F1 | Se reportan **ambos**, más AP, Brier y latencia. Y se reporta el nivel **realmente obtenido**, que puede incumplir el objetivo | `metrics.recall_at_constraint` |
| **C12** | Ordenar el día completo por `p × monto` exigiría ver el futuro | Admisión **por llegada** mientras haya cupo; la prioridad ordena el servicio. Reserva atómica por `event_id` | `test_admision_es_causal_no_top_k_del_dia` |
| **C13** | La auditoría aleatoria se describía como generadora de etiquetas no sesgadas | Full-information declarado. El analista simulado **no entrena** nada. Selective labels queda como limitación abierta | `decision.simulate_outcomes` · informe §8 |
| **C14** | 150 casos/día se justificaba con 2×8×10 = 160 | 150 efectivos y 10 de holgura **declarada como convención**, no como medición | `configs/decision.yaml` |
| **C18** | «Hasta 120 días» se presentaba como plazo regulatorio universal | L = 30 se describe como **supuesto de simulación**. Se eliminan afirmaciones legales sin sustento | `docs/contrato_sistema.md` |
| **C28** | Un veredicto humano derivado de `isFraud` antes de `available_at` usaría una etiqueta inmadura | El veredicto se calcula **solo al madurar**, en el evaluador. La acción emitida no cambia | `test_el_veredicto_del_analista_no_altera_decisiones_previas` |

---

## Evidencia, cómputo y despliegue

| ID | Ambigüedad | Decisión implementada | Dónde se verifica |
|---|---|---|---|
| **C3** | El benchmark histórico llamaba al gap «prueba genuina» de concept drift | Se cita como evidencia **consistente**, no causal. ADWIN se prueba con dos streams sintéticos de cambio conocido | `drift.adwin_selftest` |
| **C11** | El enunciado pone despliegue (3.4) antes de drift (3.5) | Se ejecuta drift primero: el paquete y la frecuencia usan resultados ya medidos | Orden de fases en `pipeline.py` |
| **C15** | Cifras del benchmark con nombres ambiguos (raw vs corrected) | Los históricos se conservan intactos y se citan con nombres distintos | Informe, sección de referencias |
| **C23** | Una corrida nocturna sin supervisión frente a la aprobación humana | Manifest de autorización **previa y enumerada**; promociones rotuladas `simulada`. Ningún script despliega | `runs/*/authorization.json` |
| **C24** | Entrega cloud manual frente a cupo persistente | Endpoint sin estado de cupo; el replay es el único orquestador, con ledger transaccional | `test_replay_idempotency.py` |
| **C25** | Exigir tres acciones y una alerta podría inducir a fabricar evidencia | Acciones reales cuando existen; **fixtures rotulados** para las ramas faltantes. Sin mover umbrales | `replay.contract_fixtures` |
| **C26** | Repetir fits en notebooks excedería el presupuesto | Los notebooks **leen artefactos**; no reentrenan. Presupuesto persistente que no se reinicia al reanudar | `test_checkpoint_resume.py` |
| **C27** | Tiempo y RAM desconocidos frente a tres modelos | Tres configuraciones por familia congeladas; cap de RF con muestreo estratificado por día y clase | `models.stratified_day_subsample` |

---

## Decisiones tomadas durante la implementación

Estas no estaban en el plan: surgieron al ejecutar y se registran por completitud.

| Situación encontrada | Decisión | Motivo |
|---|---|---|
| IEEE-CIS no disponible en la máquina (sin token de Kaggle) | Generar un **sustituto sintético** con el mismo esquema y drift conocido, rotulado en cada artefacto | Sin él, el pipeline quedaría sin ejecutar ni verificar. Los comandos son idénticos con los datos reales |
| La madurez se comparaba contra `cutoff = T − L` en vez de contra `job_time = T` | Corregido: `VersionRoles.job_time` | Con el criterio anterior, las colas de calibración, política y validación quedaban **vacías** |
| El rezago de monto dependía del orden de llegada entre eventos empatados | Incorporación al historial en orden canónico por `TransactionID` | El plan exige que permutar empatados deje las features iguales; ahora es propiedad de la función, no del llamador |
| La latencia medida incluía solo el commit al ledger | Medición separada **una fila por petición**, con 100 de calentamiento y 1 000 medidas | El replay puntúa el lote por adelantado; mezclarlo daba 0,013 ms en vez de 66 ms |
| El daemon de Docker no estaba activo | `Dockerfile` entregado **sin build verificado**; servicio verificado de forma nativa sobre HTTP | El plan prevé este caso: no se llama «imagen probada» a lo que no se probó |
| La zona gris es ciega al monto | Se **documenta** como característica, con su vía de mejora, en lugar de cambiarla | Cambiarla desviaría del diseño de dos umbrales globales especificado |

---

## Consultas al docente (no bloqueantes)

Cada una tiene un valor por defecto ya aplicado. Ninguna detuvo la implementación.

| Consulta | Valor por defecto aplicado |
|---|---|
| ¿Basta considerar multimodalidad con adaptadores futuros junto a dos fuentes tabulares reales? | Sí: se documenta la consideración y su limitación, sin añadir modalidades ni datasets |
| ¿Las referencias cuentan dentro del máximo de ocho páginas? | Sí. El PDF tiene 8 páginas **incluyendo** referencias, verificado automáticamente |
| ¿Hay formato adicional para infografía o presentación? | SVG + PDF de infografía y guion de 18 minutos. Un PPTX derivado queda como extensión opcional |
