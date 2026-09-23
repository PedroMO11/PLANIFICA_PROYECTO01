# Semana 7 — Desviación conceptual (Concept Drift) · Guía de estudio

| Campo | Valor |
|---|---|
| **Curso** | Planificación y Toma de Decisiones en IA |
| **Sesión** | Semana 7 · *Concept Drift* · 2026-2 |
| **Docente** | Dra. Aurea Soriano-Vargas (`aureasoriano` · asoriano@utec.edu.pe · `aurea-soriano`) |
| **Institución** | UTEC |
| **Fuente** | Deck *Concept Drift · 2026-2* (PDF) — 62 diapositivas |
| **Atribución del pie** | *Planificación y Toma de Decisiones en IA*, Dr. Aurea Soriano-Vargas (2026) |

> **Propósito de esta guía:** reconstruir fielmente el contenido teórico de las 62 diapositivas
> de la sesión —con especial cuidado en los **diagramas** (matriz de estrategias adaptativas,
> ventanas de entrenamiento, ensambles) y en el **pseudocódigo de ADWIN**, que en el PDF están
> como imágenes— para poder estudiar la sesión sin depender del deck original.

---

## 1. Mapa del curso: ubicación de la sesión

Diagrama **"La historia de este curso"** (diapositivas 2, 6 y 59). El recuadro con borde
amarillo marca la sesión en foco; los grises son sesiones ya cubiertas.

```
Fundamentos        ┌──────────────────────┐
                   │ Motivación y         │
                   │ terminología         │
                   └──────────┬───────────┘
                              │
Ciclo de vida      ┌──────────▼───┐   ┌──────────────┐   ┌───────────┐   ┌──────────────────┐
de la IA           │  Iniciación  │──▶│  Modelado y  │──▶│ Despliegue│──▶│ Desviación       │
                   └──────────────┘   │  rendimiento │   └───────────┘   │ conceptual       │
                                      └──────────────┘    ↑ resaltado    │ (Concept Drift)  │
                                                            en la diap. 2 └────────┬─────────┘
                                                            (repaso)        ↑ ESTA SESIÓN
                                                                              (diap. 6)
                              ┌───────────────────────────────────────────────────┘
IA en sistemas     ┌──────────▼──────┐   ┌──────────────┐   ┌──────────────────┐
                   │    Modelos      │──▶│ Aprendizaje  │──▶│ Colaboración     │
                   │  Fundacionales  │   │ en todo el   │   │ entre humanos    │
                   └─────────────────┘   │   sistema    │   │ e IA             │
                    ↑ resaltado en la    └──────────────┘   └────────┬─────────┘
                      diap. 59                                       │
                      (PRÓXIMA SEMANA)                               │
                              ┌────────────────────────────────────────┘
Temas importantes  ┌──────────▼───┐   ┌──────────────────┐
de IA              │ IA Creativa  │──▶│ Ética y equidad  │
                   └──────────────┘   │    en la IA      │
                                      └──────────────────┘
```

| Diap. | Resaltado (borde amarillo) | En gris (ya visto) |
|---|---|---|
| **2** | Despliegue *(repaso)* | Motivación y terminología · Iniciación · Modelado y rendimiento |
| **6** | **Desviación conceptual (Concept Drift)** *(esta sesión)* | Lo anterior + Despliegue |
| **59** | Modelos Fundacionales *(próxima semana)* | Todo el bloque *Ciclo de vida de la IA* |

> *Nota:* a diferencia del deck de Modelado (Sem 4), este deck **sí adelanta la sesión
> siguiente**: la diapositiva 59, tras el separador "Próxima Semana", resalta **Modelos
> Fundacionales** y deja en gris todo el ciclo de vida.

### 1.1 Repaso: el proceso de despliegue (diap. 3, 7)

```
┌──────────────┐   ┌─────────────────┐   ┌──────────────┐   ┌──────────────┐
│ Entrenar     │──▶│ Opciones de     │──▶│ Desplegar el │──▶│ Garantizar   │
│ modelo final │   │ implementación  │   │ modelo       │   │ la validez   │
└──────────────┘   └─────────────────┘   └──────────────┘   └──────▲───────┘
                                                                    │
                                              diap. 7: una flecha señala este paso
                                              → aquí se ubica el concept drift
```

### 1.2 Repaso: opciones de despliegue (diap. 4)

**¿Cuáles son las decisiones para el despliegue del modelo?**

| Pregunta | Opciones |
|---|---|
| **¿Cómo encapsular el modelo?** | Monolítico · Microservicios |
| **¿Qué componentes se necesitan para controlar, desplegar y ejecutar los servicios?** | En las instalaciones (*on-premises*) · IaaS (Infraestructura como Servicio) · PaaS (Plataforma como Servicio) · SaaS (Software como Servicio) |
| **¿Cómo encapsular el entorno de despliegue respecto a los componentes físicos?** | Instancia única · Máquina Virtual · Contenedor (e.g. Docker) |

### 1.3 Repaso: conclusiones de Despliegue (diap. 5)

| Idea clave | Desarrollo |
|---|---|
| **Un modelo genera valor cuando pasa de experimento a producción** | y se integra correctamente con un sistema real. |
| **No existe una única estrategia de despliegue:** | la elección depende del control, costo, escalabilidad, infraestructura y autonomía requerida. |
| **MLOps convierte el ML en un proceso reproducible y mantenible** | automatizando pruebas, integración, despliegue y monitoreo. |

> **MLflow** aporta trazabilidad y reproducibilidad, permitiendo registrar y comparar código,
> parámetros, métricas y artefactos de los experimentos.

---

## 2. Objetivos y agenda de la sesión

### 2.1 Al finalizar la clase deberíamos tener los medios para responder (diap. 8)

1. Importancia del monitoreo continuo de soluciones de IA desplegadas.
2. Adaptar modelos de predicción.
3. Conocer concept drift.
4. Detección de concept drift.

### 2.2 Agenda (diap. 9)

1. **Introducción:** ¿Qué es la desviación conceptual?
2. **Fundamentos teóricos:** definición, tipos y ejemplos
3. **Estrategias de aprendizaje adaptativo**
4. **Algoritmos de detección de concept drift** (ADWIN, Page-Hinkley, CVFDT)

> ⚠ La agenda promete tres algoritmos, pero en el cuerpo del deck **solo ADWIN se desarrolla**;
> Page-Hinkley y CVFDT únicamente se nombran (ver §6.8).

---

## 3. Introducción (sección, diap. 10–16)

### 3.1 ¿Cómo mantener correcto el aprendizaje automático (microservicios)? (diap. 11)

```
 [imagen: manómetro nuevo en        ┌──────────────────────┐
  planta industrial limpia]         │ Entorno de           │- - - - - ┐  Modelo basado en
            │                       │ entrenamiento/prueba │          │  entorno de entrenamiento.
            │  • Cambios en los     └──────────────────────┘          ▼
            │    datos o contexto          ⚠ Distribución de     ┌────────┐
            ▼    a lo largo del              datos diferente     │ Modelo │  Modelo desplegado
 [imagen: el mismo manómetro,       ┌──────────────────────┐     └────────┘
  sucio y deteriorado]              │ Entorno productivo   │──────────▲
            • Acciones sobre modelo └──────────────────────┘ Flujo de datos continuo
```

- El modelo se construye con el **entorno de entrenamiento/prueba**, pero en producción recibe
  un **flujo de datos continuo** del **entorno productivo**.
- Entre ambos puede existir una **distribución de datos diferente** (⚠).
- Las dos consecuencias que la diapositiva lista: **cambios en los datos o contexto a lo largo
  del tiempo** y **acciones sobre el modelo**.

### 3.2 Los flujos de datos pueden cambiar… (diap. 12)

| Cambio de contexto/concepto de los datos | Aprendizaje automático | Resultados |
|---|---|---|
| **Predicción de sentimientos:** cambios en patrones de comunicación. | ⇨ | **?** |
| **Predicción de precios:** cambios en políticas del mercado. | ⇨ | **?** |
| **Predicción de inactividad:** cambios en parámetros de máquinas. | ⇨ | **?** |

### 3.3 Validez continua de los modelos garantizada mediante adaptaciones (diap. 13–14)

Diagrama: plano 2D con cuadrados **azules** (arriba) y **rojos** (abajo) separados por una recta
verde, **"Modelo desplegado originalmente"**. Aparecen un rojo sobre la recta y un azul bajo
ella (mal clasificados); en la diap. 14 se marcan con círculos y se dibuja una segunda recta,
**"Modelo adaptado o reentrenado"**, que sí los separa.

| Paso | Descripción según el deck |
|---|---|
| **Monitoreo constante** | … del flujo de datos (por ejemplo, basado en propiedades estadísticas). |
| **Detección de cambios** | El sistema decide si el modelo debe actualizarse. |
| **Adaptación** | El modelo se ajusta **ligeramente** en función de los nuevos datos. |
| **Reentrenamiento** | El modelo se reentrena **completamente desde cero**. |

> *"El mismo modelo que funcionaba ayer puede estar desfasado hoy."* (diap. 14)

### 3.4 Definición de Concept Drift (diap. 15)

> **Concept Drift** describe cómo los datos y relaciones que un modelo aprendió en
> entrenamiento pueden dejar de ser válidos en el tiempo, **afectando la validez de sus
> predicciones**.

### 3.5 Pregunta para la clase (diap. 16)

> *¿Qué ejemplos cotidianos conocen donde las reglas "cambian con el tiempo" y afectan a un
> sistema de predicción?*

---

## 4. Fundamentos teóricos (sección, diap. 17–28)

### 4.1 Concepto y concept drift (diap. 18)

**Definición de concepto:**

$$\text{Concepto} = P(X, y)$$

**Definición de concept drift entre dos puntos temporales:**

$$P_{t_0}(X, y) \neq P_{t_1}(X, y)$$

- Asociado a **datos en flujo con marcas de tiempo**.
- El modelo aprende de **datos anteriores a $t$**.
- Se aplica para predecir **datos posteriores a $t$**.

La diapositiva acompaña las fórmulas con dos diagramas de dispersión (antes → después): en el
segundo, las nubes azul y roja se han reacomodado.

### 4.2 ¿Por qué la desviación conceptual es inevitable? (diap. 19–20)

> *Los datos son reflejo de la realidad, y la realidad siempre cambia.*

*(La diap. 19 es el estado previo con solo la pregunta.)*

### 4.3 Drift real vs. drift virtual (diap. 21)

| | **Datos originales** | **Concept drift real** | **Drift virtual** |
|---|---|---|---|
| **Qué cambia** | — | Cambios en $P(y \mid X)$; $P(X)$ puede mantenerse constante o no. | $P(X)$ cambia **sin afectar** a $P(y \mid X)$. |
| **Diagrama** | Frontera curva que separa azules y rojos. | La **frontera cambia de forma** (pasa a ser casi horizontal y ondulada). | La **frontera es la misma** que en los datos originales; lo que se desplaza es la nube de puntos. |

> *Nota de lectura (no está en el deck):* como $P(X, y) = P(y \mid X)\,P(X)$, un cambio en
> cualquiera de los dos factores cambia el "concepto" de §4.1; el deck llama **real** al cambio
> en $P(y \mid X)$ y **virtual** al cambio solo en $P(X)$.

### 4.4 Pregunta para la clase (diap. 22)

> *¿Cuál creen que es más difícil de detectar: cuando cambia la relación entre variables y
> etiquetas, o cuando solo cambia la distribución de entrada?*

*(El deck no ofrece respuesta explícita a esta pregunta.)*

### 4.5 Ejemplo: clasificación de noticias (diap. 23)

> **Tarea:** Clasificar en un flujo de noticias en línea los artículos sobre bienes raíces como
> relevantes o no relevantes para un usuario dado.
> **Escenario:** El usuario está buscando un nuevo departamento.

| | Antes de $t$ | Evento en $t$ | Después de $t$ |
|---|---|---|---|
| **Drift virtual** | Las noticias sobre departamentos en la ciudad son relevantes; las noticias sobre casas de vacaciones no son relevantes. | El editor del portal de noticias cambia → estilo de redacción | Los departamentos en la ciudad **siguen siendo relevantes**. |
| **Concept drift real** | Las noticias sobre departamentos en la ciudad son relevantes; las noticias sobre casas de vacaciones no son relevantes. | El usuario compra una casa → comienza a buscar una casa de vacaciones | Los departamentos en la ciudad **se vuelven irrelevantes**, las casas de vacaciones **son relevantes**. |

### 4.6 Ejemplo: música (diap. 24)

| **Drift Real** (cambia la relación entre X e Y, es decir, cambia lo que determina la etiqueta) | **Drift Virtual** (cambia la distribución de X, pero no cambia la relación con Y) |
|---|---|
| Ilustración: *"La música también evoluciona conmigo"* — **El gusto musical cambia con la edad:** 15 años → Pop · 25 años → Indie / Rock · 35 años → Jazz / Clásica. | Ilustración: *"Mis gustos siguen iguales"* — una app de streaming musical pasa de la **interfaz anterior** (Inicio, Buscar, Tu biblioteca; listas "Lo mejor de hoy", "Rock Clásico", "Indie Hits") a una **nueva interfaz** (buscador, filtros Todo/Música/Podcasts, sección "Hecho para ti"). |

### 4.7 Los datos cambian con el tiempo de diversas maneras (diap. 25)

| Tipo | Forma de la señal en el tiempo | Ejemplo del deck |
|---|---|---|
| **Cambios repentinos** | `▂▂▂▂▂▂▆▆▆▆▆▆` | Cambio de un sensor. |
| **Cambios lentos** | `▂▂▂▃▄▄▅▆▆▆▆▆` | Sensor que se degrada lentamente. |
| **Cambio gradual** | `▂▂▆▂▂▆▆▂▆▆▆▆` | Usuario interesado en finanzas, luego en deportes, pero que vuelve a consultar finanzas. |
| **Conceptos recurrentes** | `▂▂▂▂▆▆▆▆▂▂▂▂` | Patrón estacional en la predicción de ventas. |
| **Valores atípicos (outliers)** | `▂▂▂▂▂▂█▂▂▂▂▂` | Dificultad para no clasificar erróneamente los valores atípicos como drift; de lo contrario, **adaptación falsa**. |

> *(Formas de señal transcritas de los cinco mini-gráficos "valor vs. tiempo" de la diapositiva.)*

### 4.8 Desafíos (diap. 26)

- **Detectar** lo antes posible.
- **Adaptar** el modelo rápidamente.
- **Distinguir** la desviación del ruido.
- **Reconocer** contextos recurrentes.
- **Operar** en menos tiempo que la llegada de un nuevo ejemplo.

### 4.9 ¿Qué pasa si ignoramos la desviación? (diap. 27–28)

- Modelos obsoletos → predicciones inútiles.
- Costos altos (ejemplo: fallos en mantenimiento predictivo).
- Pérdida de confianza en la IA.

---

## 5. Estrategias de aprendizaje adaptativo (diap. 29–45)

### 5.1 Recapitulación: aprendizaje supervisado (diap. 29–30)

```
                 ┌───────────┐          ┌─────────────┐
                 │ Población │ ── = ?? ─│ Población'  │ (diap. 30)
                 └─────┬─────┘          └──────┬──────┘
                       ▼                       ▼
            X  ┌─────────────────┐      ┌─────────────────┐  X'
               │ Datos Históricos│      │ Datos de Prueba │
            y  │ Etiquetas de    │      └────────┬────────┘
               │ entrenamiento   │               ▼
               └────────┬────────┘   Entrenamiento   ┌──────────┐
                        └───────────────────────────▶│ Modelo ML│
                                     y = M(X)        └────┬─────┘
                                                          ▼
                                                 Etiquetas de prueba  y'
```

**Entrenamiento:** entrenar un modelo

$$y = M(X)$$

**Aplicación:** aplicar el modelo a datos no vistos

$$y' = M(X')$$

> En la diap. 30 la población de prueba se convierte en **Población'** y se pregunta si
> **Población = Población'** ("= ??"); la ecuación $y' = M(X')$ aparece en rojo junto al aviso
> ⚠ **Distribución de datos diferente**.

### 5.2 Reentrenar vs. adaptar (diap. 31)

Dos GIFs rotulados **REENTRENAR** y **ADAPTAR** ("We must continue to adapt"). Las
definiciones son las de §3.3: *reentrenamiento* = desde cero; *adaptación* = ajuste ligero con
los nuevos datos.

### 5.3 El modelo en un entorno de flujo de datos / procesamiento en línea (diap. 32)

```
Flujo de    (●)──(●)──(◐)──(◐)──(◐)──(◐)──( ? )──────▶ Tiempo
datos        │    │    │    │    │    │     │       ● Concepto 1 (amarillo)
            [▤]  [▤]  [▤]  [▤]  [▤]  [▤]   [▤]      ◐ Concepto 2 (gris)
            └──────── Entrenamiento ────────┘│
                                             ▼
                                      ┌──────────┐
                                      │ Modelo ML│  Modelo actualizado
                                      └────┬─────┘
                                           ▼
                                       Predicción
```

> **Aprendizaje adaptativo:** Con el tiempo, el modelo se actualiza o se reentrena a sí mismo si
> es necesario.

### 5.4 Matriz de estrategias de aprendizaje adaptativo (diap. 33, 34, 38, 40, 43, 45)

> *Existen diferentes estrategias de aprendizaje adaptativo disponibles.*

| **Forma del modelo** ↓ \ **Actualizaciones del modelo** → | **Con triggers** | **Evolutivo** |
|---|---|---|
| **Modelo único** | **Detectores** (§5.6) | **Olvido** (§5.5) |
| **Ensamble** | **Contextual** (§5.8) | **Ensamble dinámico** (§5.7) |

La matriz se repite resaltando un cuadrante cada vez, en este orden:
**Olvido** (diap. 34) → **Detectores** (38) → **Ensamble dinámico** (40) → **Contextual** (43)
→ matriz sin resaltar como cierre (45).

### 5.5 Olvido: una ventana de entrenamiento fija garantiza el olvido (diap. 34–37)

*(Diap. 35: GIF sin texto que alude a "borrar la memoria".)*

La ventana de entrenamiento tiene **tamaño fijo (6 lotes)** y se desliza un lote a la vez; el
lote siguiente es el que se predice. Lotes **oscuros** = anteriores; **claros** = nuevos.

```
Paso 1:  [▓ ▓ ▓ ░ ░ ░] ░                  ← entrenar sobre 3 antiguos + 3 nuevos, predecir
Paso 2:  ▓ [▓ ▓ ░ ░ ░ ░] ░
Paso 3:  ▓ ▓ [▓ ░ ░ ░ ░ ░] ░
Paso 4:  ▓ ▓ ▓ [░ ░ ░ ░ ░ ░] ░            ← los datos antiguos ya salieron de la ventana
         └──── Entrenamiento ────┘ Predecir               ────────────────▶ Tiempo
```

> Como la ventana nunca crece, los datos viejos salen de ella **automáticamente**: el olvido
> está garantizado sin necesidad de detectar el cambio.

### 5.6 Detectores: la detección de cambios conduce a la eliminación de datos antiguos (diap. 38–39)

```
 ▓   ▓   ▓  ⚠  ░   ░   ░    ░              ──────▶ Tiempo
          Detectar cambio

┌───────────────┐┌───────────────┐
│ ▒▒▒▒▒▒▒▒▒▒▒▒▒ ││  ░   ░   ░    │   ░
└───────────────┘└───────────────┘
 Descartar datos   Reentrenar el    Predecir
 antiguos          modelo
```

1. **Detectar cambio** (⚠ entre el tercer y el cuarto lote).
2. **Descartar datos antiguos.**
3. **Reentrenar el modelo** solo con los lotes posteriores al cambio.
4. **Predecir.**

### 5.7 Ensamble dinámico: varios modelos de ML en paralelo (diap. 40–42)

Cuatro clasificadores, cada uno entrenado con una ventana distinta del flujo; sus predicciones
se combinan con un **meta-modelo o votación**.

| Clasificador | Ventana de entrenamiento | Predicción | Con **etiqueta verdadera: +** (diap. 42) |
|---|---|---|---|
| **1** | 3 lotes antiguos | **+** | **Recompensa:** aumento de peso |
| **2** | 1 antiguo + 2 nuevos | **+** | **Recompensa:** aumento de peso |
| **3** | 5 nuevos | **−** | **Castigo:** disminución de peso |
| **4** | 1 antiguo + 5 nuevos | **−** | **Castigo:** disminución de peso |

```
Clasificador 1 ─┐
Clasificador 2 ─┤
Clasificador 3 ─┼──▶ ⟩ Meta-modelo o votación
Clasificador 4 ─┘
```

> *Nota de lectura:* con votación simple el ejemplo queda **empatado 2–2**; son los **pesos**,
> actualizados con recompensa/castigo según cada acierto, los que deciden la combinación.

### 5.8 Contextual: identificar la afiliación de grupo de la nueva instancia (diap. 43–44)

```
Grupo 1: Clasificador 1   ▓ ▓ ▓
Grupo 2: Clasificador 2   ▓ ▓ ░ ░
Grupo 3: Clasificador 3   ░ ░ ░ ░ ░   ◀──── nueva instancia ░
```

| Fase | Descripción según el deck |
|---|---|
| **Entrenamiento** | Particionar los datos de entrenamiento en varios grupos y construir modelos separados para cada grupo. |
| **Predicción** | La nueva instancia se asigna a un grupo y se aplica el modelo correspondiente. |
| **Caso de uso** | Especialmente adecuado para **conceptos recurrentes** (por ejemplo, patrones estacionales en la predicción de ventas). Mini-gráfico: `▂▂▂▂▆▆▆▆▂▂▂▂`. |

---

## 6. Algoritmos de detección (sección, diap. 46–54)

### 6.1 Métodos informados vs. métodos ciegos (diap. 47)

> *Amplia variedad de enfoques de detección de cambios y adaptación disponibles.*

| | **Métodos informados** *(Métodos basados en detectores)* | **Métodos ciegos** *(Métodos basados en el olvido)* |
|---|---|---|
| **Idea** | Detección **explícita** de cambios. | Modelo que se adapta de forma incremental o que se reentrena con frecuencia. |
| **Diagrama** | *Algoritmo de detección de cambios* (Detect change ⚠) → *Modelo de predicción* (Drop old data → Retrain model → Predict). Es el esquema de §5.6. | Ventana fija deslizándose sobre el tiempo. Es el esquema de §5.5. |
| **Técnicas** | La detección de cambios normalmente se basa en el **error de predicción**: <br>• Análisis secuencial: **Page-Hinkley** <br>• Monitoreo de dos distribuciones: **ADWIN** | • Disminución incremental de peso para observaciones más antiguas <br>• **CVFDT** (Concept-Adapting Very Fast Decision Trees) |

### 6.2 Los métodos informados pueden basarse en la distribución o en la tasa de error (diap. 48)

| | **Detección de drift basada en la distribución de datos** | **Detección de desviación basada en la tasa de error** |
|---|---|---|
| **Objetivo** | Cuantificar la disimilitud entre la distribución de los datos antiguos y los nuevos con una **función de distancia**. | Detectar cambios considerando la **tasa de error** del modelo de aprendizaje automático subyacente. |
| **Concepto** | Dos distribuciones (campanas) desplazadas y solapadas. | Curva de *Performance* vs. *Time* estable que cae bruscamente. |
| **Algoritmos de ejemplo** | Distancias posibles: divergencia de **Kullback–Leibler** o **Kolmogorov–Smirnov**. | **ADWIN**, **Page-Hinkley-Test** |
| **En la práctica** | **Verificaciones de consistencia para los datos de entrada (media, varianza).** | A menudo se requiere supervisión manual de la calidad de predicción y del impacto en los KPIs. |
| **Desventajas** | Computacionalmente intensivo. | Requiere **etiquetas verdaderas** para la detección de drift. |

> En la diapositiva, "Verificaciones de consistencia…" y "ADWIN" aparecen **enmarcados**: son
> los dos hilos que se desarrollan a continuación (§6.3 y §6.4, respectivamente).

### 6.3 Funciones de costo evalúan saltos en las características de la distribución (diap. 49)

**Funciones de costo – un ejemplo sencillo:**

- La detección de puntos de cambio utiliza una **ventana deslizante** con una **función de
  costo** para identificar cambios en la señal.
- La **desviación estándar** puede detectar cambios en la media, aumentando cuando la señal
  presenta saltos.
- Los puntos de cambio se detectan mediante:
  - **(a)** comparación con un **umbral fijo**, o
  - **(b)** comparación con una **segunda ventana deslizante**.

| Gráfico | Título | Lectura |
|---|---|---|
| **(a)** | *Sliding window – length: 15* | Señal ≈3,0 para $t$ = 0–19 y salto a ≈4,7–5,4 desde $t$ ≈ 20; media (naranja) sobre la primera ventana, **Std 0.17**. Los puntos de cambio se marcan si los costos superan un umbral en desviaciones estándar. |
| **(b)** | *Sliding window – past length: 8 – future length 8* | Misma señal; mean (past) naranja y mean (future) roja, con **Std 0.17** y **Std 0.16**. El punto de cambio puede detectarse comparando los costos de estas dos ventanas. |

*(Valores leídos de los gráficos; aproximados. Ambos gráficos llevan la referencia "[1]".)*

### 6.4 ADWIN: Adaptive Sliding Window compara ventanas deslizantes (diap. 50)

**Algoritmo Adaptive Sliding Window (ADWIN)** — pseudocódigo de la figura *"Figure 1: Algorithm
ADWIN"*:

```
ADWIN: ADAPTIVE WINDOWING ALGORITHM
1  Initialize Window W
2  for each t > 0
3      do W ← W ∪ {x_t}   (i.e., add x_t to the head of W)
4         repeat Drop elements from the tail of W
5            until |μ̂_W0 − μ̂_W1| ≥ ε_cut holds
6               for every split of W into W = W0 · W1
7         output μ̂_W
```

Condición de las líneas 5–6:

$$\left|\hat{\mu}_{W_0} - \hat{\mu}_{W_1}\right| \ge \varepsilon_{cut}
\qquad\text{para cada partición}\qquad W = W_0 \cdot W_1$$

> ⚠ **Nota de lectura, no corregida:** la propia diapositiva explica que *cuando las medias de
> dos subventanas difieren lo suficiente se descartan los elementos más antiguos*; con esa
> lógica, el bucle debería **terminar** cuando la diferencia es **menor** que $\varepsilon_{cut}$
> en todos los cortes. El símbolo **≥** que muestra la figura parece invertido respecto de la
> explicación. Se conserva tal cual. **[verificar con la docente]**

- Utiliza una ventana de detección **W** que se adapta de manera iterativa.
- Cuando dos (sub)ventanas grandes de W muestran medias lo suficientemente distintas, el
  algoritmo **descarta los elementos más antiguos**.
- El umbral $\varepsilon_{cut}$ está definido por la **cota de Hoeffding**.

**Concepto** (recuadro de la diapositiva): *Detección de drift basada en la tasa de error* —
curva de Desempeño vs. Tiempo que cae.

**Gráfico de ejemplo** (serie temporal de 0 a 200 pasos; eje derecho: tamaño de ventana 0–100):

| Evento | Serie (azul) | Media de la ventana (rojo, discontinuo) | Tamaño de ventana (verde) |
|---|---|---|---|
| **Cambio incremental** ($t$ ≈ 30–60) | Rampa de 0 a ≈14; en $t$ ≈ 60 cae a 0. | Sigue la rampa con retraso; tras la caída decae lentamente. | Crece linealmente hasta ≈40 y **colapsa** a ≈5 en $t$ ≈ 43. |
| **Cambio repentino** ($t$ ≈ 100) | Salto de 0 a 10 y se mantiene. | Salta y converge a 10. | Había crecido hasta ≈60; **colapsa** en $t$ ≈ 103 y vuelve a crecer hasta ≈100. |

*(Valores leídos del gráfico; aproximados.)*

### 6.5 ADWIN paso a paso: cambios repentinos (diap. 51–52)

**Sin cambio (diap. 51):**

```
Comienza aquí
     ▼
▪  [ ▪  ▪  ▪  ▪ ] [x_t]        ← W y el nuevo elemento x_t (mismo nivel)
   [W0 │   │   │   │ W1]       ← se prueban todos los cortes W = W0 · W1
       [W0 │   │   │ W1]
           [W0 │   │ W1]
               [W0 │ W1]
```

> **Las líneas 5 y 6 del algoritmo nunca se aplican**, por lo tanto no se detecta ningún cambio
> y la ventana (W) crece con el nuevo (x_t).

**Con cambio repentino (diap. 52):**

```
                          [x_t]     ← x_t llega a un nivel más alto
▪  [ ▪  ▪  ▪  ▪  ▪ ]
   [W0 │   │   │   │   │ W1]
       [W0 │   │   │   │ W1]
           [W0 │   │   │ W1]
               [W0 │   │ W1]
                   [W0 │ W1]  ◀── "Ahí es donde las líneas 5 y 6 del algoritmo
                                    se aplican en el caso de un cambio repentino."
```

> **ADWIN reduce bruscamente la ventana y detecta rápidamente el drift.**

### 6.6 ADWIN ante cambios lentos (diap. 52)

Mini-gráfico **Cambios Lentos**: `▂▂▂▃▄▄▅▆▆▆▆▆`.

> *¿Cómo se comporta ADWIN en el caso de un cambio lento?*
> **ADWIN ajusta gradualmente el tamaño de la ventana, detectando el drift de forma progresiva.**

### 6.7 ¿Por qué detectar un drift lento es más difícil que uno repentino? (diap. 53–54)

| **Drift lento** | **Drift repentino** |
|---|---|
| Se parece al **"ruido natural"**. | Genera **saltos bruscos** que son estadísticamente más fáciles de identificar. |

### 6.8 Page-Hinkley y CVFDT: lo que dice el deck

| Algoritmo | Dónde aparece | Qué dice el deck |
|---|---|---|
| **Page-Hinkley** | Agenda (9), diap. 47, diap. 48 | Método **informado**, de **análisis secuencial**, basado en la **tasa de error**. |
| **CVFDT** | Agenda (9), diap. 47 | *Concept-Adapting Very Fast Decision Trees*; método **ciego** (basado en el olvido). |

> El deck **no desarrolla** el funcionamiento de ninguno de los dos más allá de esta
> clasificación.

---

## 7. Conclusiones (diap. 55–57)

| Idea clave | Desarrollo |
|---|---|
| **Los modelos no permanecen válidos para siempre** | los datos y las relaciones aprendidas cambian con el tiempo, por lo que el concept drift es prácticamente inevitable. |
| **No todo cambio significa lo mismo** | el drift real modifica la relación entre las variables y la etiqueta, mientras que el drift virtual cambia la distribución de los datos sin alterar esa relación. |
| **Desplegar un modelo no es el final** | *e* requiere monitoreo continuo para detectar cambios y decidir cuándo adaptar o reentrenar el modelo. |

### 7.1 ¿Cómo integrarían la detección de concept drift en un ciclo de MLOps? (diap. 57)

Diagrama **"Aplicaciones basadas en aprendizaje automático"**:

```
          LLEGARON NUEVOS DATOS          EL ALGORITMO DE ML CAMBIÓ
        ┌───────────────────────┐      ┌─────────────────────────┐
        │                       ▼      │                         ▼
   ┌─────────┐              ┌─────────┐                    ┌─────────┐
   │  DATOS  │              │ MODELO  │                    │ CÓDIGO  │
   └─────────┘              └─────────┘                    └─────────┘
        ▲                       │      ▲                         │
        └───────────────────────┘      └─────────────────────────┘
          SE REQUIERE REETIQUETADO         SE NECESITA UN MODELO DIFERENTE
```

*(Diap. 58: "Próxima Semana". Diap. 59: mapa del curso. Diap. 60: "Thank you! / Obrigado! /
¡Gracias!". Diap. 61: "Aquí se contestan preguntas raras. ¡Pregunten sin miedo!". Diap. 62:
portada de cierre.)*

---

## 8. Preguntas de discusión planteadas en clase

> Recopilación de las preguntas abiertas del deck; se indica si el propio deck las responde.

| Diap. | Pregunta | ¿Respuesta en el deck? |
|---|---|---|
| 16 | ¿Qué ejemplos cotidianos conocen donde las reglas "cambian con el tiempo" y afectan a un sistema de predicción? | No (ver ejemplos en §3.2, §4.5–4.7) |
| 19 | ¿Por qué la desviación conceptual es inevitable? | **Sí** (diap. 20, §4.2) |
| 22 | ¿Cuál es más difícil de detectar: el cambio de la relación variables–etiquetas o solo el de la distribución de entrada? | No |
| 27 | ¿Qué pasa si ignoramos el drift? | **Sí** (diap. 28, §4.9) |
| 52 | ¿Cómo se comporta ADWIN en el caso de un cambio lento? | **Sí** (misma diap., §6.6) |
| 53 | ¿Por qué detectar un drift lento es más difícil que uno repentino? | **Sí** (diap. 54, §6.7) |
| 57 | ¿Cómo integrarían la detección de concept drift en un ciclo de MLOps? | No (solo el diagrama de §7.1) |

---

## 9. Glosario de términos del curso

> **Nota de organización:** este glosario **reordena y recoge las definiciones tal como las da
> el deck**; no añade teoría nueva. La diapositiva de origen se indica entre paréntesis.

| Término | Definición según el curso |
|---|---|
| **Adaptación** | El modelo se ajusta ligeramente en función de los nuevos datos. (14, 31) |
| **ADWIN (Adaptive Sliding Window)** | Método informado basado en la tasa de error; usa una ventana W que se adapta iterativamente y descarta los elementos más antiguos cuando dos subventanas muestran medias suficientemente distintas. (47, 48, 50–52) |
| **Aprendizaje adaptativo** | Con el tiempo, el modelo se actualiza o se reentrena a sí mismo si es necesario. (32) |
| **Cambio gradual** | Alternancia entre conceptos antes de asentarse; ej.: usuario interesado en finanzas, luego en deportes, que vuelve a consultar finanzas. (25) |
| **Cambios lentos** | Ej.: sensor que se degrada lentamente; ADWIN los detecta de forma progresiva. (25, 52) |
| **Cambios repentinos** | Ej.: cambio de un sensor; ADWIN reduce bruscamente la ventana y los detecta rápido. (25, 52) |
| **Concept drift** | Cómo los datos y relaciones aprendidos en entrenamiento pueden dejar de ser válidos en el tiempo, afectando la validez de las predicciones: $P_{t_0}(X,y) \neq P_{t_1}(X,y)$. (15, 18) |
| **Concept drift real** | Cambios en $P(y \mid X)$; $P(X)$ puede mantenerse constante o no. Cambia lo que determina la etiqueta. (21, 24, 56) |
| **Concepto** | $P(X, y)$. (18) |
| **Conceptos recurrentes** | Conceptos que reaparecen; ej.: patrón estacional en la predicción de ventas. (25, 44) |
| **Contextual (enfoque)** | Ensamble con triggers: particiona los datos en grupos con un modelo por grupo; la nueva instancia se asigna a un grupo. Adecuado para conceptos recurrentes. (33, 44) |
| **Cota de Hoeffding** | Define el umbral $\varepsilon_{cut}$ de ADWIN. (50) |
| **CVFDT** | Concept-Adapting Very Fast Decision Trees; método ciego. (47) |
| **Detectores** | Modelo único con triggers: detectar cambio → descartar datos antiguos → reentrenar → predecir. (33, 39) |
| **Drift virtual** | $P(X)$ cambia sin afectar a $P(y \mid X)$. Cambia la distribución de X, no la relación con Y. (21, 24, 56) |
| **Ensamble dinámico** | Ensamble evolutivo: varios modelos en paralelo combinados por meta-modelo o votación, con recompensa (aumento de peso) o castigo (disminución de peso). (33, 41–42) |
| **Función de costo** | Se usa con una ventana deslizante para identificar cambios en la señal; se compara con un umbral fijo o con una segunda ventana. (49) |
| **Kullback–Leibler / Kolmogorov–Smirnov** | Distancias posibles para la detección basada en la distribución de datos. (48) |
| **Métodos ciegos** | Basados en el olvido: modelo que se adapta de forma incremental o se reentrena con frecuencia. (47) |
| **Métodos informados** | Basados en detectores: detección explícita de cambios, normalmente sobre el error de predicción. (47) |
| **MLOps** | Convierte el ML en un proceso reproducible y mantenible automatizando pruebas, integración, despliegue y monitoreo. (5, 57) |
| **Monitoreo constante** | Seguimiento del flujo de datos, por ejemplo basado en propiedades estadísticas. (13–14) |
| **Olvido** | Modelo único evolutivo: una ventana de entrenamiento fija garantiza el olvido. (33, 36–37) |
| **Page-Hinkley** | Método informado de análisis secuencial basado en la tasa de error. (47, 48) |
| **Reentrenamiento** | El modelo se reentrena completamente desde cero. (14, 31) |
| **Valores atípicos (outliers)** | Riesgo de clasificarlos erróneamente como drift → adaptación falsa. (25) |
| **Ventana deslizante** | Ventana que recorre la señal; base de las funciones de costo y de ADWIN. (49–52) |
| **$\varepsilon_{cut}$** | Umbral de ADWIN para la diferencia de medias entre $W_0$ y $W_1$; definido por la cota de Hoeffding. (50) |

---

## 10. Checklists de síntesis

> **Nota de organización:** esta sección **reorganiza criterios ya enunciados en el deck** para
> facilitar su uso; **no añade teoría nueva**. Cada ítem remite a la diapositiva de origen.

### 10.1 ¿Drift real o virtual? (diap. 21, 23, 24, 56)

- [ ] ¿Cambió **lo que determina la etiqueta**, $P(y \mid X)$? → **drift real**
      (ej.: el usuario ahora busca casas de vacaciones; el gusto musical cambia con la edad).
- [ ] ¿Solo cambió **cómo se ven las entradas**, $P(X)$, con la misma relación? → **drift
      virtual** (ej.: nuevo estilo de redacción; nueva interfaz de la app).

### 10.2 Guía de elección de estrategia adaptativa (diap. 33–44)

| Si necesitas… | Estrategia del curso |
|---|---|
| Un solo modelo que se renueve continuamente, sin decidir cuándo | **Olvido** (ventana fija) |
| Un solo modelo que se reentrene solo cuando se detecta un cambio | **Detectores** |
| Varios modelos en paralelo cuyo peso se ajuste según sus aciertos | **Ensamble dinámico** |
| Manejar **conceptos recurrentes** (estacionalidad) | **Contextual** |

### 10.3 Guía de elección del tipo de detección (diap. 47–48)

- [ ] ¿No quieres detectar el cambio explícitamente? → **método ciego** (olvido, CVFDT).
- [ ] ¿Tienes **etiquetas verdaderas** disponibles? Si no, la detección basada en **tasa de
      error** no es viable (la requiere). (48)
- [ ] ¿Puedes asumir el **costo computacional**? La detección basada en **distribución** es
      computacionalmente intensiva. (48)
- [ ] En la práctica: distribución → **verificaciones de consistencia** (media, varianza);
      tasa de error → **supervisión manual** de la calidad y del impacto en KPIs. (48)

### 10.4 Tipo de cambio → qué esperar (diap. 25, 26, 44, 52, 54)

| Tipo de cambio | Qué dice el curso |
|---|---|
| **Repentino** | ADWIN reduce bruscamente la ventana y detecta rápido; saltos estadísticamente más fáciles de identificar. |
| **Lento** | ADWIN ajusta gradualmente la ventana; se parece al "ruido natural" → más difícil. |
| **Recurrente** | Enfoque **contextual** especialmente adecuado; desafío: *reconocer contextos recurrentes*. |
| **Outliers** | Riesgo de **adaptación falsa**; desafío: *distinguir la desviación del ruido*. |
| **Gradual** | El deck lo ilustra con un ejemplo, pero no le asocia una estrategia específica. |

### 10.5 Los tres recordatorios de cierre (diap. 56)

- [ ] ¿Asumo que el modelo **dejará de ser válido** en algún momento?
- [ ] ¿Distingo si el cambio es **real** o **virtual**?
- [ ] ¿Hay **monitoreo continuo** y un criterio para decidir entre **adaptar** o **reentrenar**?

---

## 11. Notas de fidelidad sobre la reconstrucción

- **Recuento:** el deck tiene **62 diapositivas**. Las diap. 19, 27 y 53 son estados previos
  (solo la pregunta) de las diap. 20, 28 y 54.
- **Pseudocódigo de ADWIN (diap. 50–52):** es una imagen incrustada (*Figure 1: Algorithm
  ADWIN*); se transcribió por lectura visual.
- ⚠ **Discrepancia lógica señalada, no corregida:** la condición de la línea 5 aparece como
  $|\hat{\mu}_{W_0} - \hat{\mu}_{W_1}| \ge \varepsilon_{cut}$, lo que no encaja con la explicación
  de la misma diapositiva (descartar elementos *mientras* las medias difieran). Ver §6.4.
  **[verificar con la docente]**
- **Fórmulas en la capa de texto:** se extraen sin subíndices ni paréntesis (p. ej.
  `𝑃𝑡0 𝑋, 𝑦 ≠ 𝑃𝑡1(𝑋, 𝑦)`); se transcribieron desde la diapositiva rasterizada.
- **Contenido recuperado solo por lectura visual:** la matriz de estrategias (33–45); los
  diagramas de ventanas (36–37, 39), ensambles (41–42) y contextual (44); las formas de señal
  de los cinco tipos de cambio (25); los gráficos de funciones de costo (49) y de ADWIN (50),
  cuyos valores son **aproximados**; el diagrama de MLOps (57).
- **Solapamiento de texto (diap. 25):** el título "Conceptos recurrentes" se superpone a su
  descripción, y la capa de texto los mezcla ("Patrón estacional recurrentes…"); se separaron
  según la lectura visual.
- **Frase truncada (diap. 56):** "**e** requiere monitoreo continuo…" — probablemente "*sino
  que* requiere" o "*y* requiere". Se conserva tal cual. **[verificar]**
- **Título recortado (diap. 49):** "…características de la distribución" queda parcialmente
  tapado por el logo de UTEC; la frase "evalúan **nuestros** saltos" se reporta como aparece.
- **Referencia sin fuente (diap. 49):** los dos gráficos citan "[1]", pero el deck no incluye
  la bibliografía correspondiente.
- **Numeración residual (diap. 50):** el pie muestra el número **"37"**, heredado de otra
  versión del deck; no corresponde a la posición real (50).
- **Imagen vacía (diap. 54):** el recuadro de la derecha ("drift repentino") es un cuadrado
  oscuro sin contenido visible (probablemente un GIF que no se renderizó).
- **Rótulos en inglés dentro de los diagramas:** *Prediction* (41–42); *Detect change, Drop
  old data, Retrain model, Predict, Time* (47); *Performance, Time* (48). En la diap. 50 el
  mismo gráfico conceptual aparece ya en español (*Desempeño, Tiempo*).
- **Terminología variable:** el deck alterna "drift" y "desviación" (p. ej. diap. 27 vs. 28;
  columnas de la diap. 48). También llama "cambio incremental" (50) a lo que en la diap. 25 y
  52 se denomina "cambios lentos" (misma forma de rampa).
- **Colores de los lotes:** el deck no rotula qué representan los lotes oscuros y claros en
  36–44; por su posición en el eje del tiempo se interpretan como **anteriores** y **nuevos**.
- **Flecha "🡪"** (diap. 23): la capa de texto la devuelve como un glifo no reconocido; se
  transcribe como "→".
- **Algoritmos anunciados pero no desarrollados:** Page-Hinkley y CVFDT (ver §6.8).
- **Elementos marcados [verificar]:** la condición de parada de ADWIN (diap. 50) y la frase
  truncada "e requiere" (diap. 56).

---

*Guía de estudio elaborada a partir del deck* Concept Drift · 2026-2 *(62 diapositivas).*
*Curso: **Planificación y Toma de Decisiones en IA**, Dra. Aurea Soriano-Vargas — UTEC, 2026.*
