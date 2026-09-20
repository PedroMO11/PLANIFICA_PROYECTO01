# Semana 6 — Despliegue · Guía de estudio

| Campo | Valor |
|---|---|
| **Curso** | Planificación y Toma de Decisiones en IA |
| **Sesión** | Semana 6 · *Despliegue* · 2026-2 |
| **Docente** | Dra. Aurea Soriano-Vargas (`aureasoriano` · asoriano@utec.edu.pe · `aurea-soriano`) |
| **Institución** | UTEC |
| **Fuente** | `Sem6_Despliegue.pdf` — 92 diapositivas |
| **Atribución del pie** | *Planificación y Toma de Decisiones en IA*, Dr. Aurea Soriano-Vargas (2026) |

> **Propósito de esta guía:** reconstruir fielmente el contenido teórico de las 92 diapositivas
> de la sesión —incluyendo los marcos de decisión, diagramas de arquitectura y ejemplos de
> código que en el PDF están como imágenes— para poder estudiar la sesión sin depender del deck
> original.

---

## 1. Mapa del curso: ubicación de la sesión

Diagrama **"La historia de este curso"** (diapositivas 2, 7 y 88).

```
Fundamentos        ┌──────────────────────┐
                   │ Motivación y         │  (gris = ya visto)
                   │ terminología         │
                   └──────────┬───────────┘
                              │
Ciclo de vida      ┌──────────▼───┐   ┌──────────────┐   ┌───────────┐   ┌──────────────────┐
de la IA           │  Iniciación  │──▶│  Modelado y  │──▶│ Despliegue│──▶│ Desviación       │
                   │   (gris)     │   │  rendimiento │   └───────────┘   │ conceptual       │
                   └──────────────┘   │   (gris)     │    ↑ resaltado    │ (Concept Drift)  │
                                      └──────────────┘      en diap. 7   └────────┬─────────┘
                                                            (ESTA SESIÓN)  ↑ resaltado en
                                                                            diap. 88
                              ┌─────────────────────────────────────────── (PRÓXIMA SEMANA)
IA en sistemas     ┌──────────▼──────┐   ┌──────────────┐   ┌──────────────────┐
                   │    Modelos      │──▶│ Aprendizaje  │──▶│ Colaboración     │
                   │  Fundacionales  │   │ en todo el   │   │ entre humanos    │
                   └─────────────────┘   │   sistema    │   │ e IA             │
                                         └──────────────┘   └────────┬─────────┘
                              ┌────────────────────────────────────────┘
Temas importantes  ┌──────────▼───┐   ┌──────────────────┐
de IA              │ IA Creativa  │──▶│ Ética y equidad  │
                   └──────────────┘   │    en la IA      │
                                      └──────────────────┘
```

> La diapositiva 2 aún resalta *Modelado y rendimiento* (cierre de la sesión previa); la 7 ya
> marca **Despliegue** como sesión en curso; la 88 avanza a **Desviación conceptual (Concept
> Drift)** como próxima semana.

### 1.1 Repaso de la sesión anterior (diap. 3–5)

El deck abre repasando la Semana 5: la curva de error vs complejidad (diap. 3), el diagnóstico
Bias vs Varianza (diap. 4) y las conclusiones sobre evaluación (diap. 5).

---

## 2. Objetivos y agenda de la sesión

### 2.1 Al finalizar la clase deberíamos tener los medios para responder (diap. 6)

1. Comprender el proceso de **despliegue** de un modelo de IA.
2. Comparar diferentes **opciones de implementación y despliegue**.
3. Entender cómo **integrar modelos en aplicaciones** mediante APIs, microservicios, máquinas
   virtuales y contenedores.
4. Aplicar conceptos fundamentales de **MLOps y MLflow** para mejorar la reproducibilidad,
   trazabilidad y seguimiento de experimentos.

### 2.2 Agenda (diap. 9)

1. Del entrenamiento al despliegue
2. Implementación de modelos de IA
3. Arquitecturas de despliegue
4. Plataformas de despliegue
5. Introducción a MLOps
6. MLflow

### 2.3 Ubicación en el ciclo de vida y en el proceso (diap. 10–11)

```
┌────────────┐   ┌───────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│ Iniciación │──▶│ Entrenamiento │──▶│ Estimación │──▶│ Validación │──▶│ Despliegue │
└────────────┘   └───────────────┘   └────────────┘   └────────────┘   └────────────┘
```

**El proceso de esta sesión y la siguiente (diap. 11):**

```
┌──────────────┐   ┌───────────────┐   ┌──────────────┐   ┌──────────────┐
│   Entrenar   │──▶│  Opciones de  │──▶│ Desplegar el │──▶│ Garantizar   │
│ modelo final │   │implementación │   │    modelo    │   │  la validez  │
└──────────────┘   └───────────────┘   └──────────────┘   └──────────────┘
└──────────────── Esta semana ──────────────────────────┘ └─ Próxima semana ─┘
```

---

## 3. Del entrenamiento al despliegue

### 3.1 La brecha en el despliegue de Machine Learning (diap. 12–13)

- El **Machine Learning** puede resolver numerosos problemas prácticos.
- El **valor para el negocio** solo se genera cuando los modelos se despliegan como parte de un
  **sistema de software más amplio**.
- **Cerrar la brecha entre construir modelos y desplegarlos** sigue siendo un desafío importante.

**Gráfico de apoyo (imagen, diap. 13).** Barras de frecuencia sobre el *porcentaje de modelos
que se tenía previsto desplegar y que realmente fueron desplegados*, según una encuesta de
**KDnuggets de 2021**. Eje Y = *Frecuencia*.

| Rango de modelos desplegados | Frecuencia |
|---|---|
| **0 – 20 %** | **65** |
| 21 – 40 % | 25 |
| 41 – 60 % | 8 |
| 61 – 80 % | 8 |
| 81 – 100 % | 8 |

> ✔ **Lectura:** el eje Y son **frecuencias (conteos de respuestas)**, no porcentajes; suman 114
> respuestas. La barra dominante es la de **0–20 %**: la mayoría de las organizaciones despliega
> menos de una quinta parte de los modelos que planeaba desplegar.

### 3.2 Prácticas problemáticas (diap. 14)

- Mucho trabajo **manual y repetitivo**.
- **Seguimiento desordenado** de los experimentos.
- Datos, modelos y, a veces, incluso el código **sin control de versiones**.
- Falta de **pruebas de código y revisiones de código**.
- Enfoque excesivo en **mejorar una única métrica**.
- …

### 3.3 ¿Qué implica la reproducibilidad? (diap. 15)

Viñeta del *caballo de Troya*: el caballo está rotulado **"reproducibilidad"** y en su interior
van tres soldados rotulados **"revisión de código"**, **"pruebas unitarias"** y **"Docker"**. A
las puertas, las figuras **"yo"** e **"investigadores"** lo reciben.

> La idea: pedir reproducibilidad introduce, por la puerta grande, prácticas de ingeniería de
> software que el equipo de investigación no habría adoptado por sí mismo.

### 3.4 ¿Qué pasa antes del entrenamiento final? (diap. 16–17)

- Probamos **múltiples configuraciones**:
  - Distintos algoritmos,
  - Distintas arquitecturas,
  - Distintos hiperparámetros,
- **Separación de datos (train/validation/test)** para estimar rendimiento.
- Para entender qué combinación funciona mejor y para **evitar sobreajuste**.

### 3.5 El entrenamiento final (diap. 18)

> **Se necesitan todos los datos disponibles para finalizar el entrenamiento.**

| | Enfoque |
|---|---|
| **Antes** | Entrenamiento y prueba del modelo para **estimar el rendimiento futuro**. |
| **Ahora** | Usar **todos los datos disponibles** y construir el modelo para su **integración en un entorno productivo**. |

```
ANTES:   ● ──TRAIN──▶ ┐
         ● ──TRAIN──▶ ├──▶ modelo (para estimar rendimiento)
         ● ──TEST───▶ ┘

AHORA:  ┌●┐
        │●│ ──TRAIN──▶ modelo definitivo (para producción)
        └●┘
```

### 3.6 El modelo "definitivo" (diap. 19)

- Este es el **artefacto que será empaquetado y desplegado** en el entorno productivo.
- Se documenta junto con los **parámetros elegidos**, **métricas de referencia** y el **pipeline
  de preprocesamiento** usado.

---

## 4. Opciones de Implementación (sección, diap. 20–26)

### 4.1 El rol del modelo define el nivel de autonomía (diap. 21)

> Los modelos no siempre cumplen el mismo rol:
> - Algunos solo **predicen** resultados.
> - Otros **recomiendan** decisiones.
> - En casos más avanzados, la IA puede **ejecutar acciones** directamente en el mundo real.
>
> Esto define el **nivel de autonomía** del sistema y condiciona los requisitos técnicos
> (sensores, actuadores, monitoreo).

### 4.2 Los cuatro niveles de implementación (diap. 22)

*(Tabla que en el PDF está íntegramente como imagen.)*

| | **1 · IA predictiva** | **2 · IA recomendadora** | **3 · IA semiautónoma** | **4 · IA autónoma** |
|---|---|---|---|---|
| **Qué hace** | Genera predicciones sobre un resultado. | Sugiere acciones u opciones para que el usuario decida. | Ejecuta algunas acciones de forma automática, pero requiere supervisión humana en ciertos casos. | Decide y actúa directamente en el entorno, sin intervención humana inmediata. |
| **Ejemplos** | • Predecir el precio de una vivienda.<br>• Clasificar si un email es spam o no. | • Recomendar productos en Amazon.<br>• Sugerir películas en Netflix. | • Sistema de control de inventario que ordena más stock si el nivel es bajo.<br>• Moderación automática de contenido con revisión humana en casos críticos. | • Vehículo autónomo que toma todas las decisiones de conducción en tiempo real.<br>• Agente de IA que ejecuta tareas en un entorno digital (ej. gestionar reservas, realizar transacciones). |
| **Salida** | Una predicción numérica o categórica. | Una lista de recomendaciones o sugerencias. | Acciones concretas en algunos contextos, con intervención humana en casos críticos. | Ejecución de acciones en el entorno físico o digital sin intervención humana inmediata. |

> Eje inferior del diagrama: **Menor autonomía ──────────▶ Mayor autonomía**

### 4.3 Qué implica esta distinción (diap. 23)

| | **1 · IA predictiva** | **2 · IA recomendadora** | **3 · IA semiautónoma** | **4 · IA autónoma** |
|---|---|---|---|---|
| **Descripción** | Predice un resultado; la persona decide qué hacer. | Sugiere opciones o acciones al usuario. | Ejecuta algunas acciones, con supervisión humana. | Percibe, decide y actúa por sí sola. |
| **Componentes técnicos** | • Modelo<br>• Datos de entrada<br>• Interfaz de salida | • Modelos<br>• Filtros<br>• Contexto del usuario | • Modelo o reglas<br>• Integración con APIs<br>• Sistemas de control | • Sensores<br>• Actuadores<br>• Control en tiempo real |
| **Ejemplo** | Predicción de precios o detección de spam | Recomendaciones en Netflix o Amazon | Reposición automática de inventario | Robot, auto o máquina industrial |

> Eje inferior: **Menor complejidad técnica ──────────▶ Mayor complejidad técnica**
>
> **⚠ Responsabilidad y riesgo:**
> - Un recomendador que falla **no suele ser grave**.
> - Un auto autónomo que falla **puede ser letal**.

### 4.4 Desafíos técnicos de la implementación (diap. 24–26)

| Desafío | Diapositiva |
|---|---|
| **Múltiples lenguajes de programación** | 24 |
| **Uso paralelo de GPU** | 25 |
| **Costos impredecibles** | 26 |

---

## 5. APIs (diap. 27–31)

### 5.1 Qué es una API (diap. 29)

- La API **expresa un componente (de servicio)** en términos de sus **funcionalidades, entradas,
  salidas y tipos subyacentes**.
- Ejemplo con ChatGPT:

```python
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 20..."},
        {"role": "user", "content": "Where was it played?"}
    ]
)
```

> *(La línea del mensaje `assistant` aparece **cortada por el borde del recuadro de código** en
> la diapositiva; se transcribe hasta donde es legible.)* **[verificar]**

### 5.2 Una breve mirada a la historia (diap. 30)

| Hito | Descripción |
|---|---|
| **Sales Force Automation API** | *first complicated XML API* — **2000** |
| **Roy Fielding** | *originator of REST* |

> Nota del pie de la diapositiva: **"No incluye las APIs no públicas."**

### 5.3 Diseña tu API personalizada (diap. 31)

```
                      ┌──────────────────────────────────────┐
  Desarrollo Web      │  Python App                          │
        ⇄  R E S T  ⇄ │  Endpoint: /v1/xyz/prediction        │
                      │                    ┌────────┐        │
                      │                    │ modelo │        │
                      │                    └────────┘        │
                      └──────────────────────────────────────┘
```

**Pasos a implementar:**
- Cargar modelo ML
- Crear ruta
- Preparar datos
- Ejecutar predicción
- Postprocesamiento
- Obtener resultado

---

## 6. Opciones de Despliegue: el marco de las cuatro decisiones (diap. 32–45)

### 6.1 El marco completo (diap. 33)

*(Marco que en el PDF está íntegramente como imagen; el deck lo recorre resaltando una columna
por vez.)*

| | **1 · Encapsulamiento del modelo** | **2 · Entorno físico de ejecución** | **3 · Plataforma de despliegue** | **4 · Portabilidad y escalabilidad** |
|---|---|---|---|---|
| **Pregunta guía** | ¿Cómo organizamos los componentes del modelo? | ¿Dónde se ejecuta el modelo? | ¿Qué nivel de servicio y control necesitas? | ¿Cómo asegurar que el modelo pueda crecer y moverse? |
| **Opciones** | **Monolítico:** una sola aplicación que integra API, preprocesamiento, modelo y postprocesamiento (Aplicación única → Modelo → Predicción).<br><br>**Microservicios:** componentes independientes que se comunican entre sí (API → Preprocesamiento → Servicio del modelo → Resultado). | **Instancia única:** un solo servidor (servidor físico o en la nube).<br><br>**Máquina virtual (VM):** servidor con capa de virtualización (Aplicación/Modelo → Sistema operativo (SO) → Capa de virtualización → Hardware).<br><br>**Contenedor (Docker):** paquete ligero y portable con todo lo necesario (Código + modelo + dependencias). | **IaaS** — *Más control*: gestionas SO, librerías y aplicación.<br>**PaaS** — *Equilibrio*: despliegas tu app sin gestionar infraestructura.<br>**FaaS / Serverless** — *Pago por uso*: funciones bajo demanda con escalado automático.<br>**MLaaS / SaaS** — *Más abstraído*: servicios gestionados para entrenar y desplegar modelos. | **Docker:** empaqueta la aplicación/modelo (Código + modelo + dependencias).<br><br>**Kubernetes / Cloud Foundry:** gestionan y despliegan contenedores (contenedores de la aplicación / modelo). |
| **Síntesis** | **Monolítico** = simplicidad \| **Microservicios** = flexibilidad y escalado independiente. | Los contenedores mejoran **portabilidad, consistencia y eficiencia**. | **Más control ──▶ Más abstracción** | • Portabilidad<br>• Escalado automático<br>• Alta disponibilidad<br>• Uso eficiente de recursos |

> 💡 **Idea clave (pie del marco):** *Desplegar IA implica decidir cómo **encapsular el modelo**,
> dónde **ejecutarlo** y qué **nivel de control o automatización** necesita la operación.*

**Orden de recorrido en el deck:** la diapositiva 34 resalta la columna **1**, la 37 la columna
**3** y la 40 la columna **2**. Entre ellas se intercalan las láminas de detalle (§6.2 a §6.6).
La columna **4** no recibe una diapositiva de resalte propia.

### 6.2 Programación orientada a servicios y dirigida por modelos (diap. 35)

Progresión de granularidad, ilustrada con el ejemplo de una **transacción bancaria**
(*Verificar ID · Verificar saldo · Realizar transacción · Dar recibo*):

| Etapa | Descripción del deck | Granularidad |
|---|---|---|
| **Programa "Monolítico"** | Los cuatro pasos viven dentro de un único bloque. | **Un Componente** |
| **Programación usando componentes reutilizables y preprogramados (servicios)** | Menos programación, más uso de componentes (servicios). | **Dos componentes** |
| **Uso extensivo de componentes (servicios)** | La programación se convierte en **"modelado"**: se compone a partir de servicios existentes (*Pedido A*, *Pedido B*). | **Muchos componentes** |

### 6.3 ¿Qué son los Microservicios? (diap. 36)

| **Arquitectura Monolítica** | **Arquitectura de Microservicios** |
|---|---|
| Interfaz de usuario → Lógica del Negocio → Interfaz de datos → **Base de datos** (una sola) | Interfaz de usuario → varios **Microservicios**, cada uno con su propia **Base de datos** |
| **Un solo bloque; simple, pero difícil de escalar.** | **Componentes independientes; más escalables y flexibles, ideales para pipelines de ML.** |

### 6.4 Comparación general de plataformas (diap. 39)

| OPCIÓN | NIVEL DE CONTROL | NIVEL DE ABSTRACCIÓN | VENTAJAS | DESVENTAJAS | CASOS DE USO |
|---|---|---|---|---|---|
| **On-premise** | 🔵 Muy alto | 🟢 Muy bajo | Máximo control, privacidad y personalización | Alto costo y mantenimiento | Datos sensibles, banca, salud, gobierno |
| **IaaS** | 🔵 Alto | 🟠 Bajo | Flexible, escalable, control de infraestructura | Requiere administración técnica | Modelos personalizados, GPUs, workloads complejos |
| **PaaS** | 🟡 Medio | 🟡 Medio | Despliegue rápido, menos gestión | Menor control de infraestructura | APIs de ML, aplicaciones web con IA |
| **FaaS / Serverless** | 🟠 Bajo | 🔵 Alto | Escalado automático, pago por uso | Límites de tiempo/recursos, latencia de arranque | Inferencia ocasional, eventos, tareas cortas |
| **SaaS** | 🟢 Muy bajo | 🔵 Muy alto | Fácil y rápido, casi sin mantenimiento | Poca personalización y dependencia del proveedor | IA lista para usar: chatbots, visión, traducción |

### 6.5 Máquinas Virtuales, Contenedores y Kubernetes (diap. 41–43)

> **¿Cómo encapsular el entorno de despliegue respecto a los componentes físicos?** (diap. 41)
> Microservicios · Instancia única · Máquina Virtual · Contenedor (e.g. Docker)

**Comparación (diap. 43):**

| | **Máquinas Virtuales (VM)** | **Contenedores** |
|---|---|---|
| **Pila** | App → Bins/Libraries → Sistema Operativo → Máquina Virtual → **Hipervisor** → Sistema Operativo (host) → Hardware | App 1 / App 2 → Bins/Libraries → Contenedor → **Motor de contenedores (ej. Docker)** → Sistema Operativo (host) → Hardware |
| **Características** | • Simulan un computador completo, con su propio SO, librerías y aplicaciones.<br>• Se ejecutan sobre un hipervisor. | • Encapsulan solo la aplicación y sus dependencias.<br>• Comparten el kernel del SO del host. |
| **Ventajas** | ✅ Aislamiento fuerte.<br>✅ Permiten diferentes SOs (Windows, Linux). | ✅ Ligeros y eficientes.<br>✅ Portátiles.<br>✅ Arranque rápido.<br>✅ Fáciles de escalar y replicar. |
| **Desventajas** | ⛔ Pesadas (SO completo).<br>⛔ Mayor consumo de CPU y memoria.<br>⛔ Tardan más en arrancar. | ⛔ Aislamiento menor que una VM.<br>⛔ Requiere buena gestión de seguridad. |

**Despliegue con Kubernetes (diap. 43, panel derecho):**

- Kubernetes es un sistema para la **gestión de contenedores**.
- Permite el **despliegue, mantenimiento y escalado** de aplicaciones contenerizadas.
- **Conceptos clave:**
  - **Pods:** agrupan contenedores.
  - **Nodos:** agrupan pods.
  - **Plano Maestro:** controla el clúster y programa las tareas.

```
        ┌─────────────────────────────┐
        │  Plano Maestro              │
        │  (Controla el clúster)      │
        └──────┬───────┬───────┬──────┘
          ┌────▼──┐ ┌──▼────┐ ┌▼──────┐
          │ Nodo  │ │ Nodo  │ │ Nodo  │
          │ ┌───┐ │ │ ┌───┐ │ │ ┌───┐ │
          │ │Pod│ │ │ │Pod│ │ │ │Pod│ │
          │ └───┘ │ │ └───┘ │ │ └───┘ │
          └───────┘ └───────┘ └───────┘
             Contenedores en cada Pod
```

### 6.6 Cloud Foundry y servicios gestionados (diap. 44–45)

**Cloud Foundry (PaaS)** — plataforma para desplegar y escalar aplicaciones sin gestionar
directamente la infraestructura:
- Integra servicios externos como bases de datos y almacenamiento.
- Soporta múltiples lenguajes y frameworks.
- Ofrece escalado automático.
- Incluye logging y monitoreo.

> Referencia citada: **Bernstein (2014)**, *Cloud foundry aims to become the OpenStack of PaaS*.

**Despliegue con servicios gestionados (MLaaS = SaaS) (diap. 45):**
- **Azure Machine Learning services**
- **Amazon SageMaker**
- **IBM Watson Studio**

---

## 7. MLOps (sección, diap. 46–62)

### 7.1 La raíz de todos los males (diap. 47–48)

El ciclo **CRISP-DM** (*Comprensión del negocio · Comprensión de los datos · Preparación de datos
· Modelado · Evaluación · Despliegue*, con **Datos** en el centro) aparece con un **"????"**
marcando el tramo que va del **Despliegue** de vuelta a la **Comprensión del negocio** — el
punto donde el ciclo se rompe en la práctica.

Acompaña el meme: *"Worked fine in Jupyter — now it's Ops problem"*.

### 7.2 Mientras tanto, en desarrollo de software: DevOps (diap. 49)

> **DevOps (Development + Operations):** Conjunto de prácticas, herramientas y cultura orientadas
> a desarrollar software **de mejor calidad y más rápidamente**, reduciendo la brecha entre los
> equipos de desarrollo y operaciones.

### 7.3 Ciclo automatizado de desarrollo de software (diap. 50)

| Etapa | Contenido |
|---|---|
| **Etapa 1 – Desarrollo Continuo** | Planificación · Codificación |
| **Etapa 2 – Integración Continua** | Pruebas pequeñas y rápidas · Compilación |
| **Etapa 3 – Pruebas Continuas** | Pruebas más exhaustivas · Validación · Liberación de versión |
| **Etapa 4 – Entrega Continua** | Empaquetado final · Despliegue y pruebas en un servidor operativo |
| **Etapa 5 – Monitoreo Continuo** | Recopilar datos, monitorear cada función y detectar errores · Retroalimentación de los usuarios finales |

### 7.4 Qué es MLOps (diap. 51–52)

> **MLOps (Machine Learning + Operations):**
> - Conjunto de prácticas orientadas a **desarrollar, desplegar y mantener modelos de Machine
>   Learning en producción** de forma rápida y confiable.
> - Promueve una mayor **comunicación y colaboración** entre ingenieros de datos, científicos de
>   datos y profesionales de operaciones.

La diapositiva 52 muestra la curva de **Interés en MLOps (por Google Trends)**.

### 7.5 El principal desafío de MLOps (diap. 53–54)

- Las soluciones de Machine Learning son una **combinación estrechamente interconectada de
  datos, modelos y código**.
- Los datos cambian constantemente, pero también cambian las **necesidades del negocio**.
- Esto conduce al principio: **"Cambiar algo cambia todo"** (*Changing Anything Changes
  Everything*).
- Es un desafío importante, pero **MLOps lo asume como una realidad** y diseña procesos para
  gestionarlo.

**Diagrama de los tres planos entrelazados (diap. 54).** Seis etapas en la fila superior, con
tres carriles de artefactos por debajo:

```
Construcción → Evaluación y    → Puesta en   → Pruebas → Despliegue → Monitoreo y
del modelo     experimentación   producción                            observabilidad
               del modelo        del modelo

CÓDIGO │ código de        │              │        │ código   │ código de   │
       │ entrenamiento    │              │        │ de prueba│ aplicación  │
───────┼──────────────────┼──────────────┼────────┼──────────┼─────────────┤
MODELO │ modelos          │              │ modelo │ modelo   │ código y    │
       │ candidatos       │              │ elegido│          │ modelo en   │
       │                  │              │        │          │ producción  │
───────┼──────────────────┼──────────────┼────────┼──────────┼─────────────┤
DATOS  │ datos de         │ datos de     │        │ datos de │ datos de    │
       │ entrenamiento    │ prueba /     │        │ prueba   │ producción  │
       │                  │ métricas     │        │          │             │
       │                  │ de prueba    │        │          │             │
```

> Una flecha superior cierra el ciclo desde *Monitoreo y observabilidad* de vuelta a
> *Construcción del modelo*.

### 7.6 Desacoplando en distintos pipelines (diap. 55)

*(Diagrama íntegramente como imagen; transcrito de la lectura ampliada.)*

**A · Pipeline de datos**

```
Fuentes de datos ──▶ Exploración y ──▶ Preparación ──▶ [DATOS] ──┬──▶ ENTRENAR
 • Bases de datos      validación         y limpieza              └──▶ PROBAR
 • APIs / Servicios
 • Archivos (Logs, CSV, etc.)
 • ...
```

| Etapa | Contenido |
|---|---|
| **Exploración y validación** | • Perfilado de datos<br>• Validación de calidad<br>• Detección de sesgos<br>• División train/test |
| **Preparación y limpieza** | • Limpieza de datos<br>• Transformaciones<br>• Ingeniería de variables<br>• Unificación de datos |
| **DATOS** | • Versionado de datos (p. ej. **DVC, LakeFS**) |

**B · Pipeline de aprendizaje automático**

```
Ingeniería ──▶ Evaluación ──▶ Empaquetado ──▶ [MODELO] ──▶ Servicio del modelo
del modelo     del modelo     del modelo
                                   └┄┄▶ Versionado del modelo (p. ej. MLflow)
```

| Etapa | Contenido |
|---|---|
| **Ingeniería del modelo** | • Ingeniería de variables (features)<br>• Selección de algoritmos<br>• Ajuste de hiperparámetros<br>• Validación cruzada |
| **Evaluación del modelo** | • Selección del mejor modelo<br>• Métricas: exactitud, precisión, recall, F1<br>• Pruebas en conjunto de test<br>• Análisis de errores |
| **Empaquetado del modelo** | • Formato del modelo (p. ej. **ONNX, pickle**)<br>• Artefactos y metadatos<br>• Versionado del modelo<br>• Contenedorización (**Docker**) |
| **MODELO** | • Servicio del modelo<br>• API (p. ej. **Flask**)<br>• Despliegue en contenedor (**Docker**)<br>• Orquestación (**Kubernetes**) |

**C · Pipeline de código de software**

```
[CÓDIGO] ──▶ Construcción y pruebas ──▶ Despliegue
              de integración               en producción
```

| Etapa | Contenido |
|---|---|
| **CÓDIGO** | • Desarrollo en rama principal (*trunk-based development*)<br>• Control de versiones (**Git**)<br>• Pruebas unitarias y de integración |
| **Construcción y pruebas de integración** | • Build y tests automatizados<br>• Integración con el modelo<br>• Construcción de imagen (**Docker**)<br>• Pruebas end-to-end |
| **Despliegue en producción** | • Despliegue automatizado (**CI/CD**)<br>• **Docker / Kubernetes**<br>• Infraestructura como código (**IaC**)<br>• Entornos (staging / producción) |

**D · Monitoreo y registro** (cierra el ciclo)

- Rendimiento del modelo
- Detección de deriva (*data/model*)
- Alertas y logs

Salidas del bloque de monitoreo:
- **Disparo de reentrenamiento**
- **Métricas en producción**
- **Observabilidad** (logs, trazas, etc.)
- Flecha de retorno: **"nuevos datos a partir del rendimiento del modelo"** → vuelve al pipeline
  de datos.
- Flecha de retorno: **"Mejores modelos gracias a mejores datos"** → hacia el despliegue.

> Nota manuscrita del diagrama: **"Datos + Modelos + Código = Valor en producción"**.

### 7.7 MLOps = Principios DevOps aplicados a soluciones ML (diap. 56–57)

| Principio | Cómo cambia respecto de DevOps clásico |
|---|---|
| **Integración Continua** | Ya no se limita a probar y validar código, sino que también abarca la prueba y validación de **datos, esquemas de datos y modelos**. |
| **Desarrollo continuo** | Ya no se centra en un único paquete de software, sino en un **sistema (un pipeline de ML)** que debe desplegar automáticamente otro servicio. |
| **Entrenamiento Continuo** | **Reentrena automáticamente** modelos con nuevos datos y permite comparar nuevas versiones con modelos anteriores. |
| **Entrega Continua (CD)** | Automatiza la **construcción, pruebas y despliegue** de pipelines y modelos de Machine Learning. |
| **Monitoreo Continuo (CM)** | Supervisa los sistemas en producción para detectar **degradación del modelo, cambios en los datos, errores o necesidad de reentrenamiento**. |
| **Documentación Continua** | Genera y mantiene actualizada automáticamente la documentación de **modelos, datos, experimentos y versiones**. |

### 7.8 Los cinco principios MLOps (diap. 58–62)

#### 7.8.1 Versionar todo (diap. 58)

- **Artefacto:** cualquier **resultado tangible** generado durante el desarrollo de software.
- **Ejemplos comunes:** datos, código, modelos, entornos y documentación.
- Siempre que sea posible, los artefactos deben **almacenarse y versionarse**.

**Esto es la base para:**

| Beneficio | Definición del curso |
|---|---|
| **Reproducibilidad** | Volver a ejecutar experimentos y obtener los mismos resultados. |
| **Trazabilidad** | Identificar qué código, datos o configuración generaron cada resultado. |
| **Colaboración** | Permitir que varias personas trabajen con los mismos artefactos. |

#### 7.8.2 Automatizar siempre que sea posible (diap. 59)

- La automatización es clave para habilitar los **procesos continuos** de MLOps.
- Puede aplicarse en **todas las etapas** del flujo de trabajo.
- Reduce tareas **manuales, repetitivas y propensas a errores**.
- Idealmente, se automatiza todo lo posible y luego se **reincorpora la intervención humana de
  forma estratégica**.
- Así, los equipos pueden concentrarse en **generar valor**, en lugar de mantener procesos
  manuales.

> Referencia de la viñeta: `https://xkcd.com/`

#### 7.8.3 Si no está probado, está roto (diap. 60)

**Buenas prácticas:**
- Exigir que **pasen los tests antes de integrar** el código.
- Probar **todo el sistema**, no solo el modelo.
- Probar **distintos artefactos**, no únicamente el código.
- Utilizar **diversos enfoques de testing**.

#### 7.8.4 Reutilización (diap. 61)

> Capacidad de utilizar un modelo o fragmento de código para **distintos propósitos y en
> diferentes entornos**.

**Buenas prácticas que favorecen la reutilización:**
- **Modularidad:** dividir el sistema en componentes independientes.
- **Herramientas agnósticas e interoperables:** evitar soluciones que no se integren bien con
  otras tecnologías.
- **Código y artefactos fáciles de localizar:** mantener una forma clara y consistente de
  encontrar lo que se necesita.

#### 7.8.5 Colaboración y comunicación (diap. 62)

**Buenas prácticas:**
- **Comunicación frecuente** entre todos los roles involucrados en el flujo de trabajo de MLOps.
- **Transparencia:** compartir con todo el equipo información sobre experimentos, resultados,
  modelos y despliegues.

---

## 8. Herramientas MLOps (diap. 63–66)

### 8.1 Las tres familias de herramientas (diap. 64–65)

**Herramientas DEV:**

| Herramienta | Definición del curso |
|---|---|
| **Repositorio de código** | Ubicación central donde se almacena y gestiona el código. |
| **Pipelines de CI/CD** | Automatizan la construcción, prueba y despliegue de cambios en el código. |
| **Monitoreo de modelos / datos / aplicaciones** | Registro, trazabilidad y supervisión de modelos de Machine Learning, datos y aplicaciones. |

**Herramientas ML:**

| Herramienta | Definición del curso |
|---|---|
| **Análisis de datos** | Lenguajes de programación e IDEs para explorar, limpiar y analizar datos. |
| **Entrenamiento de modelos** | Frameworks de Machine Learning para entrenar, validar y ajustar modelos. |
| **Despliegue de modelos** | Entornos de ejecución para poner modelos en producción. |
| **Servicio de predicciones** | Permite ofrecer predicciones a partir de modelos desplegados. |

**Herramientas MLOps específicas:**

| Herramienta | Definición del curso |
|---|---|
| **Versionado de datos** | Permite rastrear y gestionar los cambios en los datos a lo largo del tiempo. |
| **Feature store** | Repositorio central para almacenar, versionar y compartir características (*features*). |
| **Metadata store** | Repositorio central para almacenar y gestionar los metadatos recolectados durante el ciclo de vida del modelo. |

### 8.2 El ecosistema cambia constantemente (diap. 66)

> El ecosistema de herramientas que soportan MLOps **cambia constantemente**.

| Recurso | URL |
|---|---|
| **LF AI & Data Landscape** (panorama interactivo y curado) | `https://landscape.lfai.foundation/` |
| **Awesome Production Machine Learning** | `https://github.com/EthicalML/awesome-production-machine-learning` |
| **MLOps Stack Canvas** (plantilla de una página con preguntas para diseñar la arquitectura del sistema) | `https://miro.com/miroverse/mlops-stack-canvas/` |

---

## 9. MLOps workflows (diap. 67–73)

### 9.1 Gestión de Datos (diap. 68)

```
 Fuentes ──▶ Data ingestion ──▶ Data Preprocesing ──┬──▶ Data labelling ──┐
 de datos     [Unit]              [Expectation][Unit]│      [Labelling]     │
                                                     └─────────────────────▶ Central
                                                                             Data Store
                                                                                 │
                                                                                 ▼
                                                                      Feature Engineering [Unit]
                                                                                 │
                                                                                 ▼
                                                                          Feature Store
```

Las etiquetas moradas indican el **tipo de prueba o control** asociado a cada paso: *Unit*,
*Expectation*, *Labelling*.

> Herramienta citada para validación de datos: **Great Expectations**
> (`https://docs.greatexpectations.io/`) — *"Data validation with Great Expectations"*, cuyas
> salidas son: datos de alta calidad en los productos de datos, documentación de datos e
> informes de calidad, y *logging & alerting*.

### 9.2 Desarrollo del Modelo (diap. 69)

```
[Data drift] Central Data Storage ──▶ Data exploration / Feature engineering ──┬──▶ Test ──▶ Model testing ──▶ Model packaging
[Feature drift] Feature Store ────────────────────────────────────────────────┴──▶ Train ──▶ Model training ──▶ (AutoML)
                                                                                              Hyperparam tuning
                                                                                                    │
                                                                                                    ▼
                                                                                             Metadata Store

Model packaging ──┬──▶ Serialised File (ONNX)
                  └──▶ Docker image           ──▶ Model Registry
```

**Controles asociados (etiquetas moradas):**
- Sobre *Model testing* y *Model compression*: **Robustness test · Fairness · Behavior ·
  Slices & edges · Performance**
- Sobre *Model training*: **Infra test · Training test**

> **Model compression:** Quantization, Distillation, Pruning.

### 9.3 Elección de una estrategia de despliegue del modelo (diap. 70)

*(Tabla que en el PDF está íntegramente como imagen — es uno de los contenidos centrales de la
sesión.)*

| # | Estrategia | Qué hace | Arquitectura | ✅ Ventajas | ❌ Desventajas |
|---|---|---|---|---|---|
| **1** | **Predicción por lotes** *(Ejecución programada)* | Ejecutar el modelo periódicamente y guardar resultados en una base de datos | Usuario/Aplicación ⇄ Servidores de procesamiento *(con modelo)* ⇄ Base de datos de resultados | • simple<br>• baja latencia en consulta<br>• buena escalabilidad | • resultados desactualizados |
| **2** | **Predicción en el edge** | Desplegar el modelo en el **dispositivo del usuario** | Dispositivo del usuario *(con modelo)* ⇄ Servidor/Backend ⇄ Base de datos | • baja latencia<br>• funciona sin conexión<br>• mayor privacidad | • recursos limitados<br>• marcos de trabajo específicos<br>• actualizaciones más complejas<br>• monitoreo más difícil |
| **3** | **Modelo en servicio** | Desplegar el modelo en el **servidor de la aplicación** | Usuario/Aplicación ⇄ Servidor de aplicación *(con modelo)* ⇄ Base de datos | • reutiliza infraestructura | • problemas de compatibilidad (código, hardware, escala)<br>• mayor consumo de recursos |
| **4** | **Modelo como servicio** | Desplegar el modelo como un **servicio independiente** | Usuario/Aplicación ⇄ Servidor de aplicación ⇄ Base de datos, con un **Servidor del modelo** aparte | • confiable<br>• escalable<br>• flexible | • más latencia<br>• más infraestructura<br>• más trabajo de DevOps |
| **5** | **Modelo como función serverless** | Desplegar el modelo como una **función bajo demanda** | Usuario/Aplicación ⇄ Servidor de aplicación ⇄ Base de datos, con una **Función serverless** *(con modelo)* | • solo se paga por cómputo<br>• sin gestionar infraestructura | • límites de tamaño del paquete<br>• límites de servidores genéricos |

> 💡 **Idea clave:** *no existe una única estrategia mejor, la elección depende de los requisitos
> del caso de uso (latencia, costo, privacidad, escala, etc.).*

### 9.4 Despliegue en producción y servicio de predicciones (diap. 71)

*(Continúa el flujo de la sección; ver el esquema de §9.2 y las estrategias de §9.3.)*

### 9.5 Monitoreo y bucles de retroalimentación (diap. 72)

```
 [Data drift]  ┌──────────┐                              ┌────────────────────┐
 [Integrity]   │ Baseline │                              │ Alerts             │
               │   data   │ ──────────────────────────▶  │ Model rollback     │
               ├──────────┤                         ┌──▶ │ Model retraining   │
               │   New    │                         │    └────────────────────┘
               │   data   │ ──┐                     │
               └──────────┘   │   [Model drift] [Business metrics]
                              │   [Fairness]    [App performance]
                              │                      │
                              └──▶ New model in production env

                              Metadata Store
```

Las etiquetas moradas son las **dimensiones monitoreadas**: *Data drift*, *Integrity*, *Model
drift*, *Business metrics*, *Fairness*, *App performance*. Las salidas en rojo son las
**acciones**: *Alerts*, *Model rollback*, *Model retraining*.

### 9.6 Ejemplo de MLOps workflow (diap. 73)

| Función | Herramienta |
|---|---|
| **EXPERIMENT TRACKING** | MLflow |
| **EXPERIMENTATION** | Jupyter |
| **DATA VERSIONING** | DVC |
| **CODE VERSIONING** | Git |
| **PIPELINE ORCHESTRATION** | Apache Airflow |
| **MODEL REGISTRY** | MLflow |
| **MODEL SERVING** | BentoML |
| **ARTIFACT TRACKING** | MLflow |
| **MODEL MONITORING** | Prometheus · Grafana |

---

## 10. MLflow (diap. 74–84)

### 10.1 Por qué hace falta (diap. 74)

> **Ejecutar experimentos de ML = entorno + datos + código.**
>
> **Registrar:** Hiperparámetros + Resultados + Gráficos / visualizaciones

### 10.2 Componentes de MLflow (diap. 75)

| Componente | Definición del curso |
|---|---|
| **Seguimiento de experimentos** | Registra y permite consultar experimentos: código, datos, configuración y resultados. |
| **Proyectos** | Empaqueta el código de ciencia de datos en un formato que permite ejecuciones reproducibles en múltiples plataformas. |
| **Registro de modelos** | Almacena, anota y gestiona modelos en un repositorio central. |
| **Modelos** | Despliega modelos de aprendizaje automático en diversos entornos de servicio de predicciones. |

### 10.3 Visión general de MLflow (diap. 76)

```
┌──────────────────────────┐        ┌──────────────────────────────────────────────┐
│  Servidor de seguimiento │        │            Registro de modelos               │
│                          │        │  Científicos de datos │ Ingenieros de despliegue│
│  • Parámetros            │        ├───────────┬──────────┬────────────┬──────────┤
│  • Métricas              │ ─────▶ │Experimental│ Staging │ Producción │ Archivado│
│  • Artefactos            │        │           │          │            │          │
│  • Metadatos             │        │  (modelos circulando entre etapas)           │
│  • Modelos               │        └──────────────────────────────────────────────┘
└──────────────────────────┘
```

Las cuatro etapas del ciclo de vida de un modelo registrado son: **Experimental → Staging →
Producción → Archivado**. Los *Científicos de datos* operan sobre las primeras etapas y los
*Ingenieros de despliegue* sobre las últimas.

### 10.4 Configuración de MLflow (diap. 77)

**Instalar mlflow con pip:**

```bash
python -m pip install mlflow
```

**Establecer una ubicación para el servidor de tracking** (local filesystem, db, remote server):

```python
import mlflow
mlflow.set_tracking_uri("file:///Users/aurea/mlruns")   # in every script / notebook
```

**O mejor defínelo en tu sistema:**

```bash
export MLFLOW_TRACKING_URI="/Users/aurea/mlruns"        # in your .bashrc/.zshrc
```

**Inicia un servidor de MLflow desde una nueva terminal** para permitir el acceso a la interfaz
gráfica (UI):

```bash
mlflow ui --backend-store-uri file:///Users/aurea/mlruns
```

### 10.5 Seguimiento de experimentos (diap. 78)

**Define un nombre de experimento para organizar las ejecuciones:**

```python
import mlflow
mlflow.set_experiment("ExperimentABC")
```

**Luego ejecuta el código dentro de un gestor de contexto `mlflow.start_run()`:**

```python
with mlflow.start_run() as run:
    lr = 0.1
    mlflow.log_param("learning_rate", lr)
    ...
    mlflow.log_metric("accuracy", 0.92)
```

> Cada ejecución (*run*) registrará en MLflow Tracking información como: **Código, Parámetros,
> Métricas**. También es posible definir un nombre para cada ejecución y crear ejecuciones
> anidadas (*nested runs*).

### 10.6 Registrar parámetros, métricas y modelos (diap. 79)

**Parámetros** — pares clave–valor utilizados para registrar hiperparámetros, configuraciones,
etc.:

```python
mlflow.log_param("num_layers", 3)
mlflow.log_params({"learning_rate": 0.001, "epochs": 20})
```

**Métricas** — permiten registrar valores de evaluación como pérdida (*loss*) o exactitud
(*accuracy*). Estas métricas pueden visualizarse en la interfaz de MLflow para comparar
diferentes ejecuciones:

```python
mlflow.log_metric("loss", 0.45)
```

**Modelos** — también es posible registrar la arquitectura y los pesos del modelo en formato
**MLModel**:

```python
mlflow.tensorflow.log_model(model, "Name_for_model")
```

### 10.7 Ejemplo completo con registro manual (diap. 80)

> *"Ejemplo completo con registro manual (mejores resultados)"*

```python
import tensorflow.keras as tfk
import mlflow
import mlflow.tensorflow

mlflow.set_experiment("ExperimentABC")

model = tfk.models.Sequential()
model.add(tfk.layers.Dense(64, activation='relu', input_shape=(10,)))
model.add(tfk.layers.Dense(1, activation='sigmoid'))

learning_rate = 0.01
model.compile(
    optimizer=tfk.optimizers.Adam(lr=learning_rate, decay=0.1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

mlflow.tensorflow.log_model(model, "model_name")   # <----- ARQUITECTURA

with mlflow.start_run() as run:
    model.fit(X_train, y_train, epochs=5)
    mlflow.log_metric("loss", model.evaluate(CX_test, y_test)[0])   # <----- SCORE
    mlflow.log_param("learning_rate", learning_rate)                # <----- HIPERPARÁMETRO
```

> ⚠ **Errata del original conservada:** en la línea de `log_metric` la diapositiva escribe
> **`CX_test`** en lugar de `X_test`. Se transcribe tal cual. **[verificar]**

### 10.8 Autolog (diap. 81)

> *"Si eres flojo (como yo) → autolog"*

```python
import mlflow
mlflow.tensorflow.autolog()

model = ...
model.compile(...)
model.fit(...)
```

**`autolog` registrará automáticamente:**

1. **Métricas y parámetros**
   - pérdida de entrenamiento; pérdida de validación; métricas definidas por el usuario
   - parámetros de `fit()` o `fit_generator()`; nombre del optimizador; tasa de aprendizaje;
     epsilon
2. **Artefactos**
   - resumen del modelo al inicio del entrenamiento
   - modelo en formato MLflow Model (modelo Keras)
   - logs de TensorBoard al finalizar el entrenamiento

> ⚠ **Solo compatible con TensorFlow 2.3.0 ≤ versión ≤ 2.13.0.**

### 10.9 Registrar gráficos (diap. 82)

> *"Mejor característica de MLflow: registrar gráficos"*

```python
import mlflow
import tensorflow as tf
import matplotlib.pyplot as plt

mlflow.set_experiment("ExperimentABCwithPlots")   # <----- DEFINIR EXPERIMENTO

with mlflow.start_run() as run:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10)

    plt.plot(history.history['loss'], label='training')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model Loss Progress'); plt.ylabel('Loss'); plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('loss_plot.png')

    mlflow.log_artifact('loss_plot.png')   # <----- EL GRÁFICO SE ALMACENA COMO ARTEFACTO
                                           #        Y SE ASOCIA CON ESTA EJECUCIÓN DEL
                                           #        EXPERIMENTO. AHORA PUEDE VISUALIZARSE
                                           #        EN LA INTERFAZ DE MLFLOW.
```

### 10.10 Visión general del servidor: `mlflow ui` (diap. 83)

Captura de la interfaz gráfica de MLflow, donde se comparan las ejecuciones registradas.

> *Errata del original:* la diapositiva titula **"Visiòn general del servidor"** (acento grave).

### 10.11 Opciones para configurar MLflow Tracking (diap. 84)

*(Tabla que en el PDF está íntegramente como imagen.)*

| | **1 · Local (inicio rápido)** | **2 · Base de datos local** | **3 · Tracking Server remoto / compartido** |
|---|---|---|---|
| **Para qué** | Ideal para pruebas y desarrollo | Más opciones de base de datos; sin servidor remoto | Para equipos y entornos de producción |
| **Cliente** | Código de ML (Python, Jupyter, etc.) | Código de ML (Python, Jupyter, etc.) | Múltiples clientes (Python, UI web, REST API): Investigador · Estudiante · Equipo · … |
| **Componente MLflow** | MLflow **Tracking** (API local) | MLflow **Tracking** (API local) | MLflow **Tracking Server** (REST API + UI) |
| **Backend Store (metadatos)** | **SQLite** (por defecto en MLflow 3.7+) | SQLite · PostgreSQL · MySQL · MSSQL (SQLAlchemy) | PostgreSQL · MySQL · MSSQL |
| **Artifact Store (artefactos)** | **Sistema de archivos local** (`./mlruns` por defecto) | Local / NFS | S3 · Azure Blob · Google Cloud Storage · NFS / SFTP |
| **Extras** | — | — | Colaboración · Control de acceso · Auditoría · Escalabilidad |
| **Nota del deck** | ℹ Si existe `./mlruns`, MLflow puede reutilizarlo por compatibilidad. | ✅ Mejor gestión de metadatos; sin servidor remoto. | 🦊 **GitLab** puede actuar como backend remoto de experiment tracking mediante integración MLflow; úsalo si tu instancia lo habilita. |

---

## 11. Conclusiones (diap. 85–86)

| Idea clave | Desarrollo |
|---|---|
| **Un modelo genera valor cuando pasa de experimento a producción** | Y se integra correctamente con un sistema real. |
| **No existe una única estrategia de despliegue** | La elección depende del **control, costo, escalabilidad, infraestructura y autonomía** requerida. |
| **MLOps convierte el ML en un proceso reproducible y mantenible** | Automatizando **pruebas, integración, despliegue y monitoreo**. |
| **MLflow aporta trazabilidad y reproducibilidad** | Permitiendo registrar y comparar **código, parámetros, métricas y artefactos** de los experimentos. |

---

## 12. Próxima semana (diap. 87–88)

En el mapa "La historia de este curso", el recuadro resaltado pasa a **Desviación conceptual
(Concept Drift)**.

*(Diap. 89: "Thank you! / ¡Gracias! / OBRIGADO!". Diap. 91: **Kahoot**. Diap. 92: portada de
cierre.)*

---

## 13. Glosario de términos del curso

> **Nota de organización:** este glosario **reordena y recoge las definiciones tal como las da el
> deck**; no añade teoría nueva. La diapositiva de origen se indica entre paréntesis.

| Término | Definición según el curso |
|---|---|
| **API** | Expresa un componente (de servicio) en términos de sus funcionalidades, entradas, salidas y tipos subyacentes. (29) |
| **Artefacto** | Cualquier resultado tangible generado durante el desarrollo de software: datos, código, modelos, entornos y documentación. (58) |
| **AutoML** | Componente del pipeline de desarrollo del modelo que automatiza el entrenamiento. (69) |
| **"Cambiar algo cambia todo"** | *Changing Anything Changes Everything*: principio derivado de que datos, modelos y código están estrechamente interconectados. (53) |
| **Cloud Foundry** | PaaS para desplegar y escalar aplicaciones sin gestionar directamente la infraestructura; integra servicios externos, soporta múltiples lenguajes, ofrece escalado automático e incluye logging y monitoreo. (44) |
| **Contenedor** | Encapsula solo la aplicación y sus dependencias; comparte el kernel del SO del host. Ligero, portátil, de arranque rápido y fácil de escalar. (43) |
| **DevOps** | *Development + Operations*: prácticas, herramientas y cultura para desarrollar software de mejor calidad y más rápidamente, reduciendo la brecha entre desarrollo y operaciones. (49) |
| **Docker** | Motor de contenedores; empaqueta la aplicación/modelo como código + modelo + dependencias. (33, 43) |
| **Entrenamiento final** | Usar todos los datos disponibles para construir el modelo destinado al entorno productivo. (18) |
| **FaaS / Serverless** | Funciones bajo demanda con escalado automático; pago por uso. Control bajo, abstracción alta. (33, 39) |
| **Feature store** | Repositorio central para almacenar, versionar y compartir características (*features*). (65) |
| **IaaS** | Infraestructura como servicio: gestionas SO, librerías y aplicación. Más control. (33, 39) |
| **IA predictiva / recomendadora / semiautónoma / autónoma** | Los cuatro niveles de implementación, ordenados de menor a mayor autonomía y complejidad técnica. (22–23) |
| **Kubernetes** | Sistema para la gestión de contenedores; permite despliegue, mantenimiento y escalado de aplicaciones contenerizadas. Conceptos clave: Pods, Nodos, Plano Maestro. (43) |
| **Máquina Virtual (VM)** | Simula un computador completo con su propio SO, librerías y aplicaciones; se ejecuta sobre un hipervisor. Aislamiento fuerte, pero pesada. (43) |
| **Metadata store** | Repositorio central para almacenar y gestionar los metadatos recolectados durante el ciclo de vida del modelo. (65) |
| **Microservicios** | Componentes independientes que se comunican entre sí; más escalables y flexibles, ideales para pipelines de ML. (33, 36) |
| **MLaaS / SaaS** | Servicios gestionados para entrenar y desplegar modelos; máxima abstracción. Ej.: Azure ML services, Amazon SageMaker, IBM Watson Studio. (33, 45) |
| **MLflow** | Herramienta de MLOps con cuatro componentes: seguimiento de experimentos, proyectos, registro de modelos y modelos. (75) |
| **MLOps** | *Machine Learning + Operations*: prácticas para desarrollar, desplegar y mantener modelos de ML en producción de forma rápida y confiable; promueve colaboración entre ingenieros de datos, científicos de datos y operaciones. (51) |
| **Modelo "definitivo"** | El artefacto que será empaquetado y desplegado, documentado junto con parámetros, métricas de referencia y pipeline de preprocesamiento. (19) |
| **Modelo como función serverless** | Desplegar el modelo como una función bajo demanda. (70) |
| **Modelo como servicio** | Desplegar el modelo como un servicio independiente. (70) |
| **Modelo en servicio** | Desplegar el modelo en el servidor de la aplicación. (70) |
| **Monolítico** | Una sola aplicación que integra API, preprocesamiento, modelo y postprocesamiento. Simplicidad, pero difícil de escalar. (33, 36) |
| **On-premise** | Infraestructura propia: máximo control, privacidad y personalización; alto costo y mantenimiento. (39) |
| **PaaS** | Plataforma como servicio: despliegas tu app sin gestionar infraestructura. Equilibrio entre control y abstracción. (33, 39) |
| **Predicción en el edge** | Desplegar el modelo en el dispositivo del usuario: baja latencia, funciona sin conexión, mayor privacidad. (70) |
| **Predicción por lotes** | Ejecutar el modelo periódicamente y guardar resultados en una base de datos. (70) |
| **Reproducibilidad** | Volver a ejecutar experimentos y obtener los mismos resultados. (58) |
| **Reutilización** | Capacidad de utilizar un modelo o fragmento de código para distintos propósitos y en diferentes entornos. (61) |
| **Trazabilidad** | Identificar qué código, datos o configuración generaron cada resultado. (58) |
| **Versionado de datos** | Permite rastrear y gestionar los cambios en los datos a lo largo del tiempo (p. ej. DVC, LakeFS). (55, 65) |

---

## 14. Checklists de síntesis

> **Nota de organización:** esta sección **reorganiza criterios ya enunciados en el deck** para
> facilitar su uso; **no añade teoría nueva**. Cada ítem remite a la diapositiva de origen.

### 14.1 Checklist de las cuatro decisiones de despliegue (diap. 33)

- [ ] **1 · Encapsulamiento:** ¿monolítico o microservicios? (¿simplicidad o escalado
      independiente?)
- [ ] **2 · Entorno físico de ejecución:** ¿instancia única, máquina virtual o contenedor?
- [ ] **3 · Plataforma:** ¿on-premise, IaaS, PaaS, FaaS/Serverless o MLaaS/SaaS? (eje **más
      control ↔ más abstracción**)
- [ ] **4 · Portabilidad y escalabilidad:** ¿Docker solo, o además Kubernetes / Cloud Foundry?

### 14.2 Guía de elección de estrategia de despliegue (diap. 70)

| Si tu prioridad es… | Estrategia del curso |
|---|---|
| Simplicidad y escalabilidad, tolerando resultados desactualizados | **Predicción por lotes** |
| Baja latencia, operación sin conexión y privacidad | **Predicción en el edge** |
| Reutilizar la infraestructura existente | **Modelo en servicio** |
| Confiabilidad, escalabilidad y flexibilidad | **Modelo como servicio** |
| Pagar solo por cómputo y no gestionar infraestructura | **Modelo como función serverless** |

> 💡 Recordatorio del deck: *no existe una única estrategia mejor; depende de latencia, costo,
> privacidad y escala.*

### 14.3 Checklist de los cinco principios MLOps (diap. 58–62)

- [ ] **Versionar todo:** ¿están versionados datos, código, modelos, entornos y documentación?
      (58)
- [ ] **Automatizar siempre que sea posible:** ¿qué tareas manuales y repetitivas quedan? (59)
- [ ] **Si no está probado, está roto:** ¿pasan los tests antes de integrar? ¿se prueba todo el
      sistema y no solo el modelo? (60)
- [ ] **Reutilización:** ¿el sistema es modular, con herramientas interoperables y artefactos
      fáciles de localizar? (61)
- [ ] **Colaboración y comunicación:** ¿hay transparencia sobre experimentos, resultados,
      modelos y despliegues? (62)

### 14.4 Checklist contra las "prácticas problemáticas" (diap. 14)

- [ ] ¿Se ha reducido el trabajo manual y repetitivo?
- [ ] ¿El seguimiento de experimentos está ordenado (p. ej. con MLflow)?
- [ ] ¿Datos, modelos **y** código están bajo control de versiones?
- [ ] ¿Existen pruebas de código y revisiones de código?
- [ ] ¿Se evita el enfoque excesivo en una única métrica?

### 14.5 Correspondencia principio MLOps → herramienta (diap. 65, 73)

| Necesidad | Familia de herramienta (65) | Ejemplo del workflow (73) |
|---|---|---|
| Seguimiento de experimentos | Monitoreo de modelos/datos/aplicaciones | **MLflow** |
| Experimentación | Análisis de datos | **Jupyter** |
| Versionado de datos | Versionado de datos | **DVC** |
| Versionado de código | Repositorio de código | **Git** |
| Orquestación de pipelines | Pipelines de CI/CD | **Apache Airflow** |
| Registro de modelos | Metadata store | **MLflow** |
| Servicio de modelos | Servicio de predicciones | **BentoML** |
| Monitoreo de modelos | Monitoreo | **Prometheus · Grafana** |

---

## 15. Notas de fidelidad sobre la reconstrucción

- **Esta sesión no contiene fórmulas matemáticas.** Su contenido técnico son marcos de decisión,
  diagramas de arquitectura y ejemplos de código; todos ellos se transcribieron de la lectura
  visual de las diapositivas rasterizadas.
- **Contenido recuperado solo por lectura visual** (la capa de texto devuelve estas diapositivas
  vacías o con el título únicamente): los **cuatro niveles de implementación** (diap. 22–23); el
  **marco de las cuatro decisiones de despliegue** (diap. 33–40); la comparación **VM vs
  contenedores vs Kubernetes** (diap. 43); el diagrama **"Desacoplando en distintos pipelines"**
  (diap. 55), que hubo que **recortar y ampliar 5×** en tres regiones para leer sus viñetas; el
  diagrama de los **tres planos código/modelo/datos** (diap. 54); los esquemas de **Gestión de
  Datos** y **Desarrollo del Modelo** (diap. 68–69); la tabla de las **cinco estrategias de
  despliegue** (diap. 70); el diagrama de **monitoreo y bucles de retroalimentación** (diap. 72);
  el **workflow de herramientas** (diap. 73); las **familias de herramientas** (diap. 65); la
  **visión general de MLflow** (diap. 76); y las **tres opciones de configuración de MLflow
  Tracking** (diap. 84).
- **Dato numérico verificado:** el gráfico de la brecha de despliegue (diap. 13) reporta
  **frecuencias**, no porcentajes: 65 · 25 · 8 · 8 · 8, que suman 114 respuestas. La lectura
  correcta es que la mayoría de las organizaciones despliega menos del 20 % de los modelos
  previstos. Fuente citada: encuesta de **KDnuggets de 2021**.
- **Erratas del original conservadas y señaladas en su lugar:** `CX_test` en lugar de `X_test` en
  el ejemplo de registro manual (diap. 80); **"Visiòn general del servidor"** con acento grave
  (diap. 83).
- Elementos marcados **[verificar]**: la línea del mensaje `assistant` del ejemplo de API, que
  aparece **cortada por el borde del recuadro de código** en la diapositiva 29; y la errata
  `CX_test` de la diapositiva 80.
- **Animaciones por construcción:** varias secuencias son estados intermedios de una misma lámina
  (12–13, 16–19, 33/34/37/40, 42–43, 47–48, 53–54, 56–57, 64–65). Esta guía reconstruye el
  **estado completo** de cada una e indica el rango correspondiente. En el marco de las cuatro
  decisiones, el orden de resalte del deck es **columna 1 (diap. 34) → columna 3 (diap. 37) →
  columna 2 (diap. 40)**; la columna 4 no recibe diapositiva de resalte propia.
- **Ubicación de la sesión:** la diapositiva 2 aún resalta *Modelado y rendimiento* (cierre de la
  Semana 5); la 7 marca **Despliegue** como sesión en curso y la 88 avanza a *Concept Drift*.

---

*Guía de estudio elaborada a partir de `Sem6_Despliegue.pdf` (92 diapositivas).*
*Curso: **Planificación y Toma de Decisiones en IA**, Dra. Aurea Soriano-Vargas — UTEC, 2026.*
