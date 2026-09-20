# Semana 3 — Iniciación · Guía de estudio

| Campo | Valor |
|---|---|
| **Curso** | Planificación y Toma de Decisiones en IA |
| **Sesión** | Semana 3 · *Iniciación* · 2026-2 |
| **Docente** | Dra. Aurea Soriano-Vargas (`aureasoriano` · asoriano@utec.edu.pe · `aurea-soriano`) |
| **Institución** | UTEC |
| **Fuente** | `Sem3_Iniciación.pdf` — 108 diapositivas |
| **Atribución del pie** | *Planificación y Toma de Decisiones en IA*, Dr. Aurea Soriano-Vargas (2026) |

> **Propósito de esta guía:** reconstruir fielmente el contenido teórico de las 108 diapositivas
> de la sesión —incluyendo fórmulas, tablas, ejercicios numéricos y diagramas que en el PDF están
> como imágenes— para poder estudiar la sesión sin depender del deck original.

**Política de evaluación (diap. 2).** `https://tinyurl.com/PTDIA-2026-02` — Cada participación
vale 0.01 en el examen correspondiente. (Máximo 1.5 pts)

---

## 1. Mapa del curso: ubicación de la sesión

Diagrama **"La historia de este curso"** (diapositivas 5 y 104), con cuatro carriles. El recuadro
con borde amarillo marca la sesión; los recuadros en gris son las sesiones ya cubiertas.

```
Fundamentos        ┌──────────────────────┐
                   │ Motivación y         │  (gris = ya visto, Sem 2)
                   │ terminología         │
                   └──────────┬───────────┘
                              │
Ciclo de vida      ┌──────────▼───┐   ┌──────────────┐   ┌───────────┐   ┌──────────────────┐
de la IA           │  Iniciación  │──▶│  Modelado y  │──▶│ Despliegue│──▶│ Desviación       │
                   └──────────────┘   │  rendimiento │   └───────────┘   │ conceptual       │
                    ↑ resaltado en     └──────────────┘                  │ (Concept Drift)  │
                      la diap. 5        ↑ resaltado en la diap. 104      └────────┬─────────┘
                      (ESTA SESIÓN)       (PRÓXIMA SEMANA)                        │
                              ┌───────────────────────────────────────────────────┘
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

### 1.1 Conclusiones de la sesión anterior (diap. 4)

| Idea clave | Desarrollo |
|---|---|
| **Predecir no es decidir** | Una buena predicción no garantiza una buena decisión. Para actuar correctamente se necesitan objetivos, restricciones, métricas y responsables. |
| **Entender la IA evita expectativas irreales** | Permite elegir mejor las soluciones y comprender sus verdaderas capacidades y limitaciones. |
| **La IA no es solo un modelo** | Opera dentro de procesos, servicios y ciclos de vida completos: iniciación, entrenamiento, estimación, validación y despliegue. |
| **La confianza debe diseñarse** | Sesgo, drift, degradación y errores requieren evidencia, monitoreo continuo, límites claros y participación humana cuando sea necesaria. |
| **El éxito combina tecnología y organización** | Una solución de IA efectiva depende tanto del diseño técnico como de la gestión, la gobernanza y la capacidad de adaptarse al entorno. |

---

## 2. Objetivos y agenda de la sesión

### 2.1 Al finalizar la clase deberíamos tener los medios para responder (diap. 6)

1. Importancia de definir el problema de negocio relacionado.
2. Opciones de preprocesamiento y *feature engineering*.
3. Conocimiento sobre los distintos tipos de datos.
4. Obtener ideas sobre la recopilación de datos.

> *(La diapositiva escribe "recopilaciónb"; es una errata del original.)*

### 2.2 Agenda (diap. 7)

| Letra | Tema |
|---|---|
| **A** | Planteamiento del problema |
| **B** | Recopilación de datos |
| **C** | Muestreo |
| **D** | Distribución de datos |
| **E** | Calidad de los datos |
| **F** | Preprocesamiento de datos |
| **G** | Ingeniería de atributos |

Estas letras A–G se reutilizan en todo el deck como índice de los componentes de la iniciación.

---

## 3. ¿Qué es la etapa de iniciación? (sección, diap. 8)

### 3.1 Definición operativa (diap. 9–10)

La iniciación consiste en:

1. **Seleccionar datos relevantes** de las fuentes de datos.
2. **Preparar los datos** para que sean digeribles por algoritmos.

> **Objetivo: Resolver el problema inicial.**

### 3.2 ¿Para qué? Ejemplo de aprendizaje supervisado (diap. 11)

Tabla del deck, dividida en un bloque **Entrada** y un bloque **Objetivo**:

| ID | Clases asistidas | Trabajos Enviados | (…) | **Examen aprobado** |
|---|---|---|---|---|
| 1 | 12 | 6 | | SÍ |
| 2 | 2 | 1 | | NO |
| 3 | 13 | 4 | | SÍ |
| 4 | | | | |
| … | | | | |
| **NUEVO** | 11 | 3 | | **?** |

- Sobre las filas 1–4/…: **"Aprende relaciones…"**
- Sobre la fila NUEVO: **"…y luego aplica este conocimiento a instancias nuevas."**

### 3.3 Ubicación en el ciclo de vida (diap. 12–13)

```
┌────────────┐   ┌───────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│ Iniciación │──▶│ Entrenamiento │──▶│ Estimación │──▶│ Validación │──▶│ Despliegue │
└────────────┘   └───────────────┘   └────────────┘   └────────────┘   └────────────┘
```

> **Iniciación:** Establecer bases, incluidas las características del conjunto de datos.

**Pregunta de clase (diap. 13): ¿Cuál es el orden correcto lógico?**

- A. Modelo → datos → problema
- B. Problema → datos → features
- C. Features → modelo → problema
- D. Datos → modelo → problema

> *El deck no marca la opción correcta en la diapositiva.*

### 3.4 Los siete componentes de la iniciación (diap. 14, 17)

| Letra | Componente | Ejemplo / pregunta guía del deck |
|---|---|---|
| **A** | Planteamiento del problema | p. ej., clasificación binaria, regresión, etc. |
| **B** | Recopilación de datos | p. ej., rigor, etiquetado |
| **C** | Muestreo | p. ej., muestreo estratificado a partir de un conjunto de datos más amplio |
| **D** | Distribución de datos | p. ej., cantidad de clases, porcentaje, distribución |
| **E** | Calidad de los datos | p. ej., escasez, ruido, complejidad |
| **F** | Métodos de preprocesamiento de datos | ¿cuáles y por qué? |
| **G** | Ingeniería de atributos y vectorización | ¿Qué y por qué? |

### 3.5 El proceso de iniciación simplificado es **iterativo** (diap. 15)

Este esquema reaparece como "mapa de ubicación" en las diapositivas 59, 73, 85, 93 y 99.

```
   ──●──────────────●────────────────────────────────────────▶
     │              │          ╭──────────────────────────╮
  ┌──┴──────────┐ ┌─┴───────┐  │   Análisis Descriptivo   │
  │ Recopilación│ │Muestreo │  │  ┌────────────────────┐  │
  │  de datos  B│ │       C │  │  │Distribución datos D│  │
  └─────────────┘ └─────────┘  │  ├────────────────────┤  │
                               │  │  Calidad de datos E│  │
                               │  └────────────────────┘  │
                               ╰────────┬────────▲────────╯
                                        │        │   ITERACIÓN
                                        ▼        │
                               ╭──────────────────────────╮
                               │    Ingeniería de Datos   │
                               │  ┌────────────────────┐  │
                               │  │Métodos de preproc.F│  │
                               │  ├────────────────────┤  │
                               │  │Ing. atributos y    │  │
                               │  │  vectorización    G│  │
                               │  └────────────────────┘  │
                               ╰──────────────────────────╯
```

**Lectura del diagrama:** B y C son pasos secuenciales sobre la línea de tiempo; D–E (*Análisis
Descriptivo*) y F–G (*Ingeniería de Datos*) forman un **ciclo iterativo** — las dos flechas
curvas del original indican que se va y se vuelve entre analizar y transformar los datos.

### 3.6 Definición y meta (diap. 16)

- **Definición:** Primera fase donde se establecen las bases.
- **Meta:** Preparar los datos, definir el problema y traducir a IA.
- **Ejemplo:**
  - *Problema de negocio:* reducir la fuga de clientes en un banco.
  - *Traducción a IA:* clasificación binaria (cliente se queda vs cliente se va).

---

## 4. A · Planteamiento del problema (sección, diap. 18)

### 4.1 Lo que consideramos un problema bien definido (diap. 19)

| Elemento | Definición del curso |
|---|---|
| **Objetivos** | Describe el objetivo principal, así como otras preguntas relacionadas que puedan ser relevantes. **¿Qué quiero lograr?** |
| **Criterios de éxito** | Describe los criterios para el éxito del proyecto. Estos pueden ser **específicos y medibles**. |

### 4.2 Traducción: de problema de negocio a problema de ML (diap. 20–23)

> **Los algoritmos de aprendizaje automático necesitan objetivos mensurables.**

```
┌───────────────────────────────────┐              ┌────────────────────────────────────┐
│     Problema de negocios          │              │ Problema de aprendizaje automático │
│                                   │ ─Traducción─▶│                                    │
│ Los objetivos y criterios de      │              │ Por otro lado, los algoritmos      │
│ éxito describen el problema.      │              │ requieren un objetivo claro y      │
│ Sin embargo, subjetivas y         │              │ medible.                           │
│ carecen de mensurabilidad.        │              │                                    │
└───────────────────────────────────┘              └────────────────────────────────────┘
```

**Tabla de traducción (diap. 23):**

| Negocio | AI |
|---|---|
| Detectar fraude | Clasificación |
| Predecir ventas | Regresión |
| Agrupar clientes | Clustering |

> La diapositiva 21 muestra el paso intermedio de la animación: **Aprendizaje Supervisado** se
> ramifica en **Clasificación** y **Regresión**.

### 4.3 Ejemplo de Investigación e Industria (diap. 24)

| Fuente | Contenido |
|---|---|
| **Nature News feature, 2024** — *"Los generadores de imágenes con IA suelen arrojar resultados racistas y sexistas."* | Los modelos de GenIA de texto a imagen a menudo producen imágenes que incluyen rasgos sesgados y estereotipados relacionados con el género, el color de piel, las ocupaciones, las nacionalidades y más. |
| **David Rozado, 2023** (New Zealand Institute of Skills and Technology) — *"Los sesgos políticos de ChatGPT"* | ChatGPT impulsa la interacción tecnológica, pero plantea inquietudes sobre su sesgo político. Las pruebas políticas revelan una inclinación izquierdista en ChatGPT. |

> Pregunta de cierre: **¿Cómo garantizar la neutralidad factual?**

---

## 5. B · Recolección de Datos (sección, diap. 26)

### 5.1 Tidy Data (diap. 27)

> Este formato requiere tener **una columna por cada variable** y **una fila por cada
> observación**.

**Formato de partida (no *tidy*, formato ancho):**

| Ventas por cliente | Dic-2020 | Ene-2021 |
|---|---|---|
| Pedro | 180 | 165 |
| Luis | 160 | 180 |
| Pablo | 175 | 200 |

**Formato *tidy* (largo):**

| Cliente | Mes | Ventas |
|---|---|---|
| Pedro | Dic-2020 | 180 |
| Luis | Dic-2020 | 160 |
| Pablo | Dic-2020 | 175 |
| Pedro | Ene-2021 | 165 |
| Luis | Ene-2021 | 180 |
| Pablo | Ene-2021 | 200 |

> Verificación: las 6 celdas de la tabla ancha (3 clientes × 2 meses) se corresponden una a una
> con las 6 filas de la tabla *tidy*, con los mismos valores.

### 5.2 Varias formas de estructuras de datos (diap. 28)

| | **Datos estructurados** | **Datos semi estructurados** | **Datos No estructurados** |
|---|---|---|---|
| **Definición** | Forma bien estructurada, a menudo como una tabla o varias tablas. | Estructura autodescriptiva con ciertas propiedades organizativas. Carece de la estructura formal. | No sigue ninguna estructura. **Representa más del 80 % de todos los datos.** |
| **Ejemplo del deck** | Tabla con columnas ID, Name, Age, Degree | Fragmento XML `<University><Student ID="1">…` | Párrafo en prosa que describe lo mismo |

**El mismo contenido en los tres formatos (imágenes de la diapositiva):**

*Estructurado:*

| ID | Name | Age | Degree |
|---|---|---|---|
| 1 | John | 18 | B.Sc. |
| 2 | David | 31 | Ph.D. |
| 3 | Robert | 51 | Ph.D. |
| 4 | Rick | 26 | M.Sc. |
| 5 | Michael | 19 | B.Sc. |

*Semi estructurado:*

```xml
<University>
  <Student ID="1">
    <Name>John</Name>
    <Age>18</Age>
    <Degree>B.Sc.</Degree>
  </Student>
  <Student ID="2">
    <Name>David</Name>
    <Age>31</Age>
    <Degree>Ph.D. </Degree>
  </Student>
  ....
</University>
```

*No estructurado:*

> "The university has 5600 students. John's ID is number 1, he is 18 years old and already holds
> a B.Sc. degree. David's ID is number 2, he is 31 years old and holds a Ph.D. degree. Robert's
> ID is number 3, he is 51 years old and also holds the same degree as David, a Ph.D. degree."

### 5.3 Datos estructurados: anatomía de un dataset (diap. 29–30)

> Bases de datos, hojas de cálculo / archivos csv.

Ejemplo con el conjunto **Iris** (imagen de la diapositiva):

| | Sepal length | Sepal width | Petal length | Petal width | *(etiqueta)* |
|---|---|---|---|---|---|
| 1 | 5.1 | 3.5 | 1.4 | 0.2 | Setosa |
| 2 | 4.9 | 3.0 | 1.4 | 0.2 | Setosa |
| … | | | | | |
| 50 | 6.4 | 3.5 | 4.5 | 1.2 | Versicolor |
| … | | | | | |
| 150 | 5.9 | 3.0 | 5.0 | 1.8 | Virginica |

- **N = 150** → número de **instancias** (las filas).
- **d = 4** → número de **atributos** (Sepal length, Sepal width, Petal length, Petal width).
- La última columna es la **variable objetivo / etiqueta** (la especie).

### 5.4 Datos no estructurados (diap. 31)

> Características como **píxeles de imagen, señales de audio, oraciones de texto**.

### 5.5 Combinar fuentes y rigor (diap. 32–33)

> WhatsApp genera datos … **estructurados** (hora, emisor) y **no estructurados** (texto libre).

- A menudo es necesario **combinar diferentes fuentes y tipos de datos**.
- La recopilación de datos resultante debe ser **rigurosa**, es decir, **completa y sin lagunas**
  en relación con el objeto de interés (por ejemplo, sin periodos de tiempo omitidos en análisis
  críticos).

### 5.6 Fuente de Datos (diap. 34)

La diapositiva presenta dos tipos de fuente sobre un podio numerado **1** y **2**:

| Fuente | Definición del deck |
|---|---|
| **1** | Los datos **generados por sus usuarios**. |
| **2** | **Provienen de una fuente que ha recopilado datos para uso general.** |

> *Nota de fidelidad:* la diapositiva identifica las fuentes **solo con los números 1 y 2**; no
> aparecen en el deck los rótulos "primaria"/"secundaria". Esta guía conserva la numeración del
> original.

### 5.7 Ejercicio de clasificación de fuentes (diap. 35–40)

La respuesta se revela fila por fila a lo largo de las diapositivas 36–40. Resultado final:

| Ejemplo | Fuente |
|---|---|
| Registros históricos de una empresa recolectados originalmente para operaciones y luego reutilizados para entrenar un modelo | **2** |
| Señales de wearables recolectadas directamente durante un experimento diseñado para desarrollar un modelo de ML | **1** |
| Base de datos clínica existente utilizada posteriormente para entrenar un modelo predictivo | **2** |
| Logs generados por una aplicación desarrollada específicamente para estudiar el comportamiento de los usuarios | **1** |
| Dataset publicado junto con un artículo científico y reutilizado para evaluar un nuevo algoritmo | **2** |

---

## 6. C · Muestreo (sección, diap. 41)

**Pregunta de clase (diap. 42): ¿Por qué necesitamos muestreo?**

- A. Reducir tiempo
- B. Representar la población
- C. Mejorar accuracy
- D. Evitar overfitting

> *El deck no marca la opción correcta en la diapositiva.*

### 6.1 Definición (diap. 43)

> ¿Cómo identificar subconjuntos representativos?

- Puede resultar **ineficiente** utilizar todos los datos recopilados.
- El **muestreo** se define como **la selección de un subconjunto representativo de una gran
  cantidad de datos**.

### 6.2 Técnicas de muestreo (diap. 44)

| Técnica | Definición del curso |
|---|---|
| **Muestreo aleatorio simple** | Cada punto de datos tiene la misma probabilidad de ser seleccionado. |
| **Muestreo sistemático** | Todos los puntos de datos se ordenan según un criterio. Se selecciona aleatoriamente un punto de inicio y cada punto de datos en una distancia específica. |
| **Muestreo estratificado** | Cada punto de datos se asigna a una categoría y en cada categoría se realiza un muestreo aleatorio simple, y la recopilación conjunta de cada muestreo aleatorio simple constituye el subconjunto final. |
| **Muestreo por clusters** | Cada punto de datos se asigna a un cluster según un criterio específico. Algunos clústeres se seleccionan aleatoriamente y el conjunto de los clústeres seleccionados constituye el subconjunto final. |

### 6.3 Dataset base de los cuatro ejercicios (diap. 45)

| ID | Edad | Género | Nota Final |
|---|---|---|---|
| 1 | 18 | F | 12 |
| 2 | 19 | M | 15 |
| 3 | 20 | F | 14 |
| 4 | 21 | F | 16 |
| 5 | 22 | M | 10 |
| 6 | 18 | M | 13 |
| 7 | 19 | F | 17 |
| 8 | 20 | F | 11 |
| 9 | 21 | M | 14 |
| 10 | 22 | F | 18 |
| 11 | 18 | M | 15 |
| 12 | 19 | M | 12 |

> Composición: 12 registros, **6 mujeres (F)** = IDs 1, 3, 4, 7, 8, 10 y **6 hombres (M)** =
> IDs 2, 5, 6, 9, 11, 12.

### 6.4 Ejercicio 1: Muestreo aleatorio simple (diap. 46–47)

**Filas seleccionadas (resaltadas en la diapositiva): IDs 3, 8, 10, 11, 12.**

| ID | Edad | Género | Nota Final |
|---|---|---|---|
| 3 | 20 | F | 14 |
| 8 | 20 | F | 11 |
| 10 | 22 | F | 18 |
| 11 | 18 | M | 15 |
| 12 | 19 | M | 12 |

> **Ejemplo de resultado aleatorio: IDs 3, 8, 10, 11, 12.**
> **La muestra no garantiza representatividad por género ni edad, pero es simple y rápida.**

### 6.5 Ejercicio 2: Muestreo sistemático (diap. 48–50)

> **Ordenamos por nota y seleccionamos cada 3.**

Tabla ordenada por *Nota Final* ascendente; **▶** marca las filas seleccionadas:

| | ID | Edad | Género | Nota Final |
|---|---|---|---|---|
| **▶** | **5** | 22 | M | 10 |
| | 8 | 20 | F | 11 |
| | 12 | 19 | M | 12 |
| **▶** | **1** | 18 | F | 12 |
| | 6 | 18 | M | 13 |
| | 3 | 20 | F | 14 |
| **▶** | **9** | 21 | M | 14 |
| | 2 | 19 | M | 15 |
| | 11 | 18 | M | 15 |
| **▶** | **4** | 21 | F | 16 |
| | 7 | 19 | F | 17 |
| | 10 | 22 | F | 18 |

**Muestra resultante: IDs 5, 1, 9, 4.**

> Verificación: las filas resaltadas ocupan las posiciones 1, 4, 7 y 10 de la tabla ordenada —
> consistente con un paso de 3 partiendo de la primera posición.

### 6.6 Ejercicio 3: Muestreo estratificado (diap. 51–53)

> Queremos que la muestra mantenga la **proporción de género** (6 mujeres y 6 hombres).
> **Dividir el dataset en dos estratos: mujeres (F) y hombres (M).**
> **Seleccionar al azar 2 mujeres y 2 hombres para una muestra de 4.**

| Estrato | Miembros del estrato | Seleccionados |
|---|---|---|
| **Mujeres (F)** | 1, 3, 4, 7, 8, 10 | **IDs 1, 7** |
| **Hombres (M)** | 2, 5, 6, 9, 11, 12 | **IDs 5, 9** |

**Muestra resultante (4 registros):**

| ID | Edad | Género | Nota Final |
|---|---|---|---|
| 1 | 18 | F | 12 |
| 7 | 19 | F | 17 |
| 5 | 22 | M | 10 |
| 9 | 21 | M | 14 |

> **Con esto, la muestra refleja mejor la población en cuanto al género.**
> Verificación: 2 F y 2 M sobre un total de 4 = 50 %/50 %, la misma proporción que la población
> (6/12 y 6/12).

### 6.7 Ejercicio 4: Muestreo por cluster (diap. 54–56)

> **Dividir el dataset en clusters. Seleccionar los clusters.**

**Paso 1 — los cuatro clusters (diap. 55), identificados por color en el original:**

| Cluster | IDs | Comprobación con los datos |
|---|---|---|
| **Hombres relativamente jóvenes** | 2, 6, 11, 12 | M, edades 19, 18, 18, 19 |
| **Mujeres con puntaje bajo-medio** | 1, 3, 8 | F, notas 12, 14, 11 |
| **Mujeres con puntajes altos** | 4, 7, 10 | F, notas 16, 17, 18 |
| **Hombres de mayor edad** | 5, 9 | M, edades 22, 21 |

**Paso 2 — clusters seleccionados (diap. 56):** se resaltan **"Hombres relativamente jóvenes"** y
**"Mujeres con puntaje bajo-medio"**.

**Muestra resultante: IDs 1, 2, 3, 6, 8, 11, 12** (7 registros; los clusters "Mujeres con
puntajes altos" y "Hombres de mayor edad" quedan fuera).

---

## 7. D · Distribución de Datos (sección, diap. 57)

**Pregunta de clase (diap. 58): ¿Qué pasa si train ≠ test distribution?**

- A. Nada
- B. Mejor rendimiento
- C. Mala generalización
- D. Más datos

> *El deck no marca la opción correcta en la diapositiva.*

La distribución de datos es **una tarea esencial del análisis descriptivo** (diap. 59) — casilla
**D** del mapa de §3.5.

### 7.1 Clasificación de EDA (*Exploratory Data Analysis*) (diap. 60–61)

> ¿Cómo empiezo a analizar estos datos?

El EDA se clasifica en **no gráfica** y **gráfica**, y se orienta por cuatro preguntas:

| Foco | Pregunta guía |
|---|---|
| 🔎 **distribuciones** | ¿qué forma tiene la data? |
| 📌 **outliers** | ¿errores o señales raras? |
| 🧲 **relaciones** | ¿qué variables se mueven juntas? |
| 🧭 **hipótesis** | ¿qué historia empieza a emerger? |

### 7.2 Resúmenes numéricos de datos — fórmulas (diap. 62)

*(Transcripción de la diapositiva. La notación se conserva exactamente como aparece en el
original, incluido el uso de $X_i$ para los elementos y de $n$ para el tamaño.)*

#### Medidas de Tendencia Central
> *"centro" alrededor del cual se distribuyen los datos.*

- **media:**

$$\mu = \sum_i X_i \,/\, n$$

- **moda:** el valor común en $X$

- **mediana:** $X = \mathrm{sort}(X)$, mediana $= X_{n/2}$ (mitad abajo, mitad arriba)

#### Medidas de variación o variabilidad
> *"dispersión de datos"*

- **cuartiles $X$:** Q1 value $= X_{0.25n}$ , Q3 value $= X_{0.75\,n}$
  - **Rango de intercuartiles:** $\mathrm{value}(Q3) - \mathrm{value}(Q1)$
  - **range:** $\max(X) - \min(X) = X_n - X_1$

- **varianza:**

$$\sigma^{2} = \sum_i (X_i - \mu)^{2} \,/\, n$$

- **desviación estándar:** $\sigma$

- **skewness:**

$$\sum_i (X_i - \mu)^{3} \ / \ \left[\ \left(\sum_i (X_i - \mu)^{2}\right)^{3/2}\ \right]$$

  - cero si es simétrico; **asimétrico a la derecha es lo más común**

*La diapositiva acompaña las fórmulas con una curva normal marcada con líneas verticales
punteadas en los cuartiles.*

### 7.3 Robustez de las medidas: ejemplo numérico (diap. 63–67)

La diapositiva contrapone dos conjuntos de puntos sobre una recta numérica de 0 a 10:

```
Caso A (sin atípicos)            Caso B (con un atípico)
 ● ● ● ● ●                        ● ● ● ●                          ●
 1 2 3 4 5                        1 2 3 4                          10
       ▲                                ▲ ▲
    media = 3                  mediana=3  media = 4
```

| | **Caso A** | **Caso B** |
|---|---|---|
| Datos | 1, 2, 3, 4, 5 | 1, 2, 3, 4, 10 |
| Media | **3** | **4** |
| Mediana | 3 | **3** |

> Verificación aritmética: Caso A, $\mu = (1+2+3+4+5)/5 = 15/5 = 3$. Caso B,
> $\mu = (1+2+3+4+10)/5 = 20/5 = 4$; la mediana del Caso B sigue siendo 3. Ambas cuadran con las
> flechas de la diapositiva.

**Conclusiones del deck:**
- **Media:** para distribuciones simétricas sin valores atípicos.
- **La media y la desviación estándar son afectadas por los valores atípicos.**
- **La mediana y los cuartiles son más robustos. Mejor para distribuciones asimétricas.**

### 7.4 Moda y el riesgo de promediar categorías (diap. 68)

> **Moda:** Para una distribución finita, un buen valor "típico" es el que aparece con mayor
> frecuencia.

**Figura (imagen) — "Promediar categorías codificadas puede ser engañoso":**

| Paso | Contenido |
|---|---|
| **1 · Preferencias reales** | Dos grupos con preferencias distintas. Conteos (frecuencias): **Rojo = 10**, **Azul = 10**. Las observaciones están divididas entre dos categorías reales: Rojo y Azul. |
| **2 · Codificación numérica** | A las categorías se les asignan números de forma arbitraria: **Rojo = 1**, **Azul = 5**. Imaginemos una muestra con la misma cantidad de Rojo y Azul (50 % y 50 %). **Cálculo del promedio: Promedio = (1 + 5) / 2 = 3.** ⚠ La media depende de la codificación: si cambiamos los números, cambia el promedio. |
| **3 · Resultado engañoso** | El promedio cae en un color intermedio que no corresponde a ninguna preferencia real. Escala codificada (1 a 5): 1 = Rojo … 5 = Azul; **Color medio (valor promedio = 3)**. ✗ Este promedio no representa la preferencia real de ningún grupo. |

> Verificación aritmética: $(1+5)/2 = 3$. Correcto.

### 7.5 ¿Cuál es la distribución de nuestra variable objetivo? (diap. 69)

> **¿Por qué?** Conocer las distribuciones de datos es clave para evaluar los resultados. También
> puede proporcionar información sobre si los algoritmos y los pasos de preprocesamiento
> utilizados son adecuados y útiles.

**¿Cómo?**

| Tipo de problema | Herramienta |
|---|---|
| **Clasificación** | Análisis con gráficos de datos o histogramas. |
| **Regresión** | Gráficos de dispersión. |

El ejemplo de clasificación muestra un gráfico circular y un gráfico de barras con las
frecuencias por clase: **8, 3, 1, 1** (cuatro clases; el eje del gráfico de barras llega a 10).

### 7.6 Visualización de datos (diap. 70)

*(Diapositiva compuesta íntegramente por una lámina ilustrada; contenido transcrito de la imagen.)*

**¿Cómo elegir el tipo de gráfico adecuado?**

| Objetivo | Gráfico recomendado |
|---|---|
| Tendencias en el tiempo | Línea |
| Comparar categorías | Barras, pastel |
| Comparar totales | Pastel, barras apiladas |
| Relaciones | Dispersión, línea, facetas, doble línea |
| Proporciones | Pastel, dona, gofre |

**Evita el engaño en los gráficos** — *Datos honestos + representación deshonesta*:
- Manipular ejes, colores.
- Comparar lo incomparable.
- Señales de color inadvertidas.

**¡Da estilo al gráfico para mejorar la legibilidad!**
- Etiqueta los ejes, añade leyendas.
- Escala los ejes, inclina el texto.

**Explora animaciones y opciones de visualización 3D…** — se menciona **D3.js**, la
*visualización dinámica* y las *redes sociales / de usuarios*.

---

## 8. E · Calidad de Datos (sección, diap. 71)

**GIGO (diap. 72):** *"Garbage in… garbage out"*.

La calidad de datos es la casilla **E** del mapa de §3.5 (diap. 73).

**Pregunta de clase (diap. 74): ¿Cuál es peor?**

- A. Datos incompletos
- B. Datos ruidosos
- C. Datos inconsistentes
- D. Todos

> *El deck no marca la opción correcta en la diapositiva.*

### 8.1 Los seis criterios de calidad (diap. 75–76)

> La calidad de los datos se puede evaluar mediante **seis criterios**.

| # | Criterio | Definición del curso |
|---|---|---|
| **1** | **Completitud** | Proporción de datos presentes (ej. % de valores faltantes). |
| **2** | **Unicidad** | Ausencia de duplicados. |
| **3** | **Actualidad** (*timeliness*) | Datos suficientemente recientes. |
| **4** | **Validez** | Los datos cumplen con las reglas de formato (ej. email con @). |
| **5** | **Exactitud** (*Accuracy*) | Los datos reflejan la realidad (ej. temperatura = 400 °C → incorrecto). |
| **6** | **Consistencia** | Coherencia entre fuentes distintas. |

### 8.2 Ejercicio: casos para analizar (diap. 76–83)

Las respuestas se revelan caso por caso a lo largo de las diapositivas 77–83. Resultado final:

| # | Caso | Criterio violado y justificación del deck |
|---|---|---|
| 1 | **Edad: 300 años** ❌ | **Exactitud:** biológicamente imposible. Podría pasar validación de formato si solo esperaba un número. |
| 2 | **Sexo: Perro** ❌ | **Validez:** campo esperaba "Masculino/Femenino/Otro", pero recibió una especie animal. |
| 3 | **Email: abc@** ❌ | **Validez:** no cumple el formato estándar de email (falta dominio). |
| 4 | **DNI repetido en dos registros distintos** ❌ | **Unicidad:** una misma identificación para dos personas diferentes. |
| 5 | **Fecha de nacimiento: 2028-05-10** ❌ | **Actualidad / Exactitud:** fecha futura no válida para alguien vivo hoy. |
| 6 | **Campo dirección: vacío** ❌ | **Completitud:** dato faltante. |
| 7 | **Base A dice salario = 1500, Base B dice salario = 2500** (mismo trabajador, mismo mes) ❌ | **Consistencia:** contradicción entre fuentes. |

---

## 9. F · Preprocesamiento de Datos (sección, diap. 84)

Casilla **F** del mapa de §3.5 (diap. 85).

### 9.1 Preprocesamiento | Imágenes (diap. 86)

> Limpiar las imágenes antes de usar en un modelo.

*(Orden de izquierda a derecha tal como aparecen en la diapositiva.)*

| Corrección | Incluye |
|---|---|
| **Corrección de sensor** | Corrección de píxeles muertos, distorsión geométrica de la lente y viñeteado. |
| **Corrección de iluminación** | Filtrado de rangos, ecualización de histograma, etc. |
| **Corrección de ruido** | Eliminación de ruido, por ejemplo, mediante suavizado. |
| **Corrección geométrica** | Rotación, volteo y cambio de perspectiva. |
| **Corrección de color** | Redistribución de la saturación del color o corrección de artefactos de iluminación. |

### 9.2 Preprocesamiento | Texto (diap. 87–89)

> Cómo estructurar datos de texto no estructurados.

**Texto de partida:** *"Esto comenzó como un texto, cumpliendo todos los requisitos."*

| Técnica | Definición del curso | Resultado sobre el texto de ejemplo |
|---|---|---|
| **Tokenización** | Proceso de separación de una cadena de texto en las palabras que la componen. | `{Esto, comenzó, como, un, texto, cumpliendo, todos, los, requisitos}` |
| **Stemming** | Proceso algorítmico que reduce las palabras a su raíz. El resultado **no incluye necesariamente palabras reales**. | `{est, comenz, com, un, text, cumpl, tod, lo, requisit}` |
| **Part-of-speech (POS)** | El etiquetado (POS) es el proceso de identificar clases de palabras gramaticales. | `{Esto/PRON, comenzó/VERB, como/SCONJ, un/DET, texto/NOUN, cumpliendo/VERB, todos/DET, los/DET, requisitos/NOUN}` |
| **Lemmatization** | Busca cada palabra en un diccionario y analiza el POS para seleccionar la forma correcta en caso de desambiguación. | p. ej., *"comer"* vs *"conjunción"* |

**Leyenda de etiquetas POS del deck:** PRON = pronombre · VERB = verbo · SCONJ = conjunción
subordinante · DET = determinante / artículo · NOUN = sustantivo.

**Técnicas adicionales (diap. 89):**

| Técnica | Definición del curso | Ejemplo del deck |
|---|---|---|
| **Sustitución** | La eliminación de símbolos, palabras o caracteres permite eliminar características irrelevantes. Ejemplos comunes son la sustitución de emoticones o la eliminación de palabras vacías. | *"comenzó texto, cumpliendo todos requisitos"* |
| **N-gramas** | Conjuntos de palabras o conceptos que coexisten con la ventana *n* dada. | Del texto anterior: bigrama (*n*=2) **"todos requisitos"**; trigrama (*n*=3) **"cumple todos requisitos"** |

**Preprocesamiento semántico (diap. 89):**

| Concepto | Definición del curso | Ejemplo del deck |
|---|---|---|
| **Conjuntos de sinónimos** | Grupos de expresiones que comprenden el mismo concepto. | Para *"cumplir"* — **Sustantivos:** {cumplimiento, ejecución}, {observancia, acatamiento}, {realización, concreción}. **Verbos:** {cumplir, realizar, ejecutar}, {obedecer, acatar, respetar}, {lograr, alcanzar, satisfacer} |
| **Hiperónimos** | Representan una relación de orden superior desde una perspectiva semántica. | Para {cumplir, realizar, ejecutar} → hiperónimo {hacer, llevar a cabo} → que a su vez tiene un hiperónimo más general: {acción, actuar, cambiar} |

### 9.3 Preprocesamiento | Números (diap. 90)

> Eliminar valores atípicos en los datos y adaptarse a los requisitos del modelo.

| Técnica | Definición del curso |
|---|---|
| **Suavizado** | Es una técnica para **reducir el efecto de los valores atípicos** en la mayor parte de los puntos de datos. Los valores atípicos se ponderan con un peso menor en comparación con otros puntos de datos más comunes. Esta técnica debería eliminar el ruido y revelar los datos subyacentes. |
| **Normalización** | Es una técnica para **transformar los rangos de diferentes atributos en uno solo**. Algunos algoritmos de aprendizaje automático priorizan los atributos con un rango mayor que otros. |

*Figuras de apoyo:* para **Suavizado**, la serie "Global Mean Estimates based on Land and Ocean
Data" (NASA GISS) con *Annual Mean* y *Lowess Smoothing*; para **Normalización**, una nube de
puntos dispersa que se comprime dentro de un cuadro de lado 1.

> ⚠ **Nota de fidelidad importante:** en la capa de texto del PDF estas dos definiciones aparecen
> **intercambiadas** respecto de sus títulos. La asignación de arriba es la que se lee en la
> imagen de la diapositiva y es la correcta.

### 9.4 Preprocesamiento | Series temporales (diap. 91)

> Cómo manejar datos de series temporales.

| Problema | Métodos listados por el deck |
|---|---|
| **Reemplazo de valores faltantes** | Valor anterior o posterior al valor faltante, media, mediana, interpolación. |
| **Representación** | Muestreo, Promedio, Interpolación lineal, solo puntos importantes como mínimos y máximos, Representación simbólica, PCA, t-SNE… |
| **Alineamiento** | *Simple alignment methods*, *indicator variable*, *Correlation*, *Optimized Warping*, *Dynamic Time Warping*… |

*Figuras:* "Representación de series de tiempo mediante promedio" (*Raw data* vs *Resampled
data*) y "Dos series temporales con diferente longitud" (*Reference batch* vs *Query batch*).

---

## 10. G · Ingeniería de Atributos (sección, diap. 92)

Casilla **G** del mapa de §3.5 (diap. 93).

### 10.1 Dónde encaja en el flujo (diap. 94)

> Podemos incorporar **conocimientos del dominio**.

```
 Datos recopilados      Datos Preprocesados        Atributos        Variable(s) objetivo

  x₁  x₂  x₃             x'₁  x'₂  x'₃          f₁  f₂  f₃  f₄
  x₄  x₅  x₆     ──▶     x'₄  x'₅  x'₆    ──▶   f₅  f₆  f₇  f₈    ──▶        Y
  x₇  x₈  x₉             x'₇  x'₈  x'₉          f₉  f₁₀ f₁₁ f₁₂
                                               f₁₃ f₁₄ f₁₅ f₁₆
      └── Pre-Procesamiento ──┘  └─ Ingeniería ─┘   └── Modelo ML ──┘
                                   de Atributos
      └────── Foco previo ─────┘
                                      independiente ◀──────▶ dependiente
```

**Lectura:** los datos brutos $x_1 \dots x_9$ pasan por **Pre-Procesamiento** ("foco previo" de
la sesión) para dar $x'_1 \dots x'_9$; la **Ingeniería de Atributos** los convierte en los
atributos $f_1 \dots f_{16}$, que son las **variables independientes** con las que el **Modelo
ML** predice la **variable objetivo $Y$** (**dependiente**).

### 10.2 Métodos para datos numéricos (diap. 95)

> **Datos Brutos:** Los datos numéricos generalmente pueden ser consumidos por algoritmos sin
> necesidad de realizar adaptaciones adicionales.

| Método | Definición del curso |
|---|---|
| **Medidas estadísticas** | min, max, mean, median, var, count, quantiles … |
| **Binarización** | Reduce la característica a uno o cero. Esto se utiliza cuando la frecuencia no es importante. |
| **Redondeo** | Reduce la precisión de los puntos de datos al convertir los valores en números enteros. |
| **Binning** | Asigna puntos de datos a características discretas (*bins*). |
| **Transformación monótonica** | La transformación logarítmica o transformación Box-Cox estabiliza la varianza y hace que los datos se parezcan más a una distribución normal. |

### 10.3 Datos categóricos: transformación y encoding (diap. 96)

| **Transformación:** de valores categóricos en etiquetas numéricas | **Encoding:** después de la transformación |
|---|---|
| • Transformación de atributos **nominales** (sin orden entre ellos) en etiquetas numéricas.<br>• Transformación de atributos **ordinales** (con orden entre ellos) en etiquetas numéricas. | Las etiquetas numéricas **no son continuas, no se pueden comparar directamente**. Por lo tanto, la codificación añade **características ficticias** para cada etiqueta numérica única.<br><br>• **One-hot:** convierte **n categorías en n variables binarias**; una vale 1 y las demás 0.<br>• **Dummy:** convierte **n categorías en n−1 variables binarias**; una categoría queda represent |

> *Nota de fidelidad:* la última viñeta aparece **truncada a mitad de palabra** ("queda
> represent") en la propia diapositiva; se transcribe tal cual. [verificar: el texto completo no
> figura en el deck]

### 10.4 Ejemplo de one-hot vs dummy (diap. 97)

> Supongamos una variable **Color** con 3 categorías: Rojo, Azul y Verde.

**One-hot encoding** — *en one-hot, las 3 categorías generan 3 columnas*:

| Color | One-hot: Rojo | One-hot: Azul | One-hot: Verde |
|---|---|---|---|
| Rojo | 1 | 0 | 0 |
| Azul | 0 | 1 | 0 |
| Verde | 0 | 0 | 1 |

**Dummy encoding** — *en dummy encoding, usamos solo n−1 = 2 columnas*:

| Color | Dummy: Rojo | Dummy: Azul |
|---|---|---|
| Rojo | 1 | 0 |
| Azul | 0 | 1 |
| Verde | 0 | 0 |

> Verificación: con $n = 3$ categorías, one-hot usa 3 columnas y dummy usa $n-1 = 2$; en dummy,
> la categoría *Verde* queda representada por la combinación (0, 0). Consistente con la
> definición de §10.3.

### 10.5 Reducción de dimensión: PCA (diap. 98)

> **La reducción de dimensión ayuda a concentrar la información.**

> **PCA:** reduce características correlacionadas a un conjunto menor de **componentes no
> correlacionados**, conservando la mayor cantidad posible de información.

*Figuras:* dos diagramas de dispersión, **Before PCA** (ejes *Feature 1* / *Feature 2*, nube
inclinada = características correlacionadas) y **After PCA** (ejes *Principal Component 1* /
*Principal Component 2*, nube alineada con el eje horizontal). Ambos ejes van de −4 a 4 (X) y de
−6 a 6 (Y).

---

## 11. Cierre: IA centrada en datos (diap. 99–101)

### 11.1 Nuevo fenómeno (diap. 100)

- **Tradicionalmente** → se pensaba que lo más importante era el **modelo**.
- **Hoy** → la atención se centra en los **datos**.

### 11.2 Modelo-céntrico vs Dato-céntrico (diap. 101)

```
┌──────────────────────────────┐                  ┌──────────────────────────────┐
│  IA centrada en el modelo    │                  │   IA centrada en los datos   │
│ Mejorar el modelo de forma   │ ◀─Complementario─▶│ Mejorar los datos de forma   │
│ sistemática                  │                  │ sistemática                  │
│                              │                  │                              │
│  Datos 🔒 ──▶ Modelo ⟳       │                  │  Datos ⟳ ──▶ Modelo 🔒       │
│  (fijos)      (se itera)     │                  │  (se itera)   (fijo)         │
└──────────────────────────────┘                  └──────────────────────────────┘
```

| Definición | Texto del deck |
|---|---|
| **Inteligencia Artificial Centrada en Modelos** | Enfatiza la elección del tipo de modelo, la arquitectura y los hiperparámetros adecuados entre una amplia gama de posibilidades. |
| **Inteligencia Artificial Centrada en Datos** | Enfatiza el diseño y la ingeniería de datos son esenciales. |

> Los dos enfoques se presentan como **complementarios**, no excluyentes.

**Referencia mostrada en la diapositiva:** *Data-Centric Artificial Intelligence* (sección
CATCHWORD) — Johannes Jakubik · Michael Vössing · Niklas Kühl · Jannis Walk · Gerhard Satzger.

---

## 12. Conclusiones (sección, diap. 102–103)

- **La iniciación prepara el terreno:** sin buenos datos ni problema definido, el modelo fracasa.
- Recordar el principio: **"Garbage in, garbage out"**.
- La tendencia actual: **IA centrada en los datos (Data-Centric AI)**.

| Planteamiento del Problema | Datos | Análisis | Preprocesamiento |
|---|---|---|---|
| Planteamiento bien definido + objetivos + criterios de éxito = éxito de una solución de IA. | Diferentes criterios, como su estructura, origen y tipo. | Análisis de la distribución y la calidad de los datos ➡ información preliminar para los siguientes pasos. | El preprocesamiento de datos y la ingeniería de atributos preparan un conjunto de datos. |

---

## 13. Próxima semana (diap. 104)

En el mapa "La historia de este curso", el recuadro resaltado pasa de *Iniciación* a **Modelado y
rendimiento**.

*(Diap. 105: "Thank you! / ¡Gracias! / OBRIGADO!". Diap. 107: **Kahoot**. Diap. 108: portada de
cierre.)*

---

## 14. Glosario de términos del curso

> **Nota de organización:** este glosario **reordena y recoge las definiciones tal como las da el
> deck**; no añade teoría nueva. La diapositiva de origen se indica entre paréntesis.

| Término | Definición según el curso |
|---|---|
| **Actualidad** (*timeliness*) | Criterio de calidad: datos suficientemente recientes. (75) |
| **Alineamiento** | Tratamiento de series temporales de distinta longitud: *simple alignment methods*, *indicator variable*, *Correlation*, *Optimized Warping*, *Dynamic Time Warping*. (91) |
| **Atributos** | Las columnas descriptivas de un dataset; en Iris, $d = 4$. Son las variables **independientes**. (30, 94) |
| **Binarización** | Reduce la característica a uno o cero; se utiliza cuando la frecuencia no es importante. (95) |
| **Binning** | Asigna puntos de datos a características discretas (*bins*). (95) |
| **Calidad de los datos** | Componente **E**; se evalúa mediante seis criterios: completitud, unicidad, actualidad, validez, exactitud y consistencia. (75) |
| **Completitud** | Criterio de calidad: proporción de datos presentes (ej. % de valores faltantes). (75) |
| **Consistencia** | Criterio de calidad: coherencia entre fuentes distintas. (75) |
| **Criterios de éxito** | Describen los criterios para el éxito del proyecto; pueden ser específicos y medibles. (19) |
| **Datos estructurados** | Forma bien estructurada, a menudo como una tabla o varias tablas (bases de datos, hojas de cálculo, csv). (28–29) |
| **Datos semi estructurados** | Estructura autodescriptiva con ciertas propiedades organizativas; carece de la estructura formal. (28) |
| **Datos no estructurados** | No sigue ninguna estructura; representa más del 80 % de todos los datos (píxeles, audio, texto). (28, 31) |
| **Dummy encoding** | Convierte n categorías en n−1 variables binarias. (96–97) |
| **EDA** (*Exploratory Data Analysis*) | Análisis exploratorio, clasificado en **no gráfica** y **gráfica**; se orienta por distribuciones, outliers, relaciones e hipótesis. (61) |
| **Exactitud** (*Accuracy*) | Criterio de calidad: los datos reflejan la realidad (ej. temperatura = 400 °C → incorrecto). (75) |
| **GIGO** | *"Garbage in, garbage out"*: si los datos de entrada son malos, la salida también lo será. (72, 103) |
| **Hiperónimos** | Representan una relación de orden superior desde una perspectiva semántica. (89) |
| **IA centrada en datos** | Enfatiza que el diseño y la ingeniería de datos son esenciales. (101) |
| **IA centrada en modelos** | Enfatiza la elección del tipo de modelo, la arquitectura y los hiperparámetros adecuados. (101) |
| **Iniciación** | Primera fase donde se establecen las bases: preparar los datos, definir el problema y traducir a IA. (16) |
| **Instancias** | Las filas de un dataset; en Iris, $N = 150$. (30) |
| **Ingeniería de atributos** | Componente **G**: convierte los datos preprocesados en los atributos $f_i$ que consume el modelo; permite incorporar conocimiento del dominio. (94) |
| **Lemmatization** | Busca cada palabra en un diccionario y analiza el POS para seleccionar la forma correcta en caso de desambiguación. (88) |
| **Muestreo** | Selección de un subconjunto representativo de una gran cantidad de datos. (43) |
| **Muestreo aleatorio simple** | Cada punto de datos tiene la misma probabilidad de ser seleccionado. (44) |
| **Muestreo sistemático** | Se ordenan los datos por un criterio, se elige un punto de inicio al azar y se toma cada punto de datos a una distancia específica. (44) |
| **Muestreo estratificado** | Cada punto se asigna a una categoría; en cada categoría se hace muestreo aleatorio simple y la unión constituye el subconjunto final. (44) |
| **Muestreo por clusters** | Cada punto se asigna a un cluster; algunos clústeres se seleccionan aleatoriamente y su unión constituye el subconjunto final. (44) |
| **N-gramas** | Conjuntos de palabras o conceptos que coexisten con la ventana *n* dada. (89) |
| **Normalización** | Técnica para transformar los rangos de diferentes atributos en uno solo, porque algunos algoritmos priorizan los atributos con mayor rango. (90) |
| **Objetivos** | Describen el objetivo principal y otras preguntas relacionadas relevantes: ¿qué quiero lograr? (19) |
| **One-hot encoding** | Convierte n categorías en n variables binarias; una vale 1 y las demás 0. (96–97) |
| **PCA** | Reduce características correlacionadas a un conjunto menor de componentes no correlacionados, conservando la mayor cantidad posible de información. (98) |
| **POS** (*Part-of-speech*) | Proceso de identificar clases de palabras gramaticales. (88) |
| **Redondeo** | Reduce la precisión de los puntos de datos al convertir los valores en números enteros. (95) |
| **Rigor** (en recopilación) | La recopilación debe ser completa y sin lagunas respecto del objeto de interés. (33) |
| **Stemming** | Proceso algorítmico que reduce las palabras a su raíz; el resultado no incluye necesariamente palabras reales. (88) |
| **Suavizado** | Técnica para reducir el efecto de los valores atípicos, ponderándolos con menor peso; elimina ruido y revela los datos subyacentes. (90) |
| **Sustitución** | Eliminación de símbolos, palabras o caracteres irrelevantes (emoticones, palabras vacías). (89) |
| **Tidy Data** | Formato que requiere una columna por cada variable y una fila por cada observación. (27) |
| **Tokenización** | Proceso de separación de una cadena de texto en las palabras que la componen. (88) |
| **Transformación monótonica** | Transformación logarítmica o Box-Cox: estabiliza la varianza y acerca los datos a una distribución normal. (95) |
| **Unicidad** | Criterio de calidad: ausencia de duplicados. (75) |
| **Validez** | Criterio de calidad: los datos cumplen con las reglas de formato (ej. email con @). (75) |
| **Variable objetivo / etiqueta** | La columna que se quiere predecir; es la variable **dependiente** $Y$. (30, 94) |

---

## 15. Checklists de síntesis

> **Nota de organización:** esta sección **reorganiza criterios ya enunciados en el deck** para
> facilitar su uso; **no añade teoría nueva**. Cada ítem remite a la diapositiva de origen.

### 15.1 Checklist de la etapa de iniciación (componentes A–G, diap. 14 y 17)

- [ ] **A — Planteamiento del problema:** ¿está traducido a una tarea de ML (clasificación,
      regresión, clustering)? ¿hay objetivos y criterios de éxito medibles? (19, 23)
- [ ] **B — Recopilación de datos:** ¿la recopilación es rigurosa (completa y sin lagunas)? ¿se
      documentó la estructura y la fuente? (33, 34)
- [ ] **C — Muestreo:** ¿la técnica elegida preserva la representatividad que el problema exige?
      (43–44)
- [ ] **D — Distribución de datos:** ¿se conoce la distribución de la variable objetivo? (69)
- [ ] **E — Calidad de los datos:** ¿se evaluaron los seis criterios? (75)
- [ ] **F — Preprocesamiento:** ¿qué métodos se aplicaron y por qué? (14)
- [ ] **G — Ingeniería de atributos y vectorización:** ¿qué atributos se construyeron y por qué?
      (14)

### 15.2 Guía de elección de técnica de muestreo (diap. 44 y ejercicios 46–56)

| Si necesitas… | Técnica del curso | Qué hace |
|---|---|---|
| Rapidez y simplicidad, sin garantía de representatividad | **Aleatorio simple** | Misma probabilidad para cada punto (47) |
| Recorrer un orden conocido de forma regular | **Sistemático** | Ordenar por un criterio y tomar cada *k*-ésimo (50) |
| Preservar la proporción de una variable (ej. género) | **Estratificado** | Muestreo aleatorio simple dentro de cada categoría (53) |
| Trabajar con grupos completos ya formados | **Por clusters** | Seleccionar clústeres enteros al azar (56) |

### 15.3 Checklist de calidad de datos (los seis criterios, diap. 75)

- [ ] **Completitud** — ¿hay campos vacíos?
- [ ] **Unicidad** — ¿hay duplicados (ej. un mismo DNI en dos registros)?
- [ ] **Actualidad** — ¿los datos son suficientemente recientes? ¿hay fechas futuras imposibles?
- [ ] **Validez** — ¿los valores cumplen las reglas de formato del campo?
- [ ] **Exactitud** — ¿los valores son posibles en la realidad?
- [ ] **Consistencia** — ¿dos fuentes se contradicen sobre el mismo hecho?

### 15.4 Guía rápida de preprocesamiento según el tipo de dato

| Tipo de dato | Técnicas del curso | Diap. |
|---|---|---|
| **Imágenes** | Corrección de sensor, de iluminación, de ruido, geométrica, de color | 86 |
| **Texto** | Tokenización, stemming, POS, lemmatization, sustitución, n-gramas, sinónimos, hiperónimos | 87–89 |
| **Números** | Suavizado, normalización | 90 |
| **Series temporales** | Reemplazo de valores faltantes, representación, alineamiento | 91 |
| **Numéricos (atributos)** | Medidas estadísticas, binarización, redondeo, binning, transformación monótonica | 95 |
| **Categóricos** | Transformación (nominal/ordinal) + encoding (one-hot / dummy) | 96–97 |

---

## 16. Notas de fidelidad sobre la reconstrucción

- Las **fórmulas de la diapositiva 62** se transcribieron leyendo la imagen de la diapositiva. La
  capa de texto las entrega con los subíndices y exponentes desplazados o perdidos (por ejemplo,
  `σ2` por $\sigma^2$ y `Xn/2` por $X_{n/2}$), por lo que no es una fuente fiable para ellas.
- ⚠ **Corrección relevante respecto de la capa de texto:** en la diapositiva 90, la extracción de
  texto plano asocia la definición de *suavizado* al título **Normalización** y viceversa. La
  lectura de la imagen muestra que **Suavizado** = reducir el efecto de los atípicos y
  **Normalización** = unificar rangos. Esta guía usa la asignación correcta (§9.3).
- Contenido recuperado **solo por lectura visual** (ausente o irrecuperable en la capa de texto):
  las **filas seleccionadas en los cuatro ejercicios de muestreo** (diap. 47, 50, 53, 55–56), que
  el original marca **por color**; la lámina de **visualización de datos** (diap. 70); la figura
  **"Promediar categorías codificadas puede ser engañoso"** (diap. 68); el **ejemplo numérico de
  media vs mediana** (diap. 63–67); la tabla **Iris** con N=150 y d=4 (diap. 30); el ejemplo en
  tres formatos de estructura (diap. 28); y el diagrama **modelo-céntrico vs dato-céntrico**
  (diap. 101).
- **Ejercicios numéricos verificados:** la aritmética de §7.3 (medias 3 y 4), §7.4 ($(1+5)/2=3$),
  §6.5 (posiciones 1, 4, 7, 10 con paso 3), §6.6 (proporción 50/50 preservada) y §10.4 (3
  columnas one-hot vs 2 dummy) **cuadra** con lo mostrado en las diapositivas.
- **Preguntas de clase sin respuesta marcada en el deck:** diapositivas 13, 42, 58 y 74. Se
  transcriben como preguntas abiertas; esta guía **no** indica una opción correcta porque el
  original no la señala.
- Elementos marcados **[verificar]**: el final truncado de la viñeta sobre *dummy encoding* en la
  diapositiva 96 ("una categoría queda represent"), que aparece cortado en la propia diapositiva.
- Erratas del original conservadas y señaladas en su lugar: "recopilaciónb" (diap. 6).

---

*Guía de estudio elaborada a partir de `Sem3_Iniciación.pdf` (108 diapositivas).*
*Curso: **Planificación y Toma de Decisiones en IA**, Dra. Aurea Soriano-Vargas — UTEC, 2026.*
