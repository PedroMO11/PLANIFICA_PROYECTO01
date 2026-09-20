# Semana 5 — Evaluación · Guía de estudio

| Campo | Valor |
|---|---|
| **Curso** | Planificación y Toma de Decisiones en IA |
| **Sesión** | Semana 5 · *Evaluación* · 2026-2 |
| **Docente** | Dra. Aurea Soriano-Vargas (`aureasoriano` · asoriano@utec.edu.pe · `aurea-soriano`) |
| **Institución** | UTEC |
| **Fuente** | `Sem5_Evaluación.pdf` — 110 diapositivas |
| **Atribución del pie** | *Planificación y Toma de Decisiones en IA*, Dra. Aurea Soriano-Vargas (2026) |

> **Propósito de esta guía:** reconstruir fielmente el contenido teórico de las 110 diapositivas
> de la sesión —incluyendo fórmulas, tablas de diagnóstico y ejemplos numéricos que en el PDF
> están como imágenes— para poder estudiar la sesión sin depender del deck original.

---

## 1. Mapa del curso: ubicación de la sesión

Diagrama **"La historia de este curso"** (diapositivas 2 y 106).

```
Fundamentos        ┌──────────────────────┐
                   │ Motivación y         │  (gris = ya visto)
                   │ terminología         │
                   └──────────┬───────────┘
                              │
Ciclo de vida      ┌──────────▼───┐   ┌──────────────┐   ┌───────────┐   ┌──────────────────┐
de la IA           │  Iniciación  │──▶│  Modelado y  │──▶│ Despliegue│──▶│ Desviación       │
                   │   (gris)     │   │  rendimiento │   └───────────┘   │ conceptual       │
                   └──────────────┘   └──────────────┘    ↑ resaltado    │ (Concept Drift)  │
                                       ↑ resaltado en      en la diap.   └────────┬─────────┘
                                       la diap. 2          106 (PRÓXIMA          │
                                       (ESTA SESIÓN)       SEMANA)               │
                              ┌──────────────────────────────────────────────────┘
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

> **Nota importante de ubicación:** aunque el PDF se titula *Evaluación*, el mapa de apertura
> (diap. 2) sitúa esta sesión todavía dentro del bloque **"Modelado y rendimiento"** —es decir,
> es la **segunda parte** de ese bloque, iniciado en la Semana 4. El mapa de cierre (diap. 106)
> ya avanza a **Despliegue** como próxima semana.

### 1.1 Repaso de la sesión anterior (diap. 3–10)

El deck abre repasando las métricas de la Semana 4: matriz de confusión (diap. 3), Precision
(4), Recall (5), $F_\beta$ Score (6), espacio ROC (7), AUC (8), métricas de regresión (9) y las
conclusiones de Modelado (10).

### 1.2 Llevando la IA a la práctica (diap. 11)

Imagen de apertura del tema: el meme *"SNOWMEN: expectation vs reality"* — la distancia entre lo
que se espera de un modelo y lo que se obtiene en la práctica.

---

## 2. Principales Desafíos (diap. 12–19)

El deck organiza los desafíos en dos grandes grupos (diap. 13, 19, 77):

| **"Malos datos"** | **"Malos Algoritmos"** |
|---|---|
| Cantidad insuficiente de datos de entrenamiento | Overfitting |
| Datos de entrenamiento no representativos | Underfitting |
| Datos de baja calidad | |
| Características irrelevantes | |

### 2.1 Los cuatro problemas de "malos datos" (diap. 14–18)

| Desafío | Definición del curso |
|---|---|
| **Cantidad insuficiente de datos de entrenamiento** | El modelo aprende a partir de un conjunto demasiado pequeño para capturar la complejidad real del problema. |
| **Datos de entrenamiento no representativos** | Fundamental que los datos de entrenamiento sean representativos de los nuevos casos a los que desea generalizar. |
| **Datos de mala calidad** | Si sus datos de entrenamiento están llenos de errores, valores atípicos y ruido, será más difícil para el sistema detectar los patrones subyacentes. |
| **Características irrelevantes** | Importante para el éxito → crear un buen conjunto de características para el entrenamiento. Esto se denomina **feature engineering**. |

**Feature engineering (diap. 17)** se descompone en:
- **Selección de características:** características más útiles.
- **Extracción de características:** combinación de características para producir una más útil.

**Síntesis (diap. 18).** Los cuatro problemas se presentan bajo un paraguas con el rótulo
**"(basura que entra, basura que sale) (GIGO)"**.

### 2.2 Ubicación en el ciclo de vida (diap. 20)

```
┌────────────┐   ┌───────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│ Iniciación │──▶│ Entrenamiento │──▶│ Estimación │──▶│ Validación │──▶│ Despliegue │
└────────────┘   └───────────────┘   └────────────┘   └────────────┘   └────────────┘
```

---

## 3. Objetivos y agenda de la sesión

### 3.1 Al finalizar la clase deberíamos tener los medios para responder (diap. 21)

1. Desbalance de clases y mitigación.
2. Errores de sobreajuste (*overfitting*) y subajuste (*underfitting*), así como la relación
   bias-varianza.
3. Métodos de validación y optimización (*holdout*, *k-fold cross-validation*, validación
   cruzada anidada, *grid/random search*).
4. Riesgos de *data leakage* y buenas prácticas.

### 3.2 Agenda (diap. 22)

1. Problemas comunes en la evaluación
2. Métodos de evaluación
3. Data Leakage
4. Selección de parámetros
5. Overfitting, underfitting y bias-variance tradeoff

---

## 4. Desbalance de clases (sección, diap. 23–35)

### 4.1 Definición y casos típicos (diap. 24)

> El desbalance de clases ocurre cuando **una clase tiene muchos menos ejemplos de entrenamiento
> que otra**.

**Ejemplos comunes:**
- Fraude con tarjetas de crédito
- Diagnóstico médico
- Detección de rostros
- Detección de objetos específicos

### 4.2 El efecto sobre la frontera de decisión (diap. 25–27)

> Cuando una clase está **subrepresentada**, el modelo puede favorecer la **clase mayoritaria**,
> afectando la clasificación de la minoritaria.

El deck muestra el mismo conjunto de datos (18 ejemplos de la clase mayoritaria **×** y 4 de la
minoritaria **▫**) con dos fronteras de decisión distintas:

| Frontera | Aciertos clase × | Aciertos clase ▫ | ACC |
|---|---|---|---|
| **Frontera A** (diap. 26) — inclinada, deja pasar parte de la minoritaria | **12 / 18** | **4 / 4** | **0.73** |
| **Frontera B** (diap. 27) — favorece a la mayoritaria | **18 / 18** | **2 / 4** | **0.91** |

> ✔ **Verificación aritmética:** con $n = 18 + 4 = 22$ ejemplos,
> Frontera A: $(12+4)/22 = 16/22 = 0.7273 \approx 0.73$;
> Frontera B: $(18+2)/22 = 20/22 = 0.9091 \approx 0.91$. Ambas cuadran.
>
> **La lección del ejemplo:** la frontera B tiene **mayor accuracy** (0.91 vs 0.73) pero
> clasifica **peor la clase minoritaria** (2/4 vs 4/4). La accuracy premia a quien ignora la
> clase rara.

### 4.3 El problema con los datos nuevos (diap. 28)

> Cuando lleguen nuevos ejemplos de prueba **▫**, lo más probable es que se clasifiquen
> incorrectamente como **×**.
>
> **¿Cómo afrontar el desbalance?**

### 4.4 Formas de abordar datos desbalanceados (diap. 29–30)

> **La mejor alternativa es recopilar más datos de la clase subrepresentada**, pero esto no
> siempre es posible.

### 4.5 Diferentes pesos para cada clase (diap. 31)

- Sin balance, el modelo tiende a **favorecer la clase mayoritaria**.
- Esto ocurre porque su error tiene mayor influencia durante el entrenamiento.
- Una estrategia es **modificar la función de costo** para considerar el desbalance.
- Para ello, se asigna un **peso mayor a los errores de la clase minoritaria**.
- Así, el modelo presta más atención a ambas clases durante el aprendizaje.

**Formulación original de la función de costo de regresión logística:**

$$J(\theta) = -\frac{1}{m}\left[\sum_{i=1}^{m} y^{(i)} log(h_\theta(x^{(i)})) + (1 - y^{(i)}) log(1 - h_\theta(x^{(i)})) + \lambda \sum_{j=1}^{n} \theta_j^{2}\right]$$

> El término resaltado $y^{(i)} log(h_\theta(x^{(i)})) + (1-y^{(i)})log(1-h_\theta(x^{(i)}))$ es
> el **error en el i-ésimo ejemplo de entrenamiento**.

**Modificando para considerar el desbalance entre clases:**

$$J(\theta) = -\frac{1}{m}\left[\sum_{i=1}^{m} w_{pos}\, y^{(i)} log(h_\theta(x^{(i)})) + w_{neg}\,(1 - y^{(i)}) log(1 - h_\theta(x^{(i)})) + \lambda \sum_{j=1}^{n} \theta_j^{2}\right]$$

donde:
- $w_{pos}$ = **peso asociado a los ejemplos de la clase positiva**.
- $w_{neg}$ = **peso asociado a los ejemplos de la clase negativa**.

### 4.6 Ejemplo numérico de los pesos (diap. 32)

> Supongamos que **la clase positiva tiene 100 ejemplos de entrenamiento** y **la clase negativa,
> 900**. Podemos definir los pesos **inversamente proporcionales al tamaño de las clases**:

$$w_{pos} = 1 - \frac{100}{900 + 100} = 0.9$$

$$w_{neg} = 1 - \frac{900}{900 + 100} = 0.1$$

**Por lo tanto:**

$$J(\theta) = -\frac{1}{m}\left[\sum_{i=1}^{m} 0.9\, y^{(i)} log(h_\theta(x^{(i)})) + 0.1\,(1 - y^{(i)}) log(1 - h_\theta(x^{(i)})) + \lambda \sum_{j=1}^{n} \theta_j^{2}\right]$$

> ✔ **Verificación aritmética:** $1 - 100/1000 = 0.9$ y $1 - 900/1000 = 0.1$. Correcto.
> La clase minoritaria (positiva, 100 ejemplos) recibe un peso **9 veces mayor**.

### 4.7 Undersampling (diap. 33)

> **Seleccionar sin repetición** ejemplos de la clase mayoritaria para lograr un conjunto de
> entrenamiento más balanceado.

### 4.8 Oversampling (diap. 34)

> **Repetir ejemplos de la clase minoritaria** hasta equilibrar la cantidad de muestras entre
> ambas clases.

El diagrama marca los ejemplos repetidos con multiplicadores: **3x**, **4x**, **5x**, **6x**.

### 4.9 SMOTE — *Synthetic Minority Oversampling TEchnique* (diap. 35)

> SMOTE selecciona a los **vecinos más cercanos** de la clase minoritaria y genera **ejemplos
> artificiales** a partir de combinaciones entre sus características.

```
        ┌───┐
        │ A │
        └───┘
              ┌───┐
              │ C │   ← Los datos artificiales C se crean promediando A y B
              └───┘
                     ┌───┐
                     │ B │
                     └───┘
```

---

## 5. Opciones de evaluación · Organización de datos

### 5.1 ¿Qué modelo es mejor? (diap. 37–39)

El deck presenta dos ajustes sobre los mismos datos: a la izquierda una **recta**, a la derecha
una **curva muy ondulada** que pasa más cerca de los puntos. La pregunta queda abierta hasta
introducir la idea de prueba.

### 5.2 ¿Por qué hacer pruebas? (diap. 40–46)

Leyenda del diagrama (diap. 45): **círculos rellenos = Train**, **círculos huecos = Test**.

Con los datos de entrenamiento ocultos y solo los de prueba visibles:

| Modelo | Comportamiento | Veredicto del deck |
|---|---|---|
| **Recta** (izquierda) | Los puntos de prueba caen cerca de la recta | ✔ (marca verde) |
| **Curva ondulada** (derecha) | Se ajusta al entrenamiento pero falla en los puntos de prueba | ✗ (marca roja) |

> **Moraleja del deck (diap. 49):** *"Los amigos no permiten que sus amigos utilicen datos de
> prueba para entrenamiento."*

### 5.3 El punto de partida: un solo conjunto de datos (diap. 47)

> En un problema real, normalmente solo nos encontramos con **un conjunto de datos**.
>
> **¿Cómo organizar estos datos para realizar el entrenamiento y obtener una estimación
> confiable del rendimiento del modelo?**

### 5.4 Holdout (diap. 48, 51)

> El primer enfoque sería **dividir aleatoriamente** los datos en un conjunto de entrenamiento y
> un conjunto de validación:

```
┌──────────────────────── Datos ────────────────────────┐
├────────────── 80% ──────────────┬─────── 20% ─────────┤
│         Entrenamiento           │       Prueba        │
└─────────────────────────────────┴─────────────────────┘
```

**Limitaciones del Holdout (diap. 51):**
- La división aleatoria puede generar conjuntos de entrenamiento y prueba con **distinta
  dificultad**.
- Si el entrenamiento contiene ejemplos fáciles y la prueba ejemplos complejos, el modelo puede
  **no generalizar bien**.
- Al reservar un 20 % para prueba, esos datos **no participan en el entrenamiento**.
- Algunos ejemplos excluidos podrían ser importantes para mejorar la generalización del modelo.

### 5.5 División porcentual y muestreo: ejemplo de clasificación binaria (diap. 50)

Esquema que muestra **dónde se aplica cada técnica de remuestreo**:

```
Datos originales (desbalanceados)
  [ + Positivo ][─────────── − Negativo ───────────]

Datos de entrenamiento (Undersampling)
  [ A ]                    [ C ]

Datos de entrenamiento (Oversampling)
  [ A ][ A ]               [ D with C ⊂ D ]

Datos de entrenamiento (SMOTE)
  [ A ][ Ã (Synthetic) ]   [ D with C ⊂ D ]
─────────────────────────────────────────────────── (línea divisoria)
Datos de Prueba
  [ B ]                    [ E ]
```

> **Lectura clave:** el remuestreo (undersampling, oversampling, SMOTE) se aplica **solo al
> conjunto de entrenamiento**. Los **Datos de Prueba (B y E)** conservan la proporción original
> desbalanceada.

### 5.6 Random Sampling / Holdout repetido (diap. 52–53)

> Repetimos el Holdout varias veces generando diferentes combinaciones de Entrenamiento y
> Validación:

```
Datos ──┬──▶ [ Entrenamiento 1 ][ Prueba 1 ]
        ├──▶ [ Entrenamiento 2 ][ Prueba 2 ]
        │              ...
        └──▶ [ Entrenamiento n ][ Prueba n ]
```

> **Calculamos la media y la desviación estándar en los conjuntos.** (diap. 53)

### 5.7 La partición en tres: entrenamiento, validación y prueba (diap. 54)

```
┌────────────────────────── Datos ──────────────────────────┐
             ▼                                    ▼
┌──────────────────────────────────┬────────────────────────┐
│          Entrenamiento           │         Prueba         │
└──────────────────────────────────┴────────────────────────┘
      ▼                    ▼
┌──────────────────┬───────────────┬────────────────────────┐
│  Entrenamiento   │  Validación   │         Prueba         │
└──────────────────┴───────────────┴────────────────────────┘
```

> El conjunto de **Prueba** se aparta desde el inicio y no se toca; el de **Entrenamiento** se
> subdivide en Entrenamiento + Validación.

### 5.8 k-fold Cross Validation (diap. 55–62)

Con **k = 5**, los datos se dividen en 5 *folds*; en cada iteración uno distinto actúa como
**Validación** y los otros cuatro como **Entrenamiento**:

```
Iteración 1:  [Train][Train][Train][Train][ VAL ]
Iteración 2:  [Train][Train][Train][ VAL ][Train]
Iteración 3:  [Train][Train][ VAL ][Train][Train]
Iteración 4:  [Train][ VAL ][Train][Train][Train]
Iteración 5:  [ VAL ][Train][Train][Train][Train]
```

**Elección de k (diap. 62):**

> Generalmente, usamos **k = {3, 5, 10}**.
>
> Cuanto **mayor** sea k:
> - **Menor** será la varianza de la estimación del rendimiento (la estimación se vuelve más
>   fiable);
> - **Mayor** será el costo computacional;

**Leave-one-out:**

> Variación de la validación cruzada donde **k = n** (número de ejemplos). Cada fold contiene un
> ejemplo y el entrenamiento se realiza con **n−1** muestras. La validación contiene solo una
> muestra.

### 5.9 Stratified K-Fold Cross-Validation (diap. 63)

> **Mantiene la misma proporción de ejemplos de cada clase** en el conjunto original en cada
> fold.

El diagrama ilustra un conjunto original con **clase mayoritaria (azul) = 80 %** y **clase
minoritaria (naranja) = 20 %**; cada uno de los 5 folds conserva esa proporción
(**≈ 80 % / ≈ 20 %**).

### 5.10 Comparación de múltiples modelos entrenados (diap. 64)

- Para comparar varias configuraciones de modelos, se utilizan **los mismos folds**.
- Se aplica validación cruzada a cada configuración.
- Se comparan **la media del rendimiento y su desviación** entre configuraciones.
- Esto permite identificar qué modelo ofrece **mejor rendimiento y mayor estabilidad**.

---

## 6. Data Leakage (sección, diap. 65–70)

### 6.1 Definición (diap. 66)

> **Rendimiento "demasiado bueno para ser verdad" — Fuga de datos**
>
> **Definición Data Leakage:** Uso durante el entrenamiento de información que **no estará
> disponible al predecir**, generando **métricas artificialmente altas** y un rendimiento real
> menor en producción.

### 6.2 La tabla de ejemplo (diap. 66–69)

| Empleado ID | Edad | Sueldo Mensual | Año | Trabajo | **Sueldo Anual** *(variable objetivo)* |
|---|---|---|---|---|---|
| 1 | 30 | 5000 | 2023 | Programador | 60000 |
| 2 | 25 | 4500 | 2023 | Analista de Datos | 54000 |
| 3 | 35 | 5500 | 2023 | Gestor | 66000 |
| 4 | 28 | 4800 | 2023 | Programador | 57600 |
| 5 | 40 | 6000 | 2023 | Analista de Datos | 72000 |
| 1 | 31 | 5100 | 2024 | Programador | 61200 |
| 2 | 26 | 4600 | 2024 | Analista de Datos | 55200 |

> ✔ **Observación sobre los datos:** en todas las filas se cumple
> $\text{Sueldo Anual} = \text{Sueldo Mensual} \times 12$
> (p. ej. $5000 \times 12 = 60000$; $4500 \times 12 = 54000$; $5100 \times 12 = 61200$).
> Esa relación exacta es justamente lo que hace del sueldo anual un caso de *feature leakage*.

### 6.3 Los tres tipos de leakage (diap. 67–69)

| Tipo | Definición del curso | Ejemplo en la tabla |
|---|---|---|
| **Feature Leakage** | Cuando una característica **contiene información sobre la variable objetivo**. | Usar **salario anual** para predecir **salario mensual**. |
| **Time Leakage** | Cuando el modelo **usa datos del futuro** durante el entrenamiento. | Mezclar datos de **2023 y 2024** en un problema temporal. |
| **Group Leakage** | Cuando **la misma entidad aparece en entrenamiento y prueba**. El modelo puede **memorizar patrones individuales** en lugar de generalizar. | El **mismo empleado** (IDs 1 y 2) aparece en train y test en años distintos. |

### 6.4 Precaución al preprocesar datos (diap. 70)

**Ejemplo:** Normalizar **todo** el conjunto de datos **antes** de dividirlo en entrenamiento y
prueba.

**Problema:** El modelo obtiene información sobre la **distribución del conjunto de prueba**,
generando data leakage y métricas demasiado optimistas.

**Solución:** Dividir **primero** los datos en entrenamiento y prueba. Luego, calcular los
parámetros de normalización **solo con el conjunto de entrenamiento** y aplicar esa misma
transformación al conjunto de prueba.

**❌ Incorrecto (genera leakage):**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing()
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

**✅ Correcto:**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing()
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

> La diferencia está en el **orden**: en el código correcto, `scaler.fit()` se ejecuta
> **después** de `train_test_split` y **solo sobre `X_train`**.

---

## 7. Selección de parámetros (sección, diap. 71–75)

### 7.1 La pregunta (diap. 72)

> **¿Cómo sabemos que los parámetros que elegimos son "los mejores"?**

### 7.2 Los tres métodos de búsqueda de hiperparámetros (diap. 73–75)

| | **GRID SEARCH** | **RANDOM SEARCH** | **OPTUNA (BAYESIAN OPTIMIZATION)** |
|---|---|---|---|
| **Qué hace** | Evalúa **todas las combinaciones posibles** en una cuadrícula definida. | Evalúa **combinaciones aleatorias** dentro del espacio de búsqueda. | Selecciona las siguientes combinaciones de **forma inteligente, aprendiendo de evaluaciones previas**. |
| **Ventajas** | • **Exhaustivo**: evalúa todas las combinaciones.<br>• Simple de entender e implementar. | • Más **simple** y **eficiente** que Grid Search.<br>• Explora un espacio más amplio. | • **Muy eficiente**: encuentra buenas configuraciones rápidamente.<br>• Ideal para espacios de búsqueda grandes o costosos de evaluar.<br>• Aprovecha resultados anteriores para orientar las siguientes pruebas. |
| **Desventajas** | • **Muy costoso** computacionalmente.<br>• Escala mal cuando aumentan los hiperparámetros. | • Puede **perder combinaciones óptimas**.<br>• Los resultados pueden variar entre ejecuciones. | • Más complejo de configurar y entender.<br>• Puede tener dificultades cuando el espacio de búsqueda o la función objetivo son muy ruidosos. |

**Representación gráfica (diap. 75).** Los tres diagramas usan los ejes *Hiperparámetro 1* e
*Hiperparámetro 2*:

- **Grid Search:** puntos distribuidos en una **cuadrícula regular**; una estrella marca la mejor
  combinación. Leyenda: *Combinaciones evaluadas* · *Mejor combinación*.
- **Random Search:** puntos **dispersos aleatoriamente**. Leyenda: *Combinaciones evaluadas
  aleatoriamente* · *Mejor combinación*.
- **Optuna:** puntos **conectados secuencialmente** por flechas, sobre un mapa de calor que va de
  *Peor* a *Mejor* (**Rendimiento (objetivo)**), concentrándose alrededor del óptimo. Leyenda:
  *Combinaciones evaluadas (secuencialmente)* · *Siguiente combinación sugerida* · *Mejor
  combinación*.

---

## 8. Malos algoritmos: Overfitting y Underfitting (sección, diap. 76–88)

### 8.1 Overfitting (diap. 78–84)

> El **sobreajuste** significa que el modelo funciona bien en los datos de entrenamiento,
> **pero no generaliza**.

| | **F1-Score** |
|---|---|
| **Entrenamiento** | **0.97** |
| **Prueba** | **0.51** |

> El sobreajuste ocurre cuando el modelo es **demasiado complejo** en relación con la cantidad y
> el ruido de los datos de entrenamiento. (diap. 84)

### 8.2 Underfitting (diap. 85–88)

> El **subajuste** es lo opuesto al sobreajuste: ocurre cuando el modelo es **demasiado simple**
> para aprender la estructura subyacente de los datos.

| | **F1-Score** |
|---|---|
| **Entrenamiento** | **0.43** |
| **Prueba** | *(celda vacía en el original)* |

> ⚠ **Nota de fidelidad:** en la última diapositiva de esta secuencia (diap. 88) la celda
> *Prueba* **queda vacía**; el deck **no proporciona** el F1 de prueba para el caso de
> underfitting. No se completa aquí para no inventar un dato. El punto pedagógico se sostiene
> igual: con subajuste el rendimiento **ya es malo en entrenamiento** (0.43).

---

## 9. Bias vs Varianza (diap. 89–102)

### 9.1 Definiciones (diap. 89)

| **BIAS** | **VARIANZA** |
|---|---|
| Error por un modelo **demasiado simple**. | **Sensibilidad a cambios** en los datos de entrenamiento. |
| **Bias alto → underfitting**: no aprende bien los patrones. | **Varianza alta → overfitting**: aprende demasiado el entrenamiento y generaliza mal. |

### 9.2 Tabla de referencia: subajuste / ajuste correcto / sobreajuste (diap. 93–97)

*(Tabla que en el PDF está íntegramente como imagen; transcrita de la lectura ampliada.)*

| | **Subajuste** | **Ajuste correcto** | **Sobreajuste** |
|---|---|---|---|
| **Síntomas** | • Error de entrenamiento **alto**<br>• Error de entrenamiento **cercano** al error de prueba<br>• Alto sesgo | • Error de entrenamiento **ligeramente inferior** al de prueba | • Error de entrenamiento **muy bajo**<br>• Error de entrenamiento **mucho menor** al de prueba<br>• Alta varianza |
| **Ilustración regresión** | Recta que no sigue la curvatura de los datos | Parábola que sigue la tendencia | Curva muy quebrada que pasa por casi todos los puntos |
| **Ilustración clasificación** | Recta que separa mal las dos nubes | Frontera curva suave | Frontera muy irregular que rodea puntos individuales |
| **Ilustración redes profundas** | Ambas curvas de error descienden y se estabilizan **juntas y altas** | Las dos curvas descienden, con una brecha **pequeña** | La curva de validación **sube** tras bajar, mientras la otra sigue descendiendo (brecha creciente) |
| **Posibles soluciones** | • **Complejizar el modelo**<br>• **Agregar más features**<br>• **Entrenar más tiempo** | *(celda vacía en el original)* | • **Aplicar regularización**<br>• **Obtener más datos** |

> ⚠ **Inconsistencia detectada en la figura original (no corregida).** En la fila *"Ilustración
> redes profundas"*, la curva inferior (roja) aparece etiquetada como **"Prueba"** en los paneles
> de *Subajuste* y *Sobreajuste*, pero como **"Training"** en el panel de *Ajuste correcto*.
> Por la lógica del propio gráfico —la curva que desciende hasta valores muy bajos mientras la de
> validación sube es, por definición, la de entrenamiento— la etiqueta **"Prueba"** del panel de
> sobreajuste parece un error de rotulación del original. Se transcribe tal como aparece.
> **[verificar con la docente]**

### 9.3 Diagnóstico por comparación train/test — Regresión (diap. 90–93)

**Error Regresión** *(la métrica es el **error**: más bajo es mejor)*

| | Caso 1 | Caso 2 | Caso 3 | Caso 4 |
|---|---|---|---|---|
| **Train** | 1 % | 15 % | 15 % | 0.5 % |
| **Test** | 11 % | 16 % | 30 % | 1 % |
| **Diagnóstico** | **Varianza alta** | **Bias alto** | **Bias y varianza altos** | **Bias y varianza bajos** |

**Cómo se lee cada caso:**
- **Caso 1** — error de entrenamiento bajo (1 %) pero de prueba mucho mayor (11 %): la brecha
  grande indica **varianza alta** (sobreajuste).
- **Caso 2** — ambos errores altos y **cercanos entre sí** (15 % y 16 %): **bias alto**
  (subajuste).
- **Caso 3** — error de entrenamiento alto (15 %) **y** brecha grande hasta prueba (30 %):
  **bias y varianza altos**.
- **Caso 4** — ambos errores muy bajos y cercanos (0.5 % y 1 %): **bias y varianza bajos**.

### 9.4 Diagnóstico por comparación train/test — Clasificación (diap. 94–97)

**Accuracy Clasificación** *(la métrica es la **accuracy**: más alta es mejor — la lectura se
invierte respecto de la tabla anterior)*

| | Caso 1 | Caso 2 | Caso 3 | Caso 4 |
|---|---|---|---|---|
| **Train** | 94 % | 49 % | 51 % | 95 % |
| **Test** | 51 % | 48 % | 20 % | 93 % |
| **Diagnóstico** | **Varianza alta** | **Bias alto** | **Bias y varianza altos** | **Bias y varianza bajos** |

**Cómo se lee cada caso:**
- **Caso 1** — accuracy alta en entrenamiento (94 %) y baja en prueba (51 %): **varianza alta**.
- **Caso 2** — accuracy baja en ambos y **cercanas** (49 % y 48 %): **bias alto**.
- **Caso 3** — accuracy baja en entrenamiento (51 %) **y** aún peor en prueba (20 %): **bias y
  varianza altos**.
- **Caso 4** — accuracy alta en ambos y cercanas (95 % y 93 %): **bias y varianza bajos**.

> **Patrón común a ambas tablas:** el **bias** se diagnostica por el **nivel** del rendimiento en
> entrenamiento; la **varianza**, por la **brecha** entre entrenamiento y prueba.

### 9.5 Diagnosticando Bias vs Varianza: la curva de complejidad (diap. 98–102)

```
error
  ▲
  │╲                                          ╱  ← Error de Validación
  │ ╲                                       ╱
  │  ╲___                              ___╱
  │      ╲___                    ___╱
  │          ╲___ ○ Balanceado ╱
  │              ╲___
  │                  ╲_____________  ← Error de entrenamiento
  └──────────────────────────────────────────▶
   ▲                                    ▲
  Subajuste                        Sobreajuste
              Complejidad del Modelo
```

| Zona | Característica |
|---|---|
| **Subajuste** (izquierda, baja complejidad) | Ambos errores altos |
| **Balanceado** (centro) | El error de validación alcanza su **mínimo** |
| **Sobreajuste** (derecha, alta complejidad) | El error de entrenamiento sigue bajando pero el de **validación sube** |

**Referencia citada en la diapositiva 102:**
*Understanding the Bias-Variance Tradeoff:* `http://scott.fortmann-roe.com/docs/BiasVariance.html`

---

## 10. Conclusiones (diap. 103–104)

| Idea clave | Desarrollo |
|---|---|
| **No existe un modelo universalmente mejor** | Su rendimiento depende de la naturaleza de los datos, su representatividad y calidad. |
| **El desbalance de clases y el data leakage** | Son amenazas comunes que distorsionan la evaluación y deben abordarse cuidadosamente. |
| **El sobreajuste y el subajuste** | Representan extremos de la complejidad del modelo: el desafío es encontrar el equilibrio. |

---

## 11. Próxima semana (diap. 105–106)

En el mapa "La historia de este curso", el recuadro resaltado pasa a **Despliegue**.

*(Diap. 107: "¡Gracias!". Diap. 110: portada de cierre.)*

---

## 12. Glosario de términos del curso

> **Nota de organización:** este glosario **reordena y recoge las definiciones tal como las da el
> deck**; no añade teoría nueva. La diapositiva de origen se indica entre paréntesis.

| Término | Definición según el curso |
|---|---|
| **Bias** | Error por un modelo demasiado simple; bias alto → underfitting: no aprende bien los patrones. (89) |
| **Características irrelevantes** | Desafío de "malos datos"; se aborda con *feature engineering*. (17) |
| **Data Leakage** | Uso durante el entrenamiento de información que no estará disponible al predecir, generando métricas artificialmente altas y un rendimiento real menor en producción. (66) |
| **Desbalance de clases** | Ocurre cuando una clase tiene muchos menos ejemplos de entrenamiento que otra. (24) |
| **Extracción de características** | Combinación de características para producir una más útil. (17) |
| **Feature Leakage** | Cuando una característica contiene información sobre la variable objetivo. Ej.: usar salario anual para predecir salario mensual. (67) |
| **GIGO** | "Basura que entra, basura que sale"; engloba los cuatro desafíos de malos datos. (18) |
| **Grid Search** | Evalúa todas las combinaciones posibles en una cuadrícula definida; exhaustivo pero muy costoso. (74–75) |
| **Group Leakage** | Cuando la misma entidad aparece en entrenamiento y prueba; el modelo puede memorizar patrones individuales en lugar de generalizar. (69) |
| **Holdout** | Dividir aleatoriamente los datos en entrenamiento y validación (ej. 80 % / 20 %). (48) |
| **k-fold Cross Validation** | Dividir en k folds; cada uno actúa por turno como validación y el resto como entrenamiento. Generalmente k = {3, 5, 10}. (55–62) |
| **Leave-one-out** | Validación cruzada donde k = n; cada fold contiene un ejemplo y el entrenamiento usa n−1 muestras. (62) |
| **Optuna (Bayesian Optimization)** | Selecciona las siguientes combinaciones de forma inteligente, aprendiendo de evaluaciones previas. (74–75) |
| **Oversampling** | Repetir ejemplos de la clase minoritaria hasta equilibrar la cantidad de muestras entre ambas clases. (34) |
| **Overfitting (sobreajuste)** | El modelo funciona bien en los datos de entrenamiento pero no generaliza; ocurre cuando es demasiado complejo en relación con la cantidad y el ruido de los datos. (78, 84) |
| **Random Sampling** | Repetir el Holdout varias veces y calcular la media y la desviación estándar en los conjuntos. (52–53) |
| **Random Search** | Evalúa combinaciones aleatorias dentro del espacio de búsqueda; más eficiente que Grid Search pero puede perder combinaciones óptimas. (74–75) |
| **Selección de características** | Elegir las características más útiles. (17) |
| **SMOTE** | *Synthetic Minority Oversampling TEchnique*: selecciona los vecinos más cercanos de la clase minoritaria y genera ejemplos artificiales promediando sus características. (35) |
| **Stratified K-Fold** | Mantiene la misma proporción de ejemplos de cada clase del conjunto original en cada fold. (63) |
| **Time Leakage** | Cuando el modelo usa datos del futuro durante el entrenamiento. Ej.: mezclar datos de 2023 y 2024 en un problema temporal. (68) |
| **Undersampling** | Seleccionar sin repetición ejemplos de la clase mayoritaria para lograr un conjunto de entrenamiento más balanceado. (33) |
| **Underfitting (subajuste)** | Lo opuesto al sobreajuste: ocurre cuando el modelo es demasiado simple para aprender la estructura subyacente de los datos. (85) |
| **Varianza** | Sensibilidad a cambios en los datos de entrenamiento; varianza alta → overfitting. (89) |
| **$w_{pos}$ / $w_{neg}$** | Pesos asociados a los ejemplos de la clase positiva y negativa en la función de costo modificada para el desbalance. (31–32) |

---

## 13. Checklists de síntesis

> **Nota de organización:** esta sección **reorganiza criterios ya enunciados en el deck** para
> facilitar su uso; **no añade teoría nueva**. Cada ítem remite a la diapositiva de origen.

### 13.1 Checklist ante un conjunto desbalanceado (diap. 29–35, 50)

- [ ] ¿Es posible **recopilar más datos** de la clase subrepresentada? (la mejor alternativa) (29–30)
- [ ] Si no: ¿conviene **modificar la función de costo** con pesos por clase? (31)
- [ ] ¿Undersampling, oversampling o SMOTE? (33–35)
- [ ] ⚠ ¿El remuestreo se aplica **solo al conjunto de entrenamiento**, dejando la prueba con la
      proporción original? (50)
- [ ] ¿Se está evaluando con una métrica que **no premie ignorar la clase minoritaria**?
      (ver el ejemplo de §4.2, donde ACC sube de 0.73 a 0.91 empeorando la clase rara) (26–27)

### 13.2 Checklist anti-*data leakage* (diap. 66–70)

- [ ] **Feature Leakage:** ¿alguna característica contiene información sobre la variable
      objetivo? (67)
- [ ] **Time Leakage:** ¿se están mezclando periodos temporales que en producción no estarían
      disponibles? (68)
- [ ] **Group Leakage:** ¿la misma entidad aparece en train y test? (69)
- [ ] **Preprocesamiento:** ¿el `fit` del escalador se hace **después** de dividir y **solo** con
      el conjunto de entrenamiento? (70)
- [ ] ¿El rendimiento es "demasiado bueno para ser verdad"? Es la señal de alarma. (66)

### 13.3 Guía de diagnóstico bias/varianza (diap. 90–97)

| Observación en train vs test | Diagnóstico | Posibles soluciones (diap. 93) |
|---|---|---|
| Rendimiento **bueno en train**, **malo en test** (brecha grande) | **Varianza alta** → sobreajuste | Aplicar regularización · Obtener más datos |
| Rendimiento **malo en ambos** y **parecido** | **Bias alto** → subajuste | Complejizar el modelo · Agregar más features · Entrenar más tiempo |
| Rendimiento **malo en train** y **aún peor en test** | **Bias y varianza altos** | (combinar ambas familias de soluciones) |
| Rendimiento **bueno en ambos** y parecido | **Bias y varianza bajos** | — (objetivo alcanzado) |

> Regla de lectura: el **bias** se juzga por el **nivel** del rendimiento en entrenamiento; la
> **varianza**, por la **brecha** entre entrenamiento y prueba.

### 13.4 Guía de elección de método de validación (diap. 48–63)

| Si… | Método del curso |
|---|---|
| Quieres lo más rápido y simple | **Holdout** (80/20) — pero con las limitaciones de §5.4 |
| Quieres reducir el efecto de una división afortunada/desafortunada | **Random Sampling** (holdout repetido) + media y desviación |
| Quieres una estimación más fiable usando todos los datos | **k-fold Cross Validation** (k = 3, 5, 10) |
| Tienes pocos datos | **Leave-one-out** (k = n) |
| Tienes clases desbalanceadas | **Stratified K-Fold** |
| Quieres comparar configuraciones | Mismos folds para todas + comparar media y desviación (64) |

---

## 14. Notas de fidelidad sobre la reconstrucción

- **Las fórmulas de la función de costo (diap. 31–32) están en el PDF como imágenes.** La capa de
  texto solo recupera los rótulos de las llamadas ("Error en el i-ésimo ejemplo de
  entrenamiento", "Peso asociado a los ejemplos de la clase positiva/negativa") pero **ninguna de
  las dos ecuaciones**. Ambas se transcribieron leyendo la diapositiva rasterizada.
- ⚠ **Inconsistencia señalada, no corregida:** en la fila *"Ilustración redes profundas"* de la
  tabla de diagnóstico (diap. 93–97), la curva inferior aparece rotulada **"Prueba"** en los
  paneles de subajuste y sobreajuste, pero **"Training"** en el de ajuste correcto. Ver §9.2.
  **[verificar con la docente]**
- ⚠ **Dato ausente en el original:** la tabla de *Underfitting* (diap. 85–88) deja la celda
  **Prueba vacía**; el deck nunca da ese F1. No se rellena. Ver §8.2.
- **Ejemplos numéricos verificados y correctos:** las dos accuracies del ejemplo de desbalance,
  $(12+4)/22 = 0.73$ y $(18+2)/22 = 0.91$ (§4.2); los pesos $w_{pos} = 1 - 100/1000 = 0.9$ y
  $w_{neg} = 1 - 900/1000 = 0.1$ (§4.6). Todos cuadran con las diapositivas.
- **Coherencia comprobada en la tabla de data leakage:** en las 7 filas se cumple
  $\text{Sueldo Anual} = \text{Sueldo Mensual} \times 12$, que es precisamente lo que convierte
  esa columna en *feature leakage* (§6.2).
- **Contenido recuperado solo por lectura visual:** la **tabla de diagnóstico subajuste / ajuste
  correcto / sobreajuste** completa (diap. 93), que no deja rastro en la capa de texto y que hubo
  que **recortar y ampliar 4×** para leer con certeza; el esquema de **división porcentual y
  muestreo** (diap. 50); el diagrama de **Stratified K-Fold** con sus porcentajes (diap. 63); los
  tres gráficos de **búsqueda de hiperparámetros** con sus leyendas (diap. 75); la **curva de
  complejidad** (diap. 102); y el paraguas **GIGO** (diap. 18).
- **Animaciones por construcción:** varias secuencias son estados intermedios de una misma lámina
  (25–27, 36–46, 55–61, 66–69, 73–75, 78–84, 85–88, 90–93, 94–97, 98–102). Esta guía reconstruye
  el **estado final** de cada una e indica el rango correspondiente.
- **Ubicación de la sesión:** pese al título *Evaluación*, el mapa del curso la sitúa dentro del
  bloque **"Modelado y rendimiento"** (§1). Se señala para evitar confusión con el índice del
  curso.
- Elementos marcados **[verificar]**: el rótulo "Prueba"/"Training" de la fila de redes profundas
  (diap. 93–97).

---

*Guía de estudio elaborada a partir de `Sem5_Evaluación.pdf` (110 diapositivas).*
*Curso: **Planificación y Toma de Decisiones en IA**, Dra. Aurea Soriano-Vargas — UTEC, 2026.*
