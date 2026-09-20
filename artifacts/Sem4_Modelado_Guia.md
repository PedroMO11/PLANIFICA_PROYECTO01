# Semana 4 — Modelado y rendimiento · Guía de estudio

| Campo | Valor |
|---|---|
| **Curso** | Planificación y Toma de Decisiones en IA |
| **Sesión** | Semana 4 · *Modelado* · 2026-2 |
| **Docente** | Dra. Aurea Soriano-Vargas (`aureasoriano` · asoriano@utec.edu.pe · `aurea-soriano`) |
| **Institución** | UTEC |
| **Fuente** | `Sem4_Modelado (1).pdf` — 109 diapositivas |
| **Atribución del pie** | *Planificación y Toma de Decisiones en IA*, Dr. Aurea Soriano-Vargas (2026) |

> **Propósito de esta guía:** reconstruir fielmente el contenido teórico de las 109 diapositivas
> de la sesión —con especial cuidado en las **fórmulas**, que en el PDF están todas como
> imágenes— para poder estudiar la sesión sin depender del deck original.

---

## 1. Mapa del curso: ubicación de la sesión

Diagrama **"La historia de este curso"** (diapositivas 2, 6 y 105). El recuadro con borde
amarillo marca la sesión en foco; los grises son sesiones ya cubiertas.

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
                      la diap. 2        ↑ resaltado en las diap. 6 y 105 └────────┬─────────┘
                      (repaso Sem 3)      (ESTA SESIÓN)                           │
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

> *Nota:* a diferencia de las sesiones anteriores, el mapa de cierre (diap. 105) **sigue
> resaltando "Modelado y rendimiento"**; el deck no adelanta la sesión siguiente.

### 1.1 Repaso: el ciclo de vida (diap. 3, 9)

```
┌────────────┐   ┌───────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│ Iniciación │──▶│ Entrenamiento │──▶│ Estimación │──▶│ Validación │──▶│ Despliegue │
└────────────┘   └───────────────┘   └────────────┘   └────────────┘   └────────────┘
```

### 1.2 Repaso: planteamiento del problema (diap. 4)

```
┌───────────────────────────────────┐              ┌────────────────────────────────────┐
│     Problema de negocios          │ ─Traducción─▶│ Problema de aprendizaje automático │
│ Los objetivos y criterios de      │              │ Por otro lado, los algoritmos      │
│ éxito describen el problema.      │              │ requieren un objetivo claro y      │
│ Sin embargo, subjetivas y         │              │ medible.                           │
│ carecen de mensurabilidad.        │              │                                    │
└───────────────────────────────────┘              └────────────────────────────────────┘
                                          Aprendizaje Supervisado → Clasificación / Regresión
```

### 1.3 Repaso: conclusiones de Iniciación (diap. 5)

- La iniciación prepara el terreno: sin buenos datos ni problema definido, el modelo fracasa.
- Recordar el principio: **"Garbage in, garbage out"**.
- La tendencia actual: **IA centrada en los datos (Data-Centric AI)**.

| Planteamiento del Problema | Datos | Análisis | Preprocesamiento |
|---|---|---|---|
| Planteamiento bien definido + objetivos + criterios de éxito = éxito de una solución de IA. | Diferentes criterios, como su estructura, origen y tipo. | Análisis de la distribución y la calidad de los datos ➡ información preliminar para los siguientes pasos. | El preprocesamiento de datos y la ingeniería de atributos preparan un conjunto de datos. |

---

## 2. Objetivos y agenda de la sesión

### 2.1 Al finalizar la clase deberíamos tener los medios para responder (diap. 7)

1. Conceptos básicos de aprendizaje automático supervisado.
2. Métricas de rendimiento para regresión y clasificación.

### 2.2 Agenda (diap. 8)

1. **Modelos**
2. **Métricas de rendimiento**

Los modelos se agrupan en tres familias, que estructuran toda la primera mitad del deck
(diap. 11–12, 21, 29):

```
┌────────────┐  ┌───────────────┐  ┌───────────┐
│ Regresión  │  │ Clasificación │  │ Ensamble  │
└────────────┘  └───────────────┘  └───────────┘
```

---

## 3. Modelos · Regresión (sección, diap. 10–20)

### 3.1 Regresión lineal univariada (diap. 13)

```
┌──────────────────────┐
│ Conjunto de          │
│ entrenamiento        │
└──────────┬───────────┘
           ▼
    ┌─────────────┐
    │  Regresión  │
    └──────┬──────┘
           ▼
 Variable1 ──▶ [ h ] ──▶ Variable objetivo
              (hipótesis)

          h mapea x's a y's
```

**¿Cómo representar $h$?**

$$h_\theta(x) = \theta_0 + \theta_1 x$$

> *Regresión Linear con una variable. **Univariate** linear regression.*

La diapositiva acompaña la fórmula con un diagrama de dispersión (ejes $x$, $y$) y una recta
ajustada a los puntos.

### 3.2 Regresión lineal multivariada (diap. 14)

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$

> **Por conveniencia, definimos $x_0 = 1$.**

$$x = \begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^{\,n+1}
\qquad
\theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \theta_2 \\ \vdots \\ \theta_n \end{bmatrix} \in \mathbb{R}^{\,n+1}$$

Al multiplicar el vector fila de parámetros por el vector columna de atributos:

$$\begin{bmatrix} \theta_0 & \theta_1 & \cdots & \theta_n \end{bmatrix}
\begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ x_n \end{bmatrix}
\quad\Longrightarrow\quad
\boxed{\,h_\theta(x) = \theta^{T} x\,}$$

> ***Multivariate** linear regression.*

### 3.3 Hipótesis, parámetros, función de costo y gradient descent (diap. 15)

**Hipótesis:**

$$h_\theta(x) = \theta^{T} x = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$

**Parámetros:** $\theta_0, \theta_1, \ldots, \theta_n$

**Función de Costo:**

$$J(\theta_0, \theta_1, \ldots, \theta_n) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^{2}$$

**Gradient Descent:**

```
repeat {
```
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1, \ldots, \theta_n)$$
```
}
```
> *(actualizamos simultáneamente para cada $j = 0, 1, \ldots, n$)*

> *Errata del original:* la diapositiva escribe **"Funciń de Costo"**.

### 3.4 Ecuación Normal / Ordinary Least Squares (diap. 16–17)

Con $m$ **ejemplos** $(x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)})$ y $n$ **atributos**:

$$X = \begin{bmatrix} \text{---} & (x^{(1)})^{\mathrm{T}} & \text{---} \\
\text{---} & (x^{(2)})^{\mathrm{T}} & \text{---} \\
\text{---} & \vdots & \text{---} \\
\text{---} & (x^{(m)})^{\mathrm{T}} & \text{---} \end{bmatrix}
\qquad
y = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(m)} \end{bmatrix}$$

**Ecuación Normal:**

$$\theta = (X^{T} X)^{-1} X^{T} y$$

> La diapositiva 17 superpone el rótulo **Ordinary Least Squares (OLS)** sobre esta misma
> lámina: la Ecuación Normal *es* OLS.

### 3.5 Gradient Descent vs Normal Equation (diap. 18)

*Con $m$ ejemplos y $n$ atributos:*

| **Gradient Descent** | **Normal Equation (OLS)** |
|---|---|
| Necesitamos escoger $\alpha$. | No se necesita escoger $\alpha$. |
| Necesita muchas iteraciones. | No necesita iterar. |
| Funciona bien cuando $n$ es grande. | **No necesita escalar.** |
| | Se necesita calcular $(X^{T}X)^{-1}$ → $O(n^{3})$. |
| | Lento si $n$ es muy grande. |

> *Nota de lectura:* en la diapositiva el símbolo de la tasa de aprendizaje se representa con
> el carácter "∝"; por el contexto (elegir un paso en gradient descent) corresponde a
> $\alpha$ (alpha). [verificar: el glifo exacto usado en el original es "∝"]

### 3.6 Variantes de regresión (diap. 19)

- **Ridge regression:** $L_2$ regularized regression
- **Lasso regression:** $L_1$ regularized regression
- **Elastic Net regression:** $L_1$ and $L_2$ regularized regression
- **Polynomial regression**

### 3.7 Tabla comparativa de modelos de regresión (diap. 20)

| Modelo ↓ | Idea principal | Tipo de relación | Regularización | Fortaleza principal | Debilidad principal |
|---|---|---|---|---|---|
| **Linear Regression** | Predice (y) usando una variable (x) | Lineal | ❌ No | Simple e interpretable | No captura relaciones complejas |
| **Multiple Linear Regression** | Predice (y) usando varias variables (x_1,x_2,...) | Lineal | ❌ No | Combina múltiples características | Sensible a multicolinealidad |
| **Ridge Regression** | Regresión lineal penalizando coeficientes grandes | Lineal | ✅ L2 | Reduce overfitting y maneja variables correlacionadas | No elimina variables |
| **Lasso Regression** | Penaliza el valor absoluto de los coeficientes | Lineal | ✅ L1 | Puede llevar coeficientes a **0** → selección de variables | Puede ser inestable con variables muy correlacionadas |
| **Elastic Net** | Combina Ridge + Lasso | Lineal | ✅ L1 + L2 | Regulariza y selecciona variables | Requiere ajustar más hiperparámetros |
| **Polynomial Regression** | Añade potencias de las variables (x^2,x^3,...) | **No lineal en (x)** | ❌ No necesariamente | Captura relaciones curvas | Grados altos pueden causar overfitting |

---

## 4. Modelos · Clasificación (sección, diap. 21–28)

### 4.1 Clasificador Lineal (diap. 22)

$$h_\theta(x) = \theta_0 + \theta_1 x$$

Diagrama: plano $(x_1, x_2)$ con una recta que separa dos nubes de puntos.
Leyenda: **● Denota +1** · **○ Denota −1**.

### 4.2 Regresión Logística (diap. 23)

$$0 \le h_\theta(x) \le 1$$

$$h_\theta(x) = g(\theta^{\mathrm{T}} x)
\qquad\text{con}\qquad
g(z) = \frac{1}{1 + e^{-z}}$$

Combinando ambas:

$$h_\theta(x) = \frac{1}{1 + e^{-\theta^{\mathrm{T}} x}}$$

> **Sigmoid Function / Logistic Function.**

La gráfica de $g(z)$ es la curva sigmoide: asíntota en 0 para $z \to -\infty$, asíntota en 1
para $z \to +\infty$, y pasa por **0.5** en $z = 0$.

### 4.3 Support Vector Machine (SVM) (diap. 24)

**Formulación:**

$$\text{minimize}\quad \frac{1}{2}\lVert \mathbf{w} \rVert^{2}$$

**Tal que**

$$y_i\left(\mathbf{w}^{T}\mathbf{x}_i + b\right) \ge 1$$

Diagrama del margen:

```
        w^T x + b = 1     ← hiperplano de soporte (lado +)
        w^T x + b = 0     ← frontera de decisión
        w^T x + b = -1    ← hiperplano de soporte (lado −)
```

La banda entre los dos hiperplanos de soporte es el **Margen**; los puntos que caen sobre
ellos (marcados $x^{+}$ y $x^{-}$) son los vectores de soporte, y $n$ denota la normal al
hiperplano.

### 4.4 Truco del Kernel (diap. 25)

Proceso en tres pasos:

| Paso | Contenido |
|---|---|
| **1. Espacio original** (no linealmente separable) | Plano $(x_1, x_2)$ con clases +1 y −1 entremezcladas. |
| **2. Espacio transformado** (linealmente separable) | Mediante un **mapeo no lineal $\phi(x)$** se pasa a $(z_1, z_2, z_3)$, donde un hiperplano separa las clases. SVM encuentra un hiperplano óptimo. |
| **3. Truco de kernel** | En lugar de calcular $\phi(x)$ explícitamente, SVM usa una **función kernel** $K(x, x')$ que calcula el producto punto en el espacio transformado. |

$$K(x, x') = \langle \phi(x), \phi(x') \rangle$$

> Permite encontrar el hiperplano óptimo en el espacio transformado **usando solo los datos
> originales**.

**KERNELS MÁS USADOS:**

| # | Kernel | Descripción | Fórmula | Parámetros |
|---|---|---|---|---|
| 1 | **Lineal** | No realiza transformación. Útil cuando los datos ya son linealmente separables. | $K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^{\top}\mathbf{x}'$ | ninguno |
| 2 | **Polinomial** | Permite fronteras de decisión curvas mediante términos polinómicos. | $K(\mathbf{x}, \mathbf{x}') = (\gamma\,\mathbf{x}^{\top}\mathbf{x}' + r)^{d}$ | $\gamma > 0,\ r \ge 0,\ d \in \mathbb{N}$ (grado del polinomio) |
| 3 | **RBF (Gaussiano)** | El más usado en la práctica. Maneja fronteras complejas con buena generalización. | $K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \lVert \mathbf{x} - \mathbf{x}' \rVert^{2}\right)$ | $\gamma > 0$ (ancho del kernel) |
| 4 | **Sigmoidal** | Similar a una red neuronal (función de activación sigmoide). | $K(\mathbf{x}, \mathbf{x}') = \tanh\left(\gamma\,\mathbf{x}^{\top}\mathbf{x}' + r\right)$ | $\gamma > 0,\ r \in \mathbb{R}$ |
| 5 | **Chi-cuadrado** | Usado en datos de histogramas o de conteo (no negativos). | $K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \sum_i \dfrac{(x_i - x'_i)^{2}}{x_i + x'_i}\right)$ | $\gamma > 0$ |

### 4.5 K-Nearest Neighbors — *Lazy Learning Algorithm* (diap. 26)

Diagrama: un punto a clasificar (★) con dos vecindades circulares, **k = 3** y **k = 6**,
sobre puntos de *Class A* y *Class B* en el plano $(x_1, x_2)$.

**Tipos de votación:**

| Caso | Composición de los vecinos | Majority vote | Plurality vote |
|---|---|---|---|
| **A** | 4 de una clase + 6 de otra | Gana la clase mayoritaria | La misma clase (**> 50 %**) |
| **B** | 3 + 4 + 4 (tres clases) | **None** | La clase con más votos (**> 33.3 %**) |

> **Majority voting** significa: ganador **>= 50 %** de los votos.
> **Plurality voting** significa: el que obtiene más votos.

### 4.6 Árbol de Decisión (diap. 27)

**Ejemplo de árbol (evaluación crediticia):**

```
                    ¿Ingreso > S/3000?
                    ╱ Sí            ╲ No
        ¿Historial crediticio        ✗ Rechazado
             bueno?                    (Ingreso insuficiente)
        ╱ Sí          ╲ No
   ¿Deuda baja?        ✗ Rechazado
  (Deuda/Ingreso        (Historial crediticio
     < 30%)              negativo)
  ╱ Sí      ╲ No
✓ Aprobado   ✗ Rechazado
(Perfil       (Deuda alta)
 favorable)
```

**En clasificación: Gini o Entropía**

$$Gini = 1 - \sum_{i=1}^{C} p_i^{2}$$

El árbol busca una división que produzca la **mayor reducción de impureza**:

$$\Delta Gini = Gini_{padre} - Gini_{hijos}$$

$$Entropy = -\sum_i p_i \log_2(p_i)$$

Se selecciona la división que maximiza la *Information Gain* (buscar una división que haga que
los nodos hijos sean más homogéneos que el nodo padre):

$$IG = Entropy_{padre} - Entropy_{hijos}$$

### 4.7 Tabla comparativa de clasificadores (diap. 28)

| Modelo | Idea principal | Frontera | Probabilidades | Escalamiento | Fortaleza | Debilidad |
|---|---|---|---|---|---|---|
| **Clasificador lineal** | Encuentra una combinación lineal que separa clases | Lineal | No necesariamente | Recomendable | Simple y rápido | No captura relaciones complejas |
| **Regresión logística** | Estima la probabilidad de pertenecer a una clase | Lineal | ✅ Sí | Recomendable | Probabilística e interpretable | Limitada ante fronteras complejas |
| **SVM** | Maximiza el margen entre clases | Lineal o no lineal con **kernel** | ❌ No directamente | ✅ Muy importante | Excelente con fronteras complejas | Sensible a (C), kernel y (\gamma) |
| **KNN** | Decide según los vecinos más cercanos | No lineal | Aproximada | ✅ Muy importante | Simple y flexible | Predicción lenta y sensible a escala |
| **Decision Tree** | Divide los datos mediante reglas sucesivas | No lineal | Aproximada | ❌ No necesario | Muy fácil de explicar | Propenso a overfitting |

> *Nota de fidelidad:* la celda de debilidad de SVM aparece en el original con la sintaxis
> LaTeX sin renderizar **"(\gamma)"**; se transcribe tal cual.

---

## 5. Modelos · Ensamble (sección, diap. 29–32)

### 5.1 Las tres técnicas (diap. 30)

| | **Bagging** | **Boosting** | **Stacking** |
|---|---|---|---|
| **Cómo entrena** | En paralelo, con diferentes muestras de datos | Secuencialmente; cada modelo corrige al anterior | Diferentes modelos en paralelo |
| **Cómo combina** | Combina con Voto/promedio | Suma ponderada | Un meta-modelo aprende a combinarlos |
| **Ejemplo** | Random Forest | AdaBoost, XGBoost, LightGBM | — |

### 5.2 Tabla comparativa de ensambles (diap. 31)

| Técnica | ¿Cómo entrena los modelos? | ¿Cómo combina? | Objetivo principal | Ejemplo típico |
|---|---|---|---|---|
| **Bagging** | En paralelo, con diferentes muestras de datos | Voto / promedio | Reducir **varianza** | Random Forest |
| **Boosting** | Secuencialmente; cada modelo corrige al anterior | Suma ponderada | Reducir **bias/error** | AdaBoost, XGBoost, LightGBM |
| **Stacking** | Diferentes modelos en paralelo | Un **meta-modelo aprende** a combinarlos | Aprovechar fortalezas distintas | RF + SVM + XGBoost → Logistic Regression |

### 5.3 ¿Cómo elegir el algoritmo adecuado? (diap. 32)

- Reglas básicas, p. ej., número de observaciones
- Lógica
- Explicabilidad y transparencia
- Experiencia
- Intuición

---

## 6. Métricas de rendimiento · Clasificación (sección, diap. 33–35)

### 6.1 Medidas de evaluación para la clasificación (diap. 35)

> En problemas de clasificación, utilizamos las siguientes métricas de evaluación:

- **Matriz de confusión;**
  - Exactitud y exactitud balanceada;
  - Tasa de error;
  - Precisión;
  - Sensibilidad/recall;
  - Especificidad;
  - Puntuación F;
  - Curva ROC y AUC;

### 6.2 La matriz de confusión: ejemplo enfermos vs. sanos (diap. 36–39)

> Considerando la **presencia de una enfermedad** como la **clase positiva** y la **ausencia de
> la enfermedad** como la **clase negativa**, tenemos:

- **FP: error de identificación.** → Persona sana clasificada como enferma;
- **FN: error de rechazo.** → Persona enferma clasificada como sana;

| | **Diagnosticado Enfermo** | **Diagnosticado Sano** |
|---|---|---|
| **Enfermo** | Verdadero Positivo (TP) | **Falso Negativo (FN)** |
| **Sano** | **Falso Positivo (FP)** | Verdadero Negativo (TN) |

### 6.3 Matriz de confusión binaria con cifras (diap. 40–44)

> Convención del deck: las **filas** son los *Datos* (clase real) y las **columnas** la
> *Predicción*.

| **Datos** \ **Predicción** | **Predicho Positivo** | **Predicho Negativo** |
|---|---|---|
| **Positivo** | **6** verdaderos positivos | **1** falso negativo |
| **Negativo** | **2** falsos positivos | **5** verdaderos negativos |

> Este mismo conjunto de 14 puntos ($6+1+2+5$) se reutiliza en los ejemplos de accuracy
> (§7.1), precisión (§7.4) y recall (§7.5). La leyenda del diagrama es **● Positivo** /
> **● Negativo**.

### 6.4 Matriz de confusión de $n$ clases (diap. 45–46)

Para tres clases, la matriz se generaliza a una cuadrícula $3 \times 3$ con las **Clases
Verdaderas** en filas y las **Clases Predichas** en columnas:

| **Clase Verdadera** \ **Clase Predicha** | **Clase 1 Predicha** | **Clase 2 Predicha** | **Clase 3 Predicha** |
|---|---|---|---|
| **Clase 1** | | | |
| **Clase 2** | | | |
| **Clase 3** | | | |

Leyenda del diagrama de dispersión que la acompaña: **Class 1: ▲** · **Class 2: ■** ·
**Class 3: ●**. *(La diapositiva presenta la matriz vacía, como plantilla.)*

---

## 7. Métricas derivadas de la Matriz de Confusión

### 7.1 Exactitud (Accuracy) y Tasa de Error (diap. 47)

**Exactitud (Accuracy):**

$$Acc = \frac{TP + TN}{TP + FP + TN + FN} = \frac{TP + TN}{n}$$

**Tasa de Error:**

$$Err = \frac{FP + FN}{n} = 1 - Acc$$

**Ejemplo con los 14 puntos (diap. 55–57):**

> *Accuracy: De todos los datos, ¿cuántos puntos clasificamos correctamente?*

$$Accuracy = \frac{\text{Correctamente clasificados}}{\text{Todos los puntos}} = \frac{11}{14} = 78.57\,\%$$

> ✔ Verificación: $TP + TN = 6 + 5 = 11$ sobre $n = 14$; $11/14 = 0.785714\ldots = 78.57\,\%$.
> Consistente con la matriz de §6.3.

### 7.2 El problema con la exactitud (diap. 48–49)

> Considere la siguiente matriz de confusión:

| | **Predicho Positivo** | **Predicho Negativo** |
|---|---|---|
| **Positivo** | 80 | 10 |
| **Negativo** | 9 | 1 |

$$Acc = \frac{TP + TN}{n} = \frac{80 + 1}{100} = 0.81$$

> ✔ Verificación: $n = 80+10+9+1 = 100$; $(80+1)/100 = 0.81$. Correcto.
> El punto del ejemplo: una accuracy de 0.81 **oculta** que de los 10 negativos reales el
> modelo solo acierta 1.

### 7.3 Exactitud Balanceada (diap. 50–53)

Sobre la **misma** matriz del apartado anterior:

$$TPR = \frac{TP}{TP + FN} = \frac{TP}{n_{+}} = \frac{80}{90} = 0.89$$

$$TNR = \frac{TN}{TN + FP} = \frac{TN}{n_{-}} = \frac{1}{10} = 0.1$$

$$Acc_{B} = \frac{0.89 + 0.1}{2} = 0.495$$

> ✔ Verificación: $80/90 = 0.8889 \approx 0.89$; $1/10 = 0.1$; $(0.89+0.1)/2 = 0.495$.
> Toda la aritmética cuadra. La accuracy balanceada (0.495) revela el fallo que la accuracy
> simple (0.81) escondía.

### 7.4 Balanced Accuracy y Weighted Accuracy (diap. 54)

> 5 clases y las clases 4 y 5 son enfermedades raras. Podemos reformular la accuracy balanceada
> como:

$$WeightAcc = \frac{W_1 \ast TPR_{1,1} + W_2 \ast TPR_{2,2} + W_3 \ast TPR_{3,3} + W_4 \ast TPR_{4,4} + W_5 \ast TPR_{5,5}}{W_1 + W_2 + W_3 + W_4 + W_5}$$

> Con $W_1 = 0.1$, $W_2 = 0.1$, $W_3 = 0.1$, $W_4 = 0.35$, $W_5 = 0.35$

> Entonces, **WeightAcc** es llamada **Weighted Accuracy** en el que se asignan pesos
> predefinidos a cada clase. Cabe destacar que la Balanced Accuracy Balanceada es un caso
> particular de Weighted Accuracy **cuando todos los pesos son iguales**.

> ✔ Verificación: los pesos suman $0.1+0.1+0.1+0.35+0.35 = 1.0$; las clases raras (4 y 5)
> reciben un peso 3.5× mayor que las comunes.
> *Errata del original:* "la Balanced Accuracy **Balanceada**" (redundancia en la diapositiva).

### 7.5 Precisión (diap. 58, 61–63)

> *Precisión: De todos los puntos que hemos predicho como positivos, ¿cuántos son correctos?*

$$Precision = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} = \frac{6}{8}$$

> ✔ Verificación: con $TP = 6$ y $FP = 2$ de la matriz de §6.3, $6/(6+2) = 6/8 = 75\,\%$
> — el valor que la diapositiva 77 reporta como *Precision: 75%*.

### 7.6 Recall (diap. 59, 64–67)

> *Recall: De todos los puntos etiquetados como positivos, ¿cuántos predijimos correctamente?*

$$Recall = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} = \frac{6}{6 + 1} = 85.7\,\%$$

> ✔ Verificación: $6/7 = 0.857142\ldots = 85.7\,\%$. Correcto.

### 7.7 ¿Cuándo importa cada una? (diap. 60)

| **Modelo Médico** | **Detector de Spam** |
|---|---|
| Falsos positivos **ok** | Falsos positivos **NO** ok |
| Falsos negativos **NO** ok | Falsos negativos **ok** |
| → **Recall Alta** | → **Precision Alta** |

**Cifras del ejemplo (diap. 68):**

| | Modelo Médico | Detector de Spam |
|---|---|---|
| **Precisión** | 55.7 % | **76.9 %** |
| **Recall** | **83.3 %** | 37 % |

---

## 8. Combinar precisión y recall: la puntuación F

### 8.1 ¿Una métrica? El promedio simple falla (diap. 69–70)

| | Modelo Médico | Detector de Spam |
|---|---|---|
| Precisión | 55.7 % | 76.9 % |
| Recall | 83.3 % | 37 % |
| **Average** | **69.5 %** | **56.9 %** |

> ✔ Verificación: $(55.7+83.3)/2 = 69.5$; $(76.9+37)/2 = 56.95 \approx 56.9$. Correcto.

### 8.2 El caso extremo: fraude en tarjeta de crédito (diap. 71)

> **Modelo: Todas las transacciones son fraudulentas.**

| Transacciones legítimas | Transacciones fraudulentas |
|---|---|
| 284,335 | 472 |

$$Precision = \frac{472}{284{,}807} = 0.016\,\%
\qquad
Recall = \frac{472}{472} = 100\,\%$$

> ⚠ **Discrepancia aritmética detectada (no corregida).** El total $284{,}335 + 472 = 284{,}807$
> es correcto, y el recall de 100 % también. Sin embargo
> $472 / 284{,}807 = 0.0016573$, lo que equivale a **0.1657 %**, no a **0.016 %** como indica
> la diapositiva. El valor mostrado difiere en aproximadamente un factor de 10 (parece
> confundirse el valor decimal $\approx 0.0017$ con su expresión porcentual $\approx 0.17\,\%$).
> Se deja el dato **tal como aparece en el original** y se señala aquí. **[verificar con la
> docente]**
> *El argumento pedagógico de la diapositiva no se ve afectado:* la precisión es
> catastróficamente baja aunque el recall sea perfecto.

### 8.3 Media armónica (diap. 72–76)

$$\text{Media aritmética} = \frac{x + y}{2}
\qquad\qquad
\text{Media armónica} = \frac{2xy}{x + y}$$

> **F1 Score = Media Armónica (Precision, Recall)**

**Ejemplos del deck:**

| Caso | Precision | Recall | Average | Harmonic Mean |
|---|---|---|---|---|
| 1 | 1 | 0 | 0.5 | **0** |
| 2 | 0.2 | 0.8 | 0.5 | **0.32** |

> ✔ Verificación: caso 1, $2(1)(0)/(1+0) = 0$; caso 2, $2(0.2)(0.8)/(0.2+0.8) = 0.32/1 = 0.32$.
> Ambos correctos. La lección: la media armónica **castiga** los desequilibrios que la media
> aritmética disimula (ambos casos promedian 0.5).

### 8.4 F1 Score sobre el ejemplo de 14 puntos (diap. 77)

| Métrica | Valor |
|---|---|
| Precision | 75 % |
| Recall | 85.7 % |
| Average | 80.3 % |
| **F1 Score** | **80 %** |

> ✔ Verificación: promedio aritmético $(75+85.7)/2 = 80.35 \approx 80.3\,\%$;
> media armónica $2(0.75)(0.857)/(0.75+0.857) = 0.7999 \approx 80\,\%$. Ambos cuadran.

### 8.5 $F_\beta$ Score (diap. 78–82)

> **Media armónica entre precision y recall:**

$$F_\beta = \frac{(\beta^{2} + 1) \times precision \times recall}{\beta^{2} \times precision + recall}$$

> $\beta$ es el peso entre precision y recall:
> - Si $\beta < 1$, **precision** es más importante;
> - Si $\beta > 1$, **recall** es más importante.

### 8.6 Comportamiento de $F_\beta$ según $\beta$ (diap. 83)

Gráfico de líneas — eje X: $\beta$ (valores 0,1 · 0,5 · 1,0 · 5 · 10); eje Y: F-score (0 a 0,9).

| Serie | Comportamiento al crecer $\beta$ |
|---|---|
| **P = 0.9, R = 0.1** | Decrece: parte de ≈0,83 en $\beta=0{,}1$ y cae a ≈0,10 en $\beta=10$ |
| **P = R = 0.5** | Constante en **0,5** para todo $\beta$ |
| **P = 0.1, R = 0.9** | Crece: parte de ≈0,10 en $\beta=0{,}1$ y sube a ≈0,83 en $\beta=10$ |

> Las curvas azul y amarilla se cruzan en $\beta = 1$ (ambas ≈0,18), que es el caso simétrico
> del F1. *(Valores leídos del gráfico; son aproximados por tratarse de una lectura visual.)*

---

## 9. Errores de Tipo I y II, curva ROC y AUC

### 9.1 Error Tipo I y Error Tipo II (diap. 84)

| **Error Tipo I** (falso positivo) | **Error Tipo II** (falso negativo) |
|---|---|
| Se afirma algo que no ocurre. | No se detecta algo que sí ocurre. |
| Ilustración del deck: a un hombre se le dice *"Estás embarazado"*. | Ilustración del deck: a una mujer visiblemente embarazada se le dice *"No estás embarazado"*. |

### 9.2 La curva ROC (diap. 85)

> La curva ROC (característica operativa del receptor) muestra el **tradeoff** entre la **tasa
> de verdaderos positivos (TPR)** y la **tasa de falsos positivos (FPR)** de un clasificador.

$$FPR = \frac{FP}{TN + FP}$$

> Es una relación entre el **costo (FPR)** y el **beneficio (TPR)** y se construye variando el
> **umbral/límite de decisión** entre clases.
>
> **Desventaja: Solo se define para problemas binarios.**

### 9.3 El espacio ROC (diap. 86–90)

Ejes: **False Positive rate** (X, de 0 a 1) y **True Positive rate** (Y, de 0 a 1).

```
TPR
 1 ┤ Ideal ←(0,1)                              ╱
   │                                         ╱
0.8┤              ● Classifier 1 (0.5, 0.72)╱
   │   ● Classifier 3 (0.2, 0.6)          ╱
0.6┤   ◯ ← resaltado como el mejor       ╱
   │        ● Classifier 2 (0.3, 0.4)  ╱
0.4┤                                 ╱  ← Aleatorio (diagonal)
   │                               ╱
0.2┤                             ╱
   │                           ╱
 0 └──────────────────────────╱───────────────▶ FPR
   0    0.2   0.4   0.6   0.8   1  → Caso Peor (1,0)
```

| Región del espacio | Significado |
|---|---|
| **Ideal** | Esquina superior izquierda: $(FPR, TPR) = (0, 1)$ |
| **Aleatorio** | La diagonal principal |
| **Caso Peor** | Esquina inferior derecha: $(1, 0)$ |

En la diapositiva 90 se resalta con un círculo el **Classifier 3**, por ser el más cercano al
punto ideal.

*Crédito de la figura en el original:* Robert Holte, 349 Athabasca Hall, Department of
Computing Science, University of Alberta.

### 9.4 Área bajo la curva (AUC) (diap. 91)

- Al analizar múltiples curvas ROC, es común encontrar **intersecciones** que dificultan el
  análisis.
- El AUC produce valores **entre 0 y 1**, donde los valores cercanos a 1 indican que la curva
  está más cerca del punto ideal (**TPR = 1 y FPR = 0**).

*Figura de ejemplo (scikit-learn): "Receiver operating characteristic example", con una curva
escalonada etiquetada **ROC curve (area = 0.79)** y la diagonal punteada de referencia.*

---

## 10. Métricas de rendimiento · Regresión (diap. 92–97)

> **Usamos métricas específicas para problemas de Regresión:**
> - Mean Squared Error (MSE);
> - Mean Absolute Error (MAE);
> - Symmetric Mean Absolute Percentage Error (sMAPE);
> - Coefficient of Determination (R²);

### 10.1 Mean Squared Error (MSE) (diap. 94)

$$MSE = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^{2}$$

> ¿Qué ocurre si los errores son graves? **Debido al cuadrado, se tiende a penalizar más los
> errores graves.**

### 10.2 Mean Absolute Error (MAE) (diap. 95)

> Similar a MSE, pero utiliza el **valor absoluto** del error.

$$MAE = \frac{1}{m} \sum_{i=1}^{m} \left| h_\theta(x^{(i)}) - y^{(i)} \right|$$

### 10.3 Symmetric Mean Absolute Percentage Error (sMAPE) (diap. 96)

> Mide el error porcentual de forma **simétrica** entre el valor real y el predicho.

$$sMAPE = \frac{100\,\%}{m} \sum_{i=1}^{m} \frac{2\left| h_\theta(x^{(i)}) - y^{(i)} \right|}{\left| y^{(i)} \right| + \left| h_\theta(x^{(i)}) \right|}$$

### 10.4 Coefficient of Determination (R²) (diap. 97)

$$R^{2} = 1 - \frac{\sum_{i=1}^{m} \left( y^{(i)} - h_\theta(x^{(i)}) \right)^{2}}{\sum_{i=1}^{m} \left( y^{(i)} - \bar{y} \right)^{2}}$$

Expresado conceptualmente (recuadro de la diapositiva):

$$R^{2} = 1 - \frac{\text{Error del modelo}}{\text{Variabilidad total de los datos}}$$

- Mide **qué proporción de la variabilidad de la variable objetivo es explicada por el modelo**.
- $R^{2}$ puede variar entre 0 y 1: **1 perfecto, 0 no mejora respecto a la media, <0 pero que
  la media**.

> *Errata del original:* "**<0 pero que la media**" — por el contexto se lee como "*peor que la
> media*". Se conserva el texto tal cual. **[verificar]**

---

## 11. Cómo elegir la métrica de rendimiento adecuada (diap. 98–101)

### 11.1 Criterios desde la Iniciación (diap. 99)

| Etapa | Preguntas guía |
|---|---|
| **Iniciación** | ¿Cuál es el problema **subyacente** y qué métrica aborda con precisión este problema? |
| | **Tomar en cuenta la distribución de datos** al elegir métricas de rendimiento. |

### 11.2 Tabla resumen — Métricas de Clasificación (diap. 100)

| Métrica | Qué mide | Rango | Mejor valor | Útil cuando… | Limitación principal |
|---|---|---|---|---|---|
| **Matriz de confusión** | Resume aciertos y errores por clase | — | — | Quieres ver el detalle de los errores | No da un único valor resumen |
| **Exactitud (Accuracy)** | Proporción total de predicciones correctas | ([0,1]) | 1 | Las clases están balanceadas | Puede ser engañosa en datos desbalanceados |
| **Exactitud balanceada** | Promedio entre sensibilidad y especificidad | ([0,1]) | 1 | Hay desbalance entre clases | Menos intuitiva que accuracy |
| **Tasa de error** | Proporción de predicciones incorrectas | ([0,1]) | **0** | Quieres medir el error global | Mismo problema que accuracy con desbalance |
| **Precisión (Precision)** | De los positivos predichos, cuántos eran realmente positivos | ([0,1]) | 1 | Importa evitar falsos positivos | No considera falsos negativos |
| **Sensibilidad / Recall / TPR** | De los positivos reales, cuántos detectó el modelo | ([0,1]) | 1 | Importa no perder positivos reales | No penaliza falsos positivos |
| **Especificidad / TNR** | De los negativos reales, cuántos detectó correctamente | ([0,1]) | 1 | Importa identificar bien los negativos | No refleja qué pasa con los positivos |
| **Puntuación F (F1-score)** | Balance entre precisión y recall | ([0,1]) | 1 | Quieres equilibrio entre FP y FN | **Ignora TN** |
| **Curva ROC** | Muestra el trade-off entre sensibilidad y tasa de falsos positivos | — | Curva más cerca de la esquina sup. izq. | Quieres comparar clasificadores para distintos umbrales | Puede verse optimista en clases muy desbalanceadas |
| **AUC** | Área bajo la curva ROC | ([0,1]) | 1 | Quieres una métrica resumen independiente del umbral | No siempre refleja bien el desempeño operativo final |

### 11.3 Tabla resumen — Métricas de Regresión (diap. 101)

| Métrica | Qué mide | Rango | Mejor valor | Útil cuando… | Limitación principal |
|---|---|---|---|---|---|
| **MSE** | Promedio de los errores al cuadrado | 0 a infinito | **0** | Quieres penalizar fuertemente los errores grandes | Muy sensible a outliers y queda expresado en unidades al cuadrado |
| **MAE** | Promedio de la magnitud absoluta de los errores | 0 a infinito | **0** | Quieres una medida sencilla, interpretable y menos sensible a outliers | No penaliza especialmente los errores muy grandes |
| **sMAPE** | Error porcentual relativo entre valores reales y predichos | 0 % a 200 % | **0 %** | Quieres comparar errores entre variables o series con diferentes escalas | Puede ser inestable cuando el valor real y el predicho están cerca de 0 |
| **R²** | Proporción de la variabilidad de los datos explicada por el modelo | Menos infinito a 1 | **1** | Quieres evaluar qué tan bien el modelo explica la variabilidad de la variable objetivo | No indica directamente cuánto se equivoca el modelo y puede ser negativo |

---

## 12. Conclusiones (diap. 102–103)

| Idea clave | Desarrollo |
|---|---|
| **El modelado comienza con un problema bien definido.** | Un problema de negocio debe traducirse en un objetivo de aprendizaje automático claro y medible. |
| **No existe un único modelo adecuado para todos los problemas.** | Los ensambles mejoran el desempeño combinando múltiples modelos de diferentes formas. |
| **Evaluar un modelo requiere elegir métricas coherentes con el problema y los datos.** | La evaluación debe realizarse sobre **datos no vistos** y **no depender de una única métrica**. |

*(Diap. 105: mapa del curso. Diap. 106: "¡Gracias!". Diap. 109: portada de cierre.)*

---

## 13. Glosario de términos del curso

> **Nota de organización:** este glosario **reordena y recoge las definiciones tal como las da
> el deck**; no añade teoría nueva. La diapositiva de origen se indica entre paréntesis.

| Término | Definición según el curso |
|---|---|
| **Accuracy (Exactitud)** | Proporción total de predicciones correctas: $Acc = (TP+TN)/n$. (47, 100) |
| **AUC** | Área bajo la curva ROC; produce valores entre 0 y 1, donde los cercanos a 1 indican que la curva está más cerca del punto ideal (TPR=1, FPR=0). (91) |
| **Bagging** | Entrena en paralelo con diferentes muestras de datos y combina con voto/promedio; reduce **varianza**. Ej.: Random Forest. (30–31) |
| **Boosting** | Entrena secuencialmente; cada modelo corrige al anterior y se combinan por suma ponderada; reduce **bias/error**. Ej.: AdaBoost, XGBoost, LightGBM. (30–31) |
| **Clasificador lineal** | Encuentra una combinación lineal que separa clases; frontera lineal. (22, 28) |
| **Curva ROC** | Muestra el tradeoff entre TPR y FPR variando el umbral/límite de decisión. Solo se define para problemas binarios. (85) |
| **Ecuación Normal / OLS** | Solución cerrada $\theta = (X^TX)^{-1}X^Ty$; no itera ni requiere escalar, pero cuesta $O(n^3)$. (16–18) |
| **Elastic Net** | Combina Ridge + Lasso: regularización L1 + L2. (19–20) |
| **Ensamble** | Familia de modelos que combina varios modelos: bagging, boosting, stacking. (29–31) |
| **Entropía** | $Entropy = -\sum_i p_i \log_2(p_i)$; se usa para elegir la división que maximiza la Information Gain. (27) |
| **Error Tipo I** | Falso positivo. (84) |
| **Error Tipo II** | Falso negativo. (84) |
| **Especificidad / TNR** | De los negativos reales, cuántos detectó correctamente: $TNR = TN/(TN+FP)$. (53, 100) |
| **$F_\beta$ Score** | $F_\beta = \frac{(\beta^2+1)\cdot P\cdot R}{\beta^2 P + R}$; con $\beta<1$ pesa más precision, con $\beta>1$ pesa más recall. (82) |
| **F1 Score** | Media armónica entre precision y recall. (76–77) |
| **Falso Negativo (FN)** | Error de rechazo: persona enferma clasificada como sana. (38) |
| **Falso Positivo (FP)** | Error de identificación: persona sana clasificada como enferma. (38) |
| **FPR** | Tasa de falsos positivos: $FPR = FP/(TN+FP)$. (85) |
| **Función de costo** | $J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2$. (15) |
| **Gini** | $Gini = 1 - \sum_{i=1}^{C} p_i^2$; el árbol busca la división con mayor reducción de impureza. (27) |
| **Gradient Descent** | $\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j}J(\theta)$, actualizando simultáneamente cada $j$. Requiere elegir $\alpha$ e iterar; funciona bien con $n$ grande. (15, 18) |
| **Hipótesis ($h$)** | Función que mapea x's a y's; $h_\theta(x) = \theta^T x$. (13–15) |
| **Information Gain (IG)** | $IG = Entropy_{padre} - Entropy_{hijos}$. (27) |
| **Kernel (truco del)** | Usar $K(x,x') = \langle\phi(x),\phi(x')\rangle$ para hallar el hiperplano óptimo en el espacio transformado usando solo los datos originales. (25) |
| **KNN** | Decide según los vecinos más cercanos; frontera no lineal, escalamiento muy importante. (26, 28) |
| **Lasso regression** | Regresión con regularización $L_1$; puede llevar coeficientes a 0 → selección de variables. (19–20) |
| **MAE** | $MAE = \frac{1}{m}\sum|h_\theta(x^{(i)})-y^{(i)}|$; similar a MSE pero con valor absoluto. (95) |
| **Majority voting** | El ganador necesita **>= 50 %** de los votos. (26) |
| **Margen** | En SVM, la banda entre los hiperplanos $w^Tx+b=1$ y $w^Tx+b=-1$. (24) |
| **Matriz de confusión** | Resume aciertos y errores por clase; filas = datos reales, columnas = predicción. (40–46, 100) |
| **MSE** | $MSE = \frac{1}{m}\sum(h_\theta(x^{(i)})-y^{(i)})^2$; por el cuadrado, penaliza más los errores graves. (94) |
| **Plurality voting** | Gana el que obtiene más votos (aunque no llegue al 50 %). (26) |
| **Polynomial regression** | Añade potencias de las variables; no lineal en $x$; grados altos pueden causar overfitting. (19–20) |
| **Precisión (Precision)** | De todos los puntos predichos como positivos, cuántos son correctos: $TP/(TP+FP)$. (63) |
| **R² (Coeficiente de determinación)** | Proporción de la variabilidad de la variable objetivo explicada por el modelo; 1 perfecto, 0 no mejora respecto a la media. (97) |
| **Recall / Sensibilidad / TPR** | De todos los puntos etiquetados como positivos, cuántos predijimos correctamente: $TP/(TP+FN)$. (67) |
| **Regresión logística** | Estima la probabilidad de pertenecer a una clase mediante la sigmoide $g(z)=1/(1+e^{-z})$. (23, 28) |
| **Ridge regression** | Regresión con regularización $L_2$; reduce overfitting y maneja variables correlacionadas, pero no elimina variables. (19–20) |
| **sMAPE** | Error porcentual simétrico entre valor real y predicho; rango 0 % a 200 %. (96, 101) |
| **Stacking** | Diferentes modelos en paralelo; un **meta-modelo aprende** a combinarlos. Ej.: RF + SVM + XGBoost → Logistic Regression. (30–31) |
| **SVM** | Maximiza el margen entre clases; minimiza $\frac{1}{2}\lVert w\rVert^2$ sujeto a $y_i(w^Tx_i+b)\ge 1$. (24, 28) |
| **Tasa de error** | $Err = (FP+FN)/n = 1 - Acc$. (47) |
| **Verdadero Negativo (TN)** | Caso negativo correctamente clasificado como negativo. (38) |
| **Verdadero Positivo (TP)** | Caso positivo correctamente clasificado como positivo. (38) |
| **Weighted Accuracy** | Accuracy balanceada con pesos predefinidos por clase; la Balanced Accuracy es el caso particular con todos los pesos iguales. (54) |

---

## 14. Checklists de síntesis

> **Nota de organización:** esta sección **reorganiza criterios ya enunciados en el deck** para
> facilitar su uso; **no añade teoría nueva**. Cada ítem remite a la diapositiva de origen.

### 14.1 Guía de elección de modelo de regresión (diap. 20)

| Si necesitas… | Modelo del curso |
|---|---|
| Máxima simplicidad e interpretabilidad | **Linear / Multiple Linear Regression** |
| Controlar overfitting con variables correlacionadas | **Ridge** (L2) |
| Seleccionar variables (llevar coeficientes a 0) | **Lasso** (L1) |
| Regularizar **y** seleccionar a la vez | **Elastic Net** (L1+L2) |
| Capturar relaciones curvas | **Polynomial Regression** |

### 14.2 Guía de elección de clasificador (diap. 28)

| Si necesitas… | Modelo del curso |
|---|---|
| Probabilidades calibradas e interpretables | **Regresión logística** |
| Fronteras complejas | **SVM** (con kernel) |
| Un modelo fácil de explicar a no técnicos | **Decision Tree** |
| Flexibilidad sin entrenar | **KNN** (predicción lenta) |
| Rapidez y simplicidad | **Clasificador lineal** |

> Recuerda que **SVM y KNN requieren escalamiento** ("Muy importante"), mientras que
> **Decision Tree no lo necesita**.

### 14.3 Guía de elección de métrica de clasificación (diap. 60, 100)

- [ ] ¿Las clases están **balanceadas**? Si no, la *accuracy* puede ser engañosa → usa
      **exactitud balanceada** o **weighted accuracy**. (49, 54)
- [ ] ¿Importa más **evitar falsos positivos** (tipo detector de spam)? → prioriza
      **Precision**. (60)
- [ ] ¿Importa más **no perder positivos reales** (tipo modelo médico)? → prioriza
      **Recall**. (60)
- [ ] ¿Quieres un **equilibrio** entre FP y FN? → **F1** (recuerda: **ignora TN**). (77, 100)
- [ ] ¿Quieres ponderar explícitamente uno u otro? → **$F_\beta$** con $\beta<1$ o $\beta>1$. (82)
- [ ] ¿Quieres comparar clasificadores **a través de umbrales**? → **ROC** y **AUC**
      (solo binario). (85, 91)

### 14.4 Guía de elección de métrica de regresión (diap. 101)

| Si… | Métrica del curso |
|---|---|
| Quieres penalizar fuertemente errores grandes | **MSE** (sensible a outliers) |
| Quieres una medida sencilla y robusta a outliers | **MAE** |
| Comparas series con **escalas distintas** | **sMAPE** (inestable cerca de 0) |
| Quieres saber cuánta variabilidad explica el modelo | **R²** (puede ser negativo) |

### 14.5 Los tres recordatorios de cierre (diap. 103)

- [ ] ¿El problema de negocio está traducido a un objetivo de ML claro y medible?
- [ ] ¿Se consideró más de un modelo (incluidos ensambles)?
- [ ] ¿La evaluación se hace sobre **datos no vistos** y con **más de una métrica**?

---

## 15. Notas de fidelidad sobre la reconstrucción

- **Todas las fórmulas de esta sesión están en el PDF como imágenes.** La capa de texto las
  devuelve vacías o destrozadas: por ejemplo, la diapositiva 15 se extrae como
  `"=      =  +  +  + ..."` y la 14 como `"=       +  +  +     +"`. Ninguna fórmula de esta
  guía procede de la capa de texto; todas se transcribieron leyendo las diapositivas
  rasterizadas.
- ⚠ **Discrepancia aritmética señalada, no corregida:** la precisión del ejemplo de fraude
  (diapositiva 71) figura como **0.016 %**, pero $472/284{,}807 = 0.0016573 \approx 0.1657\,\%$.
  Ver §8.2. Se conserva el valor del original y se marca **[verificar con la docente]**.
- **Ejemplos numéricos verificados y correctos:** accuracy $11/14 = 78.57\,\%$ (§7.1);
  $Acc = (80+1)/100 = 0.81$ (§7.2); $TPR = 80/90 = 0.89$, $TNR = 1/10 = 0.1$,
  $Acc_B = 0.495$ (§7.3); pesos de weighted accuracy suman 1.0 (§7.4); precision $6/8 = 75\,\%$
  (§7.5); recall $6/7 = 85.7\,\%$ (§7.6); promedios 69.5 % y 56.9 % (§8.1); medias armónicas
  0 y 0.32 (§8.3); F1 = 80 % y promedio 80.3 % (§8.4). Todos cuadran con las diapositivas.
- **Coherencia interna comprobada:** la matriz de confusión de la diapositiva 44
  (TP=6, FN=1, FP=2, TN=5) es consistente con los valores de accuracy (11/14), precisión (6/8)
  y recall (6/7) que el deck usa en las diapositivas 57, 63 y 67 sobre el mismo conjunto de
  14 puntos.
- **Contenido recuperado solo por lectura visual:** los cinco **kernels de SVM** con sus
  fórmulas y parámetros (diap. 25); las fórmulas de **Gini, ΔGini, Entropy e IG** (diap. 27);
  el **árbol de decisión** de evaluación crediticia (diap. 27); las cuatro **fórmulas de
  regresión** MSE/MAE/sMAPE/R² (diap. 94–97); el **espacio ROC** con las coordenadas de los
  tres clasificadores (diap. 86–90); las **tablas resumen** de métricas (diap. 100–101); y las
  tablas comparativas de modelos (diap. 20, 28, 31).
- **Animaciones por construcción:** numerosas diapositivas son estados intermedios de una misma
  lámina (13–17, 35–46, 48–57, 58–68, 69–83, 86–90, 92–97). Esta guía reconstruye el **estado
  final** de cada secuencia e indica el rango de diapositivas correspondiente.
- **Obstrucción de lectura:** en la diapositiva 17 el rótulo "Ordinary Least Squares (OLS)"
  tapa parte de la matriz $X$; la transcripción de §3.4 se tomó de la diapositiva 16, donde la
  misma matriz aparece completa.
- Elementos marcados **[verificar]**: el glifo de la tasa de aprendizaje en la diapositiva 18
  (aparece como "∝" en lugar de $\alpha$); la frase truncada "<0 pero que la media" sobre R²
  (diap. 97); y la precisión del ejemplo de fraude (diap. 71).
- **Erratas del original conservadas y señaladas en su lugar:** "Funciń de Costo" (diap. 15);
  "la Balanced Accuracy Balanceada" (diap. 54); "(\gamma)" sin renderizar en la tabla de
  clasificadores (diap. 28).

---

*Guía de estudio elaborada a partir de `Sem4_Modelado (1).pdf` (109 diapositivas).*
*Curso: **Planificación y Toma de Decisiones en IA**, Dra. Aurea Soriano-Vargas — UTEC, 2026.*
