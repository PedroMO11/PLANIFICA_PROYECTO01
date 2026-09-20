# Semana 2 — Fundamentos · Guía de estudio

| Campo | Valor |
|---|---|
| **Curso** | Planificación y Toma de Decisiones en IA |
| **Sesión** | Semana 2 · *Fundamentos* · 2026-2 |
| **Docente** | Dra. Aurea Soriano-Vargas (`aureasoriano` · asoriano@utec.edu.pe · `aurea-soriano`) |
| **Institución** | UTEC |
| **Fuente** | `Sem2_Fundamentos.pdf` — 80 diapositivas |
| **Atribución del pie** | *Planificación y Toma de Decisiones en IA*, Dr. Aurea Soriano-Vargas (2026) |

> **Propósito de esta guía:** reconstruir fielmente el contenido teórico de las 80 diapositivas
> de la sesión —incluyendo tablas, matrices y diagramas que en el PDF están como imágenes—
> para poder estudiar la sesión sin depender del deck original.

---

## 1. Mapa del curso: ubicación de la sesión

El deck cierra con el diagrama **"La historia de este curso"** (diapositivas 74 y 76). Organiza el
curso en cuatro carriles horizontales; el recuadro resaltado indica la sesión en curso.

```
Fundamentos        ┌──────────────────────┐
                   │ Motivación y         │  ← resaltado en la diap. 74 (ESTA SESIÓN)
                   │ terminología         │
                   └──────────┬───────────┘
                              │
Ciclo de vida      ┌──────────▼───┐   ┌──────────────┐   ┌───────────┐   ┌──────────────────┐
de la IA           │  Iniciación  │──▶│  Modelado y  │──▶│ Despliegue│──▶│ Desviación       │
                   └──────────────┘   │  rendimiento │   └───────────┘   │ conceptual       │
                    ↑ resaltado en     └──────────────┘                  │ (Concept Drift)  │
                      la diap. 76                                        └────────┬─────────┘
                      (PRÓXIMA SEMANA)                                            │
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

**Carriles (etiquetas verticales del original):** *Fundamentos* · *Ciclo de vida de la IA* ·
*IA en sistemas* · *Temas importantes de IA*.

---

## 2. Apertura: cierre de la sesión anterior

### 2.1 Predicción ≠ Decisión (diap. 2)

> **Una buena predicción solo ayuda si sabemos decidir con objetivos, restricciones y consecuencias.**

### 2.2 Conclusiones de la sesión previa (diap. 3)

| IA y Aprendizaje | Planificar para Decidir | IA Responsable |
|---|---|---|
| Transformar datos en predicciones y conocimiento. | Una buena predicción no garantiza una buena decisión. | La calidad de los datos determina la confiabilidad del sistema. |
| Existen distintos enfoques: supervisado, no supervisado. | Toda decisión requiere definir objetivos, restricciones y métricas. | Debemos considerar equidad, transparencia y seguridad. |
| El enfoque adecuado depende del problema y los datos disponibles. | Los sistemas deben adaptarse a la incertidumbre y cambios del entorno. | **El objetivo es construir sistemas efectivos, adaptativos y responsables.** |

### 2.3 Actividad interactiva en grupo: red de drones de reparto (diap. 4)

**Escenario.** Red de drones de reparto en un entorno urbano dinámico. La ciudad presenta
patrones de tráfico variables, un clima impredecible y zonas de exclusión aérea designadas.

| Paso 1: Definir objetivos | Paso 2: Definir restricciones | Paso 3: Métricas de éxito |
|---|---|---|
| Entregas seguras y rápidas. | Autonomía de batería. | Tiempo promedio de entrega. |
| Reducir costos y retrasos. | Capacidad máxima de carga. | % de entregas a tiempo. |
| Minimizar consumo energético. | Zonas de exclusión aérea. | Consumo energético. |
| Cumplir la normativa aérea. | Clima y tráfico. | Costo por entrega. |
| | Normas de seguridad. | Tasa de incidentes. |
| | | Cumplimiento de restricciones. |

**Solución (propuesta del deck):**
- Predecir demanda y condiciones del entorno.
- Optimizar rutas dinámicamente.
- Monitorear tráfico, clima y batería.
- Replanificar ante eventos inesperados.
- Priorizar seguridad y eficiencia.

*(Este caso se retoma en la diapositiva 18: "¿Dónde se encaja nuestra red de entrega con drones?")*

---

## 3. Objetivos y agenda de la sesión

### 3.1 Al finalizar la clase deberíamos tener los medios para responder (diap. 5)

1. ¿Por qué son necesarios los fundamentos?
2. ¿Qué conceptos básicos debemos diferenciar y qué desafíos afectan su uso?
3. ¿Qué decisiones aparecen en cada etapa del ciclo de vida de IA?

### 3.2 Agenda (diap. 6)

1. Motivación y fundamentos
2. Inteligencia Artificial y Aprendizaje Automático
3. Terminología y definiciones
4. Agentes y sistemas
5. Ciclo de vida de la IA

### 3.3 Disparadores de discusión (diap. 7–10)

- **¿Podemos confiar en la IA?** — *Mira el video y responde: ¿usarías este sistema?* (diap. 7–8;
  la diap. 12 muestra un fragmento de video de la **London School of Economics**).
- **¿Qué salió mal: datos, modelo o decisión?** (diap. 9)
- **¿Quién es responsable?** (diap. 10)

---

## 4. Motivación y fundamentos

### 4.1 ¿Por qué necesitamos "fundamentos" antes de planificar en IA? (diap. 11, 13)

| Razón | Explicación |
|---|---|
| **Evitan malentendidos** | AI, ML, IA generativa no son lo mismo. |
| **Lenguaje común** | Permite que equipos interdisciplinarios hablen la misma lengua. |
| **Guiar la toma de decisiones** | Qué es y qué no es IA evita expectativas irreales. |
| **Anticipar riesgos** | Sesgos, falta de datos, problemas requieren una visión crítica desde el inicio. |
| **Estructurar el ciclo de vida de IA** | Planificación efectiva sólo es posible si entendemos las etapas. |

### 4.2 Fundamentos = Mapa para decidir (diap. 14)

- Diseñar agentes capaces de **percibir, razonar y actuar**.
- **Planificación = anticipación** → cómo un sistema debe actuar ante escenarios futuros.
- Decisiones basadas en datos.
- Conectar cada etapa con decisiones estratégicas.

---

## 5. AI & Machine Learning — Motivación y Perspectivas (sección, diap. 15)

### 5.1 ¿Por qué IA? Una perspectiva empresarial (diap. 16)

Titulares mostrados como evidencia (*The Washington Post*):

| Fecha | Titular |
|---|---|
| Enero 2026 | *"These companies say AI is key to their four-day workweeks"* — Some companies are giving workers back more time as artificial intelligence takes on more tasks. |
| Octubre 2025 | *"As AI reshapes the job market, here are 16 roles it has created"* — Walmart, KPMG and Salesforce are among the companies debuting jobs as they adopt generative AI. |

**Ejes de valor empresarial:**
- Automatización inteligente en operaciones repetitivas.
- Experiencia del cliente: asistentes y recomendadores.
- Nuevos productos basados en datos y servicios digitales.
- Predicción para anticipar demanda, fallas o riesgos.

### 5.2 ¿Por qué IA? Una perspectiva de investigación (diap. 17)

| Ámbito | Aplicaciones |
|---|---|
| **Médica** | Diagnóstico por imágenes, monitoreo con wearables y descubrimiento de fármacos. |
| **Ambiental** | Predicción climática, optimización de energías renovables y conservación de biodiversidad. |
| **Social** | Ciudades inteligentes, predicción de crisis humanitarias y educación personalizada. |

### 5.3 Tipos de decisiones en IA (diap. 18)

| Tipo de decisión | Característica |
|---|---|
| **Bajo incertidumbre** | Información incompleta o ruidosa |
| **Secuenciales** | Cada decisión afecta las siguientes |
| **En tiempo real** | Restricción de latencia |
| **Multi-objetivo** | Balance entre múltiples criterios |

> Pregunta de discusión: **¿Dónde se encaja nuestra red de entrega con drones?**

### 5.4 ¿Por qué ahora? Evidencia cuantitativa (diap. 19)

**Cita 1 — WEF, *Future of Jobs Report 2025*:**
> Las competencias en IA se están convirtiendo en un cuello de botella crítico para el talento.
> **El 63 % de los empleadores identifica la falta de competencias como una barrera importante
> para la transformación empresarial.**

**Cita 2 — *Stanford AI Index Report 2026*:**
> La inversión corporativa global en IA alcanzó los **US$581,7 mil millones en 2025**, un aumento
> aproximado del **130 %** respecto al año anterior. La IA generativa está pasando rápidamente de
> la experimentación a la adopción masiva. **Casi el 53 % de la población adoptó herramientas de
> IA generativa en solo tres años.**

**Figura A (imagen) — WEF *Future of Jobs Report 2025*, "Top 10 fastest growing skills by 2030":**

| # | Competencia |
|---|---|
| 1 | AI and big data |
| 2 | Networks and cybersecurity |
| 3 | Technological literacy |
| 4 | Creative thinking |
| 5 | Resilience, flexibility and agility |
| 6 | Curiosity and lifelong learning |
| 7 | Leadership and social influence |
| 8 | Talent management |
| 9 | Analytical thinking |
| 10 | Environmental stewardship |

*Nota al pie de la figura original:* las competencias están categorizadas como *Cognitive skills,
Self-efficacy, Working with others, Management skills, Technology skills, Ethics*. Fuente:
World Economic Forum (2025), *Future of Jobs Report 2025*.

**Figura B (imagen) — "Speed of AI adoption by technology"** *(Source: The Project on Workforce at
Harvard, 2025 | Chart: 2026 AI Index report)*. Gráfico de líneas: eje X = *Years since the
introduction of the first mass-market product* (0 a 25); eje Y = *Adoption rate* (0 %–100 %).
Curvas con sus valores finales etiquetados:

| Serie | Etiqueta en el gráfico |
|---|---|
| Internet | **91 %, Internet** (source: ITU) |
| Internet (fuente alterna) | **76 %, Internet** (source: CPS) |
| Computer | **69 %, Computer** |
| GenAI | **53 %, GenAI** — curva corta, en los primeros ~3 años, con la pendiente más pronunciada |

> Lectura: la GenAI alcanza en ~3 años un nivel de adopción que a Internet y a la computadora
> les tomó más de una década.

### 5.5 ¿Por qué ahora? Factores habilitadores (diap. 20)

- Modelos generativos capaces de producir **texto, imagen, código y audio**.
  - *Democratización*: usuarios no expertos pueden producir contenido complejo.
- **Factores habilitadores clave**
  - Datos masivos.
  - Potencia computacional.
  - Algoritmos avanzados.
  - Ecosistema abierto: comunidades, repositorios y APIs accesibles globalmente.
  - Inversión y adopción: empresas y gobiernos apuestan fuerte en AI.
- **McKinsey:** 2023 fue el *"año de despegue"* de la IA generativa.

### 5.6 Retos actuales en la adopción de IA (diap. 21–22)

| ⚖ Sesgo en los datos | 🌱 Sostenibilidad | 👷 Condiciones laborales |
|---|---|---|
| Modelos heredan prejuicios sociales. | Entrenamiento de modelos consume gran cantidad de energía. | Trabajo invisible: *data labelers* en países del sur global. |
| Riesgo: decisiones injustas. | Dudas sobre huella de carbono y sostenibilidad a largo plazo. | Salarios bajos, exposición a contenido dañino, precariedad. |
| Ejemplo: algoritmos de selección laboral que discriminan por género. | Ejemplo: grandes LLMs consumen más energía que ciudades pequeñas. | |

**Pregunta de votación (diap. 22): ¿Qué desafío creen más urgente en la aplicación de la IA?**
- ⚖ Sesgo en los datos – Riesgo de decisiones injustas.
- 🌱 Sostenibilidad – Costos energéticos y huella de carbono.
- 👷 Condiciones laborales – Trabajo invisible y precario en etiquetado.

### 5.7 *Appropriate reliance*: ¿es seguro confiar en la IA? (diap. 23)

| Nivel de confianza | Definición del curso |
|---|---|
| 🚨 **Confianza ciega** (*over-reliance*) | Aceptar salidas de IA sin cuestionar. |
| ❌ **Desconfianza excesiva** (*under-reliance*) | Ignorar recomendaciones útiles de la IA. |
| ✅ **Confianza apropiada** (*appropriate reliance*) | Punto intermedio: **confianza calibrada** en función de la **fiabilidad del sistema** y el **contexto de uso**. |

*Materiales de apoyo mostrados en la diapositiva (imágenes):* el ensayo de opinión "Autonomous
Vehicles Are Driving Blind"; la nota "ChatGPT provided better customer service than his staff.
He fired them."; y el paper **"Appropriate Reliance on AI Advice: Conceptualization and the
Effect of Explanations"** (Karlsruhe Institute of Technology).
[verificar: la lista exacta de autores del paper no se lee con certeza en la imagen]

### 5.8 Autos autónomos: ¿confiar o no confiar? (diap. 24–26)

*(La diap. 24 muestra el video de TED-Ed "The ethical dilemma of self-driving cars".)*

| Promesa | Riesgo |
|---|---|
| Reducir errores humanos. | Casos difíciles, como señales borradas. |
| Optimizar movilidad y consumo. | Sensores sucios o fallidos. |
| Detectar situaciones complejas con sensores. | Responsabilidad difusa. |
| Aprender de grandes volúmenes de datos. | Confianza pública frágil. |

**Caso real (diap. 26).** En ciudades como **San Francisco** y **Phoenix**, hubo accidentes con
vehículos autónomos (**Waymo, Cruise**). Titular mostrado: *"Two driverless Waymo cars collide at
Phoenix Sky Harbor Airport"* — Two Waymo vehicles collided at Phoenix Sky Harbor Airport in
Arizona. Published July 31, 2025, By Joey Klender.

### 5.9 IA: promesas y miedos en la sociedad (diap. 27)

- **Empleos**
  - Automatización de tareas repetitivas.
  - Nuevas profesiones: curadores de datos, auditores de algoritmos, entrenadores de modelos.
- **Confianza social**
  - *Transparencia y explicabilidad*: la opacidad genera rechazo y miedo.
  - *Riesgo de injusticia percibida*: si la IA parece favorecer a unos y perjudicar a otros sin
    explicación, la sociedad puede bloquear su adopción.
- **Tendencia**
  - Encuestas recientes reflejan una mezcla de entusiasmo y preocupación.

### 5.10 Desafíos de IA en Latinoamérica (diap. 28)

| Desafío | Detalle |
|---|---|
| **Datos limitados o sesgados** | Datasets no representativos |
| **Infraestructura desigual** | Acceso limitado a cómputo |
| **Aplicaciones críticas** | Salud pública, transporte, seguridad |
| **Dependencia tecnológica** | Modelos desarrollados en otros contextos |

> **Diseñar IA en LATAM requiere adaptación, no solo adopción.**

---

## 6. Terminología y Definiciones (sección, diap. 29)

### 6.1 Tantas palabras… ¿qué significan realmente AI y ML? (diap. 30)

### 6.2 Ejemplo: Zuckerberg y el Senado (2018) (diap. 31)

**Facebook, 2018:** Zuckerberg presentó la IA como clave para moderar contenido a gran escala.

- ✅ **Funcionaba mejor:** detección de propaganda terrorista.
- ⚠ **Seguía siendo limitada:** *hate speech*, por depender de contexto, idioma e intención.
- 💡 **Lección:** decir que un sistema "usa IA" no define su capacidad real; esta depende de
  **la tarea, los datos y el contexto**.

> *"Today, we're just not there on that."*
> — Mark Zuckerberg, sobre la detección automática de *hate speech*, 2018.

### 6.3 IA como narrativa: las cuatro preguntas de control (diap. 32)

1. ¿Qué tarea concreta realiza?
2. ¿Qué datos usa y qué salida produce?
3. ¿Qué decisión humana o automática habilita?
4. ¿Cómo se mide si realmente funciona?

### 6.4 Mapa clásico: *Thinking vs Acting / Humanly vs Rationally* (diap. 33–40)

El deck construye la matriz por animación (una celda por diapositiva). **Los dos ejes** (diap. 33):

- **Eje horizontal — "Objetivo":** *Tipo de toma de decisiones*: apuntar a una decisión similar a
  la humana frente a una decisión ideal y racional → **Humanamente** vs **Racionalmente**.
- **Eje vertical — "Aplicación a":** *Objetivo de la aplicación de la IA*: pensar vs. actuar →
  **Pensar** vs **Actuar**.

**Matriz completa (diap. 37–38):**

| **Aplicación a** \ **Objetivo** | **Humanamente** | **Racionalmente** |
|---|---|---|
| **Pensar** | **Modelado cognitivo** — La IA debe razonar como lo hace la mente humana. ¿La IA reproduce procesos mentales? | **Leyes de pensamiento** — Seguir principios lógicos para alcanzar conclusiones racionales. ¿Llega a conclusiones lógicas? |
| **Actuar** | **Prueba de Turing** — La IA es válida si actúa como humano y pasa el test de Turing. *Chatbots avanzados* | **Agente Racional** — La IA actúa de forma autónoma para lograr el mejor resultado posible. *Agentes inteligentes, RL* |

**Pregunta de discusión (diap. 38–39):** *¿En qué corriente pondrían a ChatGPT? ¿Y a un algoritmo
de recomendación de Netflix?*

**Respuesta del deck (diap. 40):**

| ChatGPT | Netflix |
|---|---|
| Actúa humanamente: conversación natural. | Actúa racionalmente hacia un objetivo. |
| Genera texto plausible. | Su objetivo no es imitar al humano, sino maximizar la utilidad. |
| Puede incorporar preferencias vía *feedback*. | Optimiza recomendación y *engagement*. |
| No garantiza verdad ni razonamiento humano. | Aprende patrones de usuarios. |
| | Su "inteligencia" está ligada a utilidad. |

---

## 7. Enfoques de agentes (sección, diap. 41)

### 7.1 Visión basada en agentes (diap. 42)

> **Agente inteligente:** entidad que **percibe su entorno** (sensores) y **actúa sobre él**
> (actuadores) con el objetivo de cumplir metas.

```
        ┌──────────────────────┐            ┌────────────┐
        │ AGENTE               │            │            │
        │  Sensores  ◀─────────┼── percibe ─┤            │
        │      │               │            │            │
        │      ▼               │            │  AMBIENTE  │
        │ ┌──────────────────┐ │            │            │
        │ │ ¿Cómo es el      │ │            │            │
        │ │  mundo ahora?    │ │            │            │
        │ └────────┬─────────┘ │            │            │
        │          ▼           │            │            │
        │ ┌──────────────────┐ │            │            │
        │ │ Acciones a tomar │ │            │            │
        │ └────────┬─────────┘ │            │            │
        │          ▼           │            │            │
        │  Actuadores ─────────┼── acciones ▶            │
        └──────────────────────┘            └────────────┘
```

- Un agente inteligente interactúa con un entorno.
- El agente percibe los estados del entorno y decide qué acciones tomar en función de estos.

### 7.2 Modelo PEAS (diap. 43)

> Se define un agente por su **P**erformance, **E**nvironment, **A**ctuators y **S**ensors.
> *Una simple agregación de las medidas de rendimiento de un agente, en un entorno con actuadores
> y sensores (PEAS).*

**Ejemplo: el agente es un taxi autónomo.**

| Componente | Contenido en el ejemplo |
|---|---|
| **P**erformance Measures | Viajes seguros, rápidos, legales y cómodos, maximizar beneficios… |
| **E**nvironment | Carreteras, tráfico, peatones, clientes… |
| **A**ctuators | Dirección, acelerador, freno, señal, bocina, pantalla… |
| **S**ensors | Cámaras, sonar, velocímetro, GPS, acelerómetro, sensores del motor, teclado… |

### 7.3 Definición de IA orientada al nivel de inteligencia (diap. 44)

> **Inteligencia:** La capacidad de **percibir información**, **retenerla como conocimiento** y
> **aplicarla en conductas adaptativas** dentro de un entorno. Diferentes seres (y potencialmente
> máquinas) exhiben distintos grados de inteligencia.

**Figura (imagen) — Escala de *Capacidad cognitiva* (Cairo, 2011).** Curva creciente sobre el eje
vertical "Capacidad cognitiva", con seis hitos biológicos ordenados de menor a mayor:

| Orden | Ser | Capacidad asociada |
|---|---|---|
| 1 | **Goldfish** | Percepción básica, memoria de corto plazo |
| 2 | **Ratón** | Aprendizaje simple (laberintos, recompensas) |
| 3 | **Perro** | Reconoce comandos, emociones, memoria de largo plazo |
| 4 | **Ballena** | Comunicación compleja, coordinación social |
| 5 | **Mono** | Resolución de problemas, uso de herramientas |
| 6 | **Humano** | Lenguaje complejo, pensamiento abstracto, planificación, creatividad |

### 7.4 ¿Qué "niveles" de IA podemos definir? (diap. 45)

La misma curva de capacidad cognitiva se extiende más allá del **rango "biológico" de
inteligencia** (que abarca de *goldfish* a humano):

| Nivel | Definición textual del deck |
|---|---|
| **IA débil** | Buena en una cosa muy específica. |
| **IA fuerte** | Tan inteligente como un humano. |
| **Superinteligencia artificial** | *"mucho más inteligente que los mejores cerebros humanos en prácticamente todos los campos, incluyendo la creatividad científica, la sabiduría general y las habilidades sociales"* (**Bostrom 2006**). |

```
Capacidad
cognitiva  ▲                                          ╭─ Superinteligencia artificial
           │                                        ╭─╯
           │                                     ╭──╯
           │                                 ╭───╯
           │    IA débil     IA fuerte   ╭───╯
           │       ↓             ↓   ╭───╯
           │  ▁▂▃▄▅▆ ─────────────╯
           └──┴──────────┴──────────────────────────────────▶
              Rango          Superinteligencia artificial
             "biológico"
           de inteligencia
```

### 7.5 Nivel de inteligencia: IA Débil vs IA Fuerte (diap. 46)

| | **IA Débil** | **IA Fuerte** |
|---|---|---|
| **Definición** | Diseñada para un conjunto de tareas muy específicas | Diseñada para mejorar las capacidades comparables a las humanas |
| **Ejemplo 1** | Algoritmos de ajedrez | HAL en *2001: Una Odisea del Espacio* |
| **Ejemplo 2** | Asistentes inteligentes (p. ej., Siri, Alexa) | R2-D2 en *Star Wars* |
| **Ejemplo 3** | Coches autónomos | IA similar a la humana en *Her* |

> Obsérvese el contraste: los ejemplos de IA débil son sistemas reales; los de IA fuerte son
> ficción.

### 7.6 Etapas de IA: ANI / AGI / ASI (diap. 47)

*(Tabla que en el PDF está íntegramente como imagen — es el contenido que la capa de texto pierde
por completo.)*

| **ETAPAS DE IA** | **ANI** (IA Estrecha) | **AGI** (IA General) | **ASI** (Super IA) |
|---|---|---|---|
| **Etapa** | Pasado y Presente | Presente en progreso | Futuro |
| **Tipo** | Estrecha/Tarea única | General/Nivel humano | Superhumana |
| **Alcance de Inteligencia** | Solo Dominios Específicos | Comprensión multidominio | Abarcadora, auto-mejorable |
| **Estado de Desarrollo** | Ya logrado | En progreso activo | Todavía Teórico |
| **Nivel de Riesgo** | Bajo | Moderado | Aún no realizado |
| **Comparación Humana** | Un especialista cualificado | Un pensador completo | Alta (especulativa) |
| **Ejemplos** | Google Maps, recomendaciones de Netflix | LLMS experimentales | Más allá de los límites de la capacidad humana |

**Pregunta de cierre de la sección (diap. 48):** *¿Llegaremos a la Superinteligencia?*
*(La diap. 49 es una pausa de reflexión — "Let's pause for a moment of reflection".)*

---

## 8. Machine Learning en el marco de AI (sección, diap. 50)

### 8.1 Tipos de aprendizaje automático (diap. 51)

```
                    ┌──────────────────────────────┐
                    │ Tipos de Aprendizaje         │
                    │        Automático            │
                    └───┬──────────┬───────────┬───┘
                        │          │           │
        ┌───────────────▼──┐  ┌────▼─────────┐ ┌▼──────────────────┐
        │   Aprendizaje    │  │ Aprendizaje  │ │   Aprendizaje     │
        │   Supervisado    │  │No Supervisado│ │   por Refuerzo    │
        └──────────────────┘  └──────────────┘ └───────────────────┘
```

| | **Aprendizaje Supervisado** | **Aprendizaje No Supervisado** | **Aprendizaje por Refuerzo** |
|---|---|---|---|
| **Qué hace** | Correspondencia entre la entrada y la salida. | Identificar patrones previamente desconocidos en los datos. | Recibir retroalimentación (recompensas) por las acciones realizadas. |
| **Típico** | Regresión / clasificación | Reglas de agrupamiento / asociación | — *(el deck no indica tarea típica para esta categoría)* |

### 8.2 ¿Cómo el aprendizaje supervisado interactúa con los agentes? (diap. 52–53)

La diap. 52 repite el esquema simple Agente–Ambiente de §7.1; la diap. 53 lo expande en la
**arquitectura completa de un agente**, organizada en dos bloques: **Implementación** (arriba) y
**Capacidades** (abajo).

```
 IMPLEMENTACIÓN
 ┌──── Ambiente ────────────┐  ┌──────────────── Agente ─────────────────────┐
 │                          │  │ Agente de reflejo simple │ Agente de aprend. │
 │  Sensores   ┌──────────┐ │  │  ┌─────────────┐   ┌──────────────────┐     │
 │   ▤▤  ═════▶│          │═╪══╪═▶│ Backend de  │┈┈▶│   Backend de     │     │
 │             │ Frontend │ │  │  │ Ejecución*  │◀┈┈│   Aprendizaje*   │     │
 │  Actuadores │          │◀╪══╪══│             │   │                  │     │
 │   ▤▤  ◀═════└──────────┘ │  │  └──────▲──────┘   └────────▲─────────┘     │
 │                          │  └─────────┼───────────────────┼───────────────┘
 │  Conocimiento            │            │                   ┊
 │   ▭▭ ◀┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┘
 └──────────────────────────┘     ej., Reglas, Fórmulas, Modelo ML, etc.

 CAPACIDADES
        △ Actuando        △ Ejecutando        △ Aprendiendo
                               ╲                 ╱
                                └─── Pensando ──┘

 Leyenda:  ─ ─ ▶ Flujo de datos de cada agente
           ┈┈┈▶ Flujo de datos del agente de aprendizaje
           *Con diferentes grados de autonomía
```

**Lectura del diagrama:** el *Frontend* del agente recibe percepciones de los **Sensores** y emite
órdenes a los **Actuadores**; el **Backend de Ejecución** corresponde al *agente de reflejo
simple*, y el **Backend de Aprendizaje** (recuadrado y resaltado en la diapositiva) al *agente de
aprendizaje*, que alimenta el **Conocimiento** (reglas, fórmulas, modelo ML, etc.). En el plano de
**Capacidades**, *Actuando* corresponde al frontend, mientras que *Ejecutando* y *Aprendiendo*
son las dos formas de **Pensando**.

### 8.3 Backend de Ejecución vs Backend de Aprendizaje (diap. 54)

| **Backend de Ejecución** | **Backend de Aprendizaje** |
|---|---|
| Reglas fijas o lógica definida. | Ajusta patrones con datos. |
| Comportamiento más predecible. | Puede mejorar o degradarse. |
| No se adapta solo. | Requiere evaluación y monitoreo. |
| Ejemplo: temporizador simple. | Ejemplo: recomendador que aprende gustos. |

### 8.4 Agentes Reflexivos vs Agentes de Aprendizaje (diap. 55)

| **Agente Reflexivo** | **Agente de Aprendizaje** |
|---|---|
| Reacciona a percepciones inmediatas. | Usa experiencia y *feedback*. |
| No conserva experiencia útil. | Puede personalizar y adaptarse. |
| Simple, barato, limitado. | Necesita datos de calidad. |
| Bueno para entornos estables. | **Riesgo:** aprender patrones incorrectos. |

> *Nota de lectura del PDF:* en la diapositiva 55 los títulos "Backend de Ejecución vs Backend de
> Aprendizaje" y "Agentes Reflexivos vs Agentes de Aprendizaje" aparecen **superpuestos** (efecto
> de la animación al exportar a PDF). El contenido de ambas comparaciones se recoge en §8.3 y §8.4.

---

## 9. Sistemas y Pensamiento Sistémico (sección, diap. 56)

### 9.1 ¿Qué son los Sistemas de Servicio? (diap. 57)

> **"Un sistema de servicio comprende proveedores y clientes trabajando juntos para coproducir
> valor en cadenas o redes complejas"** (Spohrer et al.).

- Incluyen **entidades** (personas, organizaciones, tecnologías).
- Los vínculos entre ellas son **intercambios de valor**.
- No se trata solo de "dar un servicio" → se trata de **co-crearlo**.

**Diagrama (imagen).** Una red en la que los nodos son *entidades* (cuadros) y las aristas son
*enlaces*. En el centro, un ciclo de flechas rotula la **Co-creación de valor** entre el
*Proveedor de servicios A* y el *Servicio al Cliente I*; alrededor aparecen otros proveedores
(*Proveedor de servicios B*, *Proveedor de servicios …*) y otros servicios al cliente
(*Servicio al Cliente II*, *Servicio al Cliente …*), cada uno con su propia subred de entidades.
Leyenda del diagrama: **▭ Entidades** · **╱ Enlaces**.

### 9.2 Ejemplo: Lavaautos (diap. 58)

```
   Cliente ──▶ Recurso del cliente   ┐
                    (carro)          ├──▶ [ Value Co-Creation ] ──▶ "Limpiar el auto"
 Proveedor ──▶ Recursos del proveedor┘
               (lavadero, personal,
                productos básicos)
```

- **Proveedor:** espacio, agua, personal, productos y proceso.
- **Cliente:** auto, tiempo, pago y preferencias.
- **Valor:** auto limpio + experiencia satisfactoria.
- **Sin recursos de ambos lados, el servicio no existe.**

### 9.3 Ejemplo: Ecosistema de smartphones (diap. 59)

El diagrama representa una **pila de capas** (de abajo hacia arriba), con los clientes co-creando
valor en la cima:

```
        Cliente A  ⟲ Co-creación de valor ⟳  Cliente B  ·  Cliente C  ·  …
   ─────────────────────────────────────────────────────────────────────
   Proveedor de la aplicación,         p. ej., TikTok
   Proveedor del sistema operativo,    p. ej., Google
   Proveedor del teléfono inteligente, p. ej., Samsung
```

- **Proveedores:**
  - Fabricante de hardware (Samsung, Apple).
  - Sistema operativo (Google Android, iOS).
  - Aplicaciones (TikTok, WhatsApp).
- **Clientes:** millones de usuarios.
- **Valor co-creado:** comunicación, entretenimiento, productividad.
- Se trata de un **sistema interconectado**, no de un solo proveedor.

### 9.4 IA en sistemas de servicio (diap. 60–61)

> **La IA es habilitador, no isla tecnológica.** *(diap. 60, marcada "🧠 Conceptualización ·
> Sistemas de servicio")*

| Rol de la IA | Diapositiva 60 | Diapositiva 61 |
|---|---|---|
| **Automatización** | Chatbots y asistentes. | Chatbots, asistentes virtuales. |
| **Optimización** | Rutas, precios, asignación de recursos. | Rutas, precios dinámicos. |
| **Personalización** | Recomendadores y experiencias adaptativas. | Sistemas de recomendación. |
| **Monitoreo / Monitorización** | Predicción de fallos y mantenimiento preventivo. | Predicción de fallos, mantenimiento preventivo. |
| **Decisiones estratégicas** | *(no aparece)* | Análisis predictivo para la empresa y el cliente. |

> *Nota de lectura del PDF:* la diapositiva 60 usa una plantilla distinta (morada) al resto del
> deck y lleva su propio pie "Sem 2 · Fundamentos · 69/80". Cubre el mismo contenido que la
> diapositiva 61, con una variante de redacción.

---

## 10. Ciclo de Vida de la IA (sección, diap. 62)

### 10.1 Las cinco etapas (diap. 63–68)

- Conjunto de etapas que guían el desarrollo de soluciones de IA.
- Base para planificación y toma de decisiones en proyectos de IA.

```
┌────────────┐   ┌───────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│ Iniciación │──▶│ Entrenamiento │──▶│ Estimación │──▶│ Validación │──▶│ Despliegue │
└────────────┘   └───────────────┘   └────────────┘   └────────────┘   └────────────┘
```

**Decisiones y tareas por etapa** (cada bloque se despliega en una diapositiva distinta):

| Etapa | Diap. | Contenido textual del deck |
|---|---|---|
| **Iniciación** | 64 | • Definir problema y objetivo de negocio o investigación.<br>• Seleccionar dataset adecuado.<br>• Revisar calidad, balance, representatividad y sesgos.<br>• Documentar variables y supuestos. |
| **Entrenamiento** | 65 | • Elegir algoritmo según problema y datos.<br>• Separar entrenamiento, validación y prueba.<br>• Ajustar hiperparámetros con cuidado.<br>• Riesgo: *overfitting* y falsas expectativas. |
| **Estimación** | 66 | • Clasificación: *accuracy*, *precision*, *recall*, F1.<br>• Regresión: MAE, RMSE, R².<br>• Comparar con *baselines* simples.<br>• Medir subgrupos y no solo promedio global. |
| **Validación** | 67 | • Usar datos realmente no vistos.<br>• Evaluar robustez ante escenarios distintos.<br>• Revisar sesgos y seguridad operacional.<br>• Documentar resultados para reproducibilidad. |
| **Despliegue** | 68 | • Integrar el modelo al sistema de decisión.<br>• Asegurar escalabilidad, latencia y trazabilidad.<br>• Monitorear métricas, *logs* y errores críticos.<br>• Definir responsables, alertas y plan de *rollback*. |

> **Métricas nombradas por el curso en la etapa de Estimación** (diap. 66): para clasificación,
> *accuracy*, *precision*, *recall* y F1; para regresión, MAE, RMSE y R². El deck **no** desarrolla
> sus fórmulas en esta sesión — solo las enumera.

### 10.2 Riesgos en el ciclo de vida: Concept Drift y degradación del modelo (diap. 69)

| Concepto | Definición del curso |
|---|---|
| **Concept Drift** | Los patrones de los datos cambian con el tiempo. *Ejemplo: hábitos de consumo cambian tras pandemia.* |
| **Data Drift** | Cambia la distribución de las entradas. |
| **Degradación** | El modelo pierde precisión en el tiempo. |
| **Mitigación** | Reentrenamiento periódico, monitoreo de métricas, alertas. |

**Figura (imagen) — "Degradación del Modelo".** Gráfico de líneas; eje Y = *Precisión del Modelo*,
eje X = *Tiempo*. Una línea azul parte alta y cae en forma sigmoidal hasta estabilizarse en un
nivel bajo; una línea gris casi horizontal, ligeramente descendente, sirve de referencia. Una
línea vertical punteada marca el punto de quiebre, rotulado **"Cambio en la Distribución de Datos
(Concept Drift)"**. La región anterior al quiebre se etiqueta **"Distribución Inicial"** y la
posterior, **"Distribución Cambiada"**.

> Lectura: el drift no produce un fallo abrupto sino una **caída sostenida de la precisión** a
> partir del cambio de distribución — de ahí la necesidad de monitoreo continuo.

### 10.3 Pregunta de discusión (diap. 70)

Sobre el mismo diagrama de cinco etapas: **¿Dónde creen que ocurren más fallos?**

---

## 11. Conclusiones (sección, diap. 71–73)

| Idea clave | Desarrollo |
|---|---|
| **Predecir no es decidir** | Una buena predicción no garantiza una buena decisión. Para actuar correctamente se necesitan objetivos, restricciones, métricas y responsables. |
| **Entender la IA evita expectativas irreales** | Permite elegir mejor las soluciones y comprender sus verdaderas capacidades y limitaciones. |
| **La IA no es solo un modelo** | Opera dentro de procesos, servicios y ciclos de vida completos: iniciación, entrenamiento, estimación, validación y despliegue. |
| **La confianza debe diseñarse** | Sesgo, drift, degradación y errores requieren evidencia, monitoreo continuo, límites claros y participación humana cuando sea necesaria. |
| **El éxito combina tecnología y organización** | Una solución de IA efectiva depende tanto del diseño técnico como de la gestión, la gobernanza y la capacidad de adaptarse al entorno. |

---

## 12. Próxima semana (diap. 75–76)

En el mapa "La historia de este curso", el recuadro resaltado pasa de *Motivación y terminología*
a **Iniciación** (primera etapa del carril *Ciclo de vida de la IA*).

*(Diap. 77: "Thank you! / ¡Gracias! / OBRIGADO!". Diap. 78: espacio de preguntas — "Aquí se
contestan preguntas raras. ¡Pregunten sin miedo!". Diap. 79: **Kahoot**. Diap. 80: portada de
cierre.)*

---

## 13. Glosario de términos del curso

> **Nota de organización:** este glosario **reordena y recoge las definiciones tal como las da el
> deck**; no añade teoría nueva. La diapositiva de origen se indica entre paréntesis.

| Término | Definición según el curso |
|---|---|
| **Agente inteligente** | Entidad que percibe su entorno (sensores) y actúa sobre él (actuadores) con el objetivo de cumplir metas. (42) |
| **Agente de reflejo simple / Agente reflexivo** | Agente que reacciona a percepciones inmediatas, no conserva experiencia útil; simple, barato y limitado; bueno para entornos estables. Se implementa con el *Backend de Ejecución*. (53, 55) |
| **Agente de aprendizaje** | Agente que usa experiencia y *feedback*, puede personalizar y adaptarse; necesita datos de calidad; riesgo de aprender patrones incorrectos. Se implementa con el *Backend de Aprendizaje*. (53, 55) |
| **Agente Racional** | Corriente en la que la IA actúa de forma autónoma para lograr el mejor resultado posible (agentes inteligentes, RL). (37) |
| **AGI (IA General)** | Etapa de IA de tipo general/nivel humano, con comprensión multidominio; en progreso activo; riesgo moderado; comparable a "un pensador completo". (47) |
| **ANI (IA Estrecha)** | Etapa de IA estrecha/tarea única, solo dominios específicos; ya lograda; riesgo bajo; comparable a "un especialista cualificado". (47) |
| **Appropriate reliance (confianza apropiada)** | Punto intermedio entre *over-reliance* y *under-reliance*: confianza calibrada en función de la fiabilidad del sistema y el contexto de uso. (23) |
| **Aprendizaje supervisado** | Correspondencia entre la entrada y la salida. Típico: regresión/clasificación. (51) |
| **Aprendizaje no supervisado** | Identificar patrones previamente desconocidos en los datos. Típico: reglas de agrupamiento/asociación. (51) |
| **Aprendizaje por refuerzo** | Recibir retroalimentación (recompensas) por las acciones realizadas. (51) |
| **ASI (Super IA)** | Etapa futura, superhumana, abarcadora y auto-mejorable; todavía teórica. (47) |
| **Backend de Ejecución** | Reglas fijas o lógica definida; comportamiento más predecible; no se adapta solo. Ej.: temporizador simple. (54) |
| **Backend de Aprendizaje** | Ajusta patrones con datos; puede mejorar o degradarse; requiere evaluación y monitoreo. Ej.: recomendador que aprende gustos. (54) |
| **Capacidad cognitiva** | Eje con el que el curso ordena grados de inteligencia, del *goldfish* al humano y más allá (Cairo, 2011). (44–45) |
| **Ciclo de vida de la IA** | Conjunto de etapas que guían el desarrollo de soluciones de IA: Iniciación → Entrenamiento → Estimación → Validación → Despliegue. Base para planificación y toma de decisiones. (63) |
| **Co-creación de valor** | Producción conjunta de valor entre proveedor y cliente; sin recursos de ambos lados, el servicio no existe. (57–59) |
| **Concept Drift** | Los patrones de los datos cambian con el tiempo. Ej.: hábitos de consumo tras la pandemia. (69) |
| **Data Drift** | Cambia la distribución de las entradas. (69) |
| **Degradación (del modelo)** | El modelo pierde precisión en el tiempo. (69) |
| **Entidades / Enlaces** | En un sistema de servicio: las entidades son personas, organizaciones y tecnologías; los enlaces son intercambios de valor. (57) |
| **IA débil** | Buena en una cosa muy específica; diseñada para un conjunto de tareas muy específicas (ajedrez, Siri/Alexa, coches autónomos). (45–46) |
| **IA fuerte** | Tan inteligente como un humano; diseñada para capacidades comparables a las humanas (HAL, R2-D2, *Her*). (45–46) |
| **Inteligencia** | Capacidad de percibir información, retenerla como conocimiento y aplicarla en conductas adaptativas dentro de un entorno. (44) |
| **Leyes de pensamiento** | Corriente que sigue principios lógicos para alcanzar conclusiones racionales. (35) |
| **Modelado cognitivo** | Corriente en la que la IA debe razonar como lo hace la mente humana. (34) |
| **Over-reliance (confianza ciega)** | Aceptar salidas de IA sin cuestionar. (23) |
| **PEAS** | Modelo que define un agente por su **P**erformance, **E**nvironment, **A**ctuators y **S**ensors. (43) |
| **Planificación** | Anticipación: cómo un sistema debe actuar ante escenarios futuros. (14) |
| **Prueba de Turing** | Criterio según el cual la IA es válida si actúa como humano y pasa el test. (36) |
| **Sistema de servicio** | "Proveedores y clientes trabajando juntos para coproducir valor en cadenas o redes complejas" (Spohrer et al.). (57) |
| **Superinteligencia artificial** | "Mucho más inteligente que los mejores cerebros humanos en prácticamente todos los campos, incluyendo la creatividad científica, la sabiduría general y las habilidades sociales" (Bostrom 2006). (45) |
| **Under-reliance (desconfianza excesiva)** | Ignorar recomendaciones útiles de la IA. (23) |

---

## 14. Checklists de síntesis

> **Nota de organización:** esta sección **reorganiza criterios ya enunciados en el deck** para
> facilitar su uso; **no añade teoría nueva**. Cada ítem remite a la diapositiva de origen.

### 14.1 Checklist "¿esto realmente es IA útil?" (de las cuatro preguntas de la diap. 32)

- [ ] ¿Qué **tarea concreta** realiza el sistema?
- [ ] ¿Qué **datos** usa y qué **salida** produce?
- [ ] ¿Qué **decisión** humana o automática habilita?
- [ ] ¿Cómo se **mide** si realmente funciona?

### 14.2 Checklist de caracterización de un agente (modelo PEAS, diap. 43)

- [ ] **P** — ¿Cuáles son las medidas de rendimiento?
- [ ] **E** — ¿Cuál es el entorno?
- [ ] **A** — ¿Cuáles son los actuadores?
- [ ] **S** — ¿Cuáles son los sensores?

### 14.3 Checklist de decisiones por etapa del ciclo de vida (diap. 64–68)

| Etapa | Preguntas de control derivadas de los bullets del deck |
|---|---|
| **Iniciación** | ¿Está definido el problema y el objetivo de negocio/investigación? ¿El dataset es adecuado? ¿Se revisó calidad, balance, representatividad y sesgos? ¿Están documentadas variables y supuestos? |
| **Entrenamiento** | ¿El algoritmo corresponde al problema y a los datos? ¿Están separados entrenamiento, validación y prueba? ¿Los hiperparámetros se ajustaron con cuidado? ¿Se controló el *overfitting*? |
| **Estimación** | ¿Se usan las métricas correctas según el tipo de tarea? ¿Hay comparación contra *baselines* simples? ¿Se midieron subgrupos y no solo el promedio global? |
| **Validación** | ¿Los datos son realmente no vistos? ¿Se evaluó robustez ante escenarios distintos? ¿Se revisaron sesgos y seguridad operacional? ¿Los resultados permiten reproducir el trabajo? |
| **Despliegue** | ¿El modelo está integrado al sistema de decisión? ¿Se aseguró escalabilidad, latencia y trazabilidad? ¿Hay monitoreo de métricas, *logs* y errores críticos? ¿Están definidos responsables, alertas y plan de *rollback*? |

### 14.4 Checklist de riesgos de drift (diap. 69)

- [ ] ¿Se monitorean las métricas en producción a lo largo del tiempo?
- [ ] ¿Hay alertas definidas ante caídas de precisión?
- [ ] ¿Existe una política de reentrenamiento periódico?
- [ ] ¿Se distingue entre *concept drift* (cambian los patrones) y *data drift* (cambia la
      distribución de entradas)?

---

## 15. Notas de fidelidad sobre la reconstrucción

- Las **tablas ANI/AGI/ASI (diap. 47)**, la **matriz Thinking/Acting (diap. 33–38)**, el
  **Top 10 de competencias del WEF** y los **gráficos de adopción y degradación (diap. 19 y 69)**
  están en el PDF **únicamente como imágenes**: se transcribieron leyendo las diapositivas
  rasterizadas, no la capa de texto.
- Esta sesión **no contiene fórmulas matemáticas**. Las métricas de la etapa de Estimación
  (*accuracy*, *precision*, *recall*, F1, MAE, RMSE, R²) aparecen **solo enumeradas**, sin
  desarrollo algebraico en el deck; por fidelidad, esta guía no las desarrolla.
- Elementos marcados **[verificar]**: la lista completa de autores del paper sobre *appropriate
  reliance* mostrado en la diapositiva 23.
- Anomalías de exportación señaladas en su lugar: títulos superpuestos en la diapositiva 55 y
  plantilla/pie distintos en la diapositiva 60.

---

*Guía de estudio elaborada a partir de `Sem2_Fundamentos.pdf` (80 diapositivas).*
*Curso: **Planificación y Toma de Decisiones en IA**, Dra. Aurea Soriano-Vargas — UTEC, 2026.*
