# Informe Técnico — Agente Inteligente de Entrenamiento Personalizado

> **Tipo de documento:** Informe técnico académico  
> **Área:** Inteligencia Artificial — Sistemas híbridos de razonamiento  
> **Tecnologías:** Python · mlxtend · scikit-fuzzy · DEAP · Streamlit  
> **Pipeline:** Apriori → Lógica Difusa → Algoritmo Genético

---

## 1. Introducción

El diseño de rutinas de entrenamiento físico personalizadas es un problema con múltiples dimensiones de incertidumbre: el estado fisiológico del atleta varía día a día, las respuestas al entrenamiento son no lineales, y la combinación óptima de ejercicios depende simultáneamente del objetivo a largo plazo, la fatiga acumulada y el historial de sesiones previas. Un profesional de fitness experimentado integra estos factores de forma intuitiva; replicar ese razonamiento en un sistema computacional requiere técnicas capaces de manejar incertidumbre, extraer patrones y optimizar bajo restricciones múltiples.

Este proyecto presenta un **agente inteligente de entrenamiento** que genera rutinas semanales personalizadas de cinco días a partir del perfil del usuario. La propuesta central es encadenar tres técnicas clásicas de inteligencia artificial — minería de reglas de asociación (Apriori), lógica difusa e inferencia genética (algoritmo genético) — de forma que cada técnica aporte una capa de razonamiento diferente y el resultado final sea trazable paso a paso.

El énfasis del sistema no recae exclusivamente en la calidad de la rutina generada, sino en la **visibilidad del razonamiento**: el usuario puede inspeccionar qué reglas históricas activó su perfil, con qué intensidad difusa fue evaluado su estado físico y cómo evolucionó la calidad de la rutina a través de las generaciones del algoritmo genético. Esta transparencia convierte al sistema en una herramienta académica que ilustra cómo los distintos paradigmas de IA se pueden integrar en una cadena de razonamiento coherente.

El problema es relevante en la práctica porque el sobreentrenamiento es una de las causas más frecuentes de lesión y abandono del entrenamiento físico (American College of Sports Medicine, 2024). Un sistema que detecte señales de riesgo — como fatiga acumulada, ausencia de descanso o repetición consecutiva de grupos musculares — y las incorpore en la planificación semanal tiene valor preventivo directo.

---

## 2. Objetivos del Sistema

### 2.1 Objetivo General

Desarrollar un agente inteligente que, dado el perfil fisiológico y el historial de entrenamiento de un usuario, genere automáticamente una rutina semanal de cinco días personalizada, con razonamiento visible y justificable en cada etapa del proceso.

### 2.2 Objetivos Específicos

#### **Módulo Apriori**

- Extraer patrones de asociación estadísticamente significativos a partir de un historial de sesiones de entrenamiento.
- Identificar y activar reglas cuyo antecedente coincida con el contexto actual del usuario (grupo muscular, objetivo, intensidad estimada, días consecutivos).
- Proporcionar señales de advertencia sobre sobreentrenamiento al módulo genético.

#### **Módulo de Lógica Difusa**

- Evaluar el estado físico del usuario a partir de tres variables continuas: fatiga percibida, nivel de energía y días de descanso recientes.
- Producir un valor de intensidad recomendada en escala 0–10 con respaldo fisiológico documentado.
- Clasificar el estado en tres categorías operativas: recuperación, moderado, alto.

#### **Módulo de Algoritmo Genético**

- Generar y evolucionar una población de rutinas semanales candidatas.
- Optimizar la rutina considerando simultáneamente: alineación con el objetivo del usuario, variedad de grupos musculares, respeto al límite de intensidad difusa y señales de sobreentrenamiento de Apriori.
- Retornar la mejor rutina encontrada junto con el historial de convergencia por generación.

#### **Interfaz Streamlit**

- Presentar el razonamiento del agente en forma de pasos secuenciales expandibles.
- Visualizar funciones de membresía, grados de activación, curva de evolución del fitness y distribución de grupos musculares.
- Permitir comparar el comportamiento del agente ante tres perfiles de usuario representativos.

---

## 3. Arquitectura General del Sistema

### 3.1 Estructura de Componentes

El sistema está organizado en cuatro capas funcionales:

```
┌─────────────────────────────────────────────────────────────────┐
│                        app.py  (Streamlit)                      │
│              Entrada de usuario · Visualización                 │
└────────────────────────────┬────────────────────────────────────┘
                             │ perfil_usuario (dict)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   agent/agent.py  (GymAgent)                    │
│              Orquestador del pipeline de razonamiento           │
└──────┬──────────────────────┬──────────────────────┬────────────┘
       │                      │                      │
       ▼                      ▼                      ▼
┌─────────────┐   ┌───────────────────┐   ┌──────────────────────┐
│ apriori_    │   │  fuzzy_state.py   │   │ genetic_routine.py   │
│ rules.py    │   │  Lógica Difusa    │   │ Algoritmo Genético   │
│ Paso 1      │   │  Paso 2           │   │ Paso 3               │
└──────┬──────┘   └─────────┬─────────┘   └──────────┬───────────┘
       │ reglas_activas      │ intensidad_max          │ rutina +
       │                     │                         │ historial
       └─────────────────────┴─────────────────────────┘
                                    │
                                    ▼
                         resultado (dict con 4 claves)
```

### 3.2 Flujo de Datos

El pipeline ejecuta tres etapas en secuencia estricta:

**Etapa 0 — Inicialización**  
Al arrancar la aplicación, `GymAgent` carga `data/historial.csv` y ejecuta `extraer_reglas()` una única vez. El resultado (un DataFrame de 67 reglas de asociación) se almacena en memoria y se reutiliza en cada consulta. Esto es posible porque las reglas son estáticas respecto al historial cargado.

**Etapa 1 — Minería de reglas (Apriori)**  
El orquestador construye un diccionario de contexto con cuatro campos: grupo muscular planificado, intensidad estimada (derivada de fatiga y energía), objetivo del usuario y días consecutivos de entrenamiento. Se filtran las reglas cuyo antecedente sea subconjunto del contexto. El resultado (`reglas_activas`) es una lista de diccionarios con texto de la regla, consecuente y confianza.

**Etapa 2 — Evaluación difusa**  
Con las tres variables continuas del usuario (fatiga, energía, días de descanso), el sistema difuso infiere un valor de `intensidad_recomendada` en [0, 10]. Este valor actúa como **límite de seguridad** para la etapa siguiente.

**Etapa 3 — Optimización genética**  
El algoritmo genético recibe el objetivo del usuario, el límite de intensidad difusa y las reglas activas. Evoluciona durante 50 generaciones una población de 100 rutinas candidatas, aplicando cruce de dos puntos y mutación por atributo. Retorna la mejor rutina y el historial de fitness por generación.

**Resultado final**  
El orquestador consolida las salidas de los tres módulos en un diccionario estructurado con cuatro claves: `paso1_reglas`, `paso2_estado`, `paso3_rutina` y `recomendacion_final`. La interfaz consume este diccionario para generar todas las visualizaciones.

### 3.3 Coherencia de la Arquitectura

La arquitectura es coherente porque existe una **dependencia de datos unidireccional**: Apriori produce contexto que informa al genético sobre riesgos históricos; el difuso produce un umbral de seguridad fisiológica que el genético respeta en su función de fitness; el genético no retroalimenta ni modifica los módulos anteriores. Esto garantiza que el razonamiento sea trazable y reproducible.

---

## 4. Descripción Detallada de Módulos

### 4.1 Módulo Apriori (`modules/apriori_rules.py`)

#### Propósito

El módulo de reglas de asociación cumple la función de **memoria histórica del sistema**: detecta qué combinaciones de condiciones de entrenamiento se asociaron estadísticamente con resultados negativos (sobreentrenamiento, bajo rendimiento, fatiga elevada) o positivos (alto rendimiento, adecuada recuperación). Estas asociaciones se convierten en señales de advertencia que el algoritmo genético incorpora en su evaluación.

#### Discretización de Variables

Las variables continuas del historial no pueden usarse directamente en Apriori, que opera sobre conjuntos de ítems categóricos. El proceso de discretización convierte cada fila del CSV en una transacción:

| Variable | Rango original | Discretización |
|---|---|---|
| `fatiga_post` | 1–10 | baja (≤3), media (4–6), alta (≥7) |
| `dias_consecutivos` | 1–7+ | 1 día, 2–3 días, 4 o más días |
| `grupo_muscular` | categórico | ítem directo (`grupo_pecho`, etc.) |
| `intensidad` | bajo/media/alta | ítem directo (`intensidad_alta`, etc.) |
| `rendimiento` | bajo/normal/alto | ítem directo (`rendimiento_bajo`, etc.) |
| `objetivo` | 3 categorías | ítem directo (`objetivo_volumen`, etc.) |

Los umbrales de discretización de `fatiga_post` coinciden con la escala ROF (Rating of Fatigue) de Micklewright et al. (2017): valores ≥7 corresponden a fatiga alta, validada mediante correlación con frecuencia cardíaca y lactato sérico.

#### Extracción de Reglas

La extracción aplica el algoritmo Apriori de la librería `mlxtend` con:

- **Soporte mínimo:** 0.04 (una regla debe aparecer en al menos el 4% de las transacciones del historial).
- **Confianza mínima:** 0.70 (dado el antecedente, el consecuente debe cumplirse en al menos el 70% de los casos).

Con el historial actual de 500 sesiones, el sistema extrae **67 reglas** de asociación. Las reglas de mayor confianza (confianza = 1.0) identifican patrones como:

```
fatiga_post_alta + grupo_pecho → dias_consecutivos_4mas  (confianza: 1.0, lift: 2.67)
fatiga_post_alta + dias_consecutivos_2a3 → intensidad_alta  (confianza: 1.0, lift: 2.87)
```

Estas reglas revelan que la fatiga post-entrenamiento alta aparece sistemáticamente asociada con periodos de entrenamiento consecutivo prolongado, lo cual es consistente con la evidencia fisiológica sobre sobreentrenamiento.

#### Filtrado por Contexto y Sistema de Fallback

La función `reglas_aplicables()` aplica dos niveles de coincidencia:

**Nivel 1 — Coincidencia estricta:** El antecedente de la regla debe ser un subconjunto exacto del contexto actual del usuario. Solo se activan reglas cuyos antecedentes estén formados por ítems de tipo contexto (grupo, intensidad, objetivo, días consecutivos), excluyendo ítems de tipo outcome histórico (fatiga_post, rendimiento) que nunca pueden ser parte del input del usuario.

**Nivel 2 — Fallback por coincidencia parcial:** Si ninguna regla cumple coincidencia estricta, el sistema busca reglas con al menos un ítem en común con el contexto, con la restricción adicional de que no contengan ítems que contradigan el contexto actual (por ejemplo, si el contexto tiene `dias_consecutivos_1`, se descartan reglas con `dias_consecutivos_4mas` porque representan condiciones mutuamente excluyentes del mismo tipo de variable).

Este diseño de fallback garantiza que el sistema siempre intente proporcionar información contextual relevante, sin sacrificar la coherencia lógica de las reglas activadas.

#### Justificación del Dataset Sintético

**Decisión:** El historial (`data/historial.csv`) contiene 500 sesiones de entrenamiento generadas sintéticamente.

**Justificación:** En contextos de investigación académica y desarrollo de prototipos, el uso de datos sintéticos con patrones intencionados es una práctica legítima y documentada cuando no se dispone de datos reales por razones de privacidad, acceso o volumen. La condición de validez es que los patrones insertados reflejen fenómenos documentados en la literatura de dominio.

Los patrones del historial están basados en principios fisiológicos establecidos:

- **Piernas + intensidad alta → fatiga_post alta:** Grupos musculares grandes (cuádriceps, isquiotibiales, glúteos) generan mayor demanda metabólica y niveles de lactato más elevados que grupos pequeños (ACSM, 2024).
- **4+ días consecutivos → rendimiento bajo:** El sobreentrenamiento sin recuperación adecuada reduce el rendimiento; 24–48 horas de descanso entre sesiones es el estándar recomendado (UC Health/NASM, 2023).
- **Objetivo volumen + intensidad alta → series altas:** La hipertrofia requiere volumen de entrenamiento elevado; la combinación de alta intensidad con objetivos de volumen es coherente con los protocolos de fuerza (ACSM, 2024).

La transparencia sobre el origen sintético del dataset no debilita el sistema; al contrario, permite demostrar que Apriori extrae los patrones intencionados con métricas estadísticas verificables (confianza, soporte, lift), lo cual valida el funcionamiento del algoritmo sobre datos con estructura conocida.

---

### 4.2 Módulo de Lógica Difusa (`modules/fuzzy_state.py`)

#### Propósito

La lógica difusa resuelve el problema de evaluar el estado físico de una persona a partir de variables subjetivas y continuas. La fatiga percibida, la energía disponible y los días de descanso no tienen fronteras nítidas: no existe un umbral exacto donde la fatiga "baja" se convierte en fatiga "media". Los conjuntos difusos permiten modelar esta gradualidad de forma matemáticamente rigurosa, produciendo una recomendación de intensidad que refleja la combinación ponderada de múltiples condiciones simultáneas.

#### Variables de Entrada

| Variable | Rango | Escala de referencia |
| --- | --- | --- |
| Fatiga percibida | [0, 10] | Escala ROF (Micklewright et al., 2017); Borg CR10 (Borg, 1990) |
| Nivel de energía | [0, 10] | Hooper Index / TQR — Total Quality Recovery (Abdulaziz et al., 2024) |
| Días de descanso | [0, 7] | Guías de recuperación ACSM (2024); UC Health/NASM (2023) |

#### Funciones de Membresía

Cada variable se particiona en conjuntos lingüísticos con funciones de membresía cuya forma (triangular o trapezoidal) está justificada por la naturaleza de los conceptos fisiológicos representados.

**Principio de selección de forma:**

- **Funciones trapezoidales (trapmf)** en los conjuntos extremos: los valores en el extremo inferior o superior del universo son igualmente representativos del concepto. Una fatiga de 0 y una fatiga de 1 son ambas "plenamente baja"; una fatiga de 9 y una fatiga de 10 son ambas "plenamente alta". La zona plana evita gradientes artificiales en los límites del universo (Novatchkov & Baca, 2013).
- **Funciones triangulares (trimf)** en los conjuntos centrales: existe un valor "más representativo" (el pico del triángulo) y la pertenencia decrece simétricamente a ambos lados. Una fatiga de exactamente 5 es "más media" que una fatiga de 3 o de 7.

**Definiciones de membresía:**

*Fatiga — umbrales Borg CR10 / ROF:*

```
baja:  trapmf [0, 0, 2, 4]    — zona plana 0–2 (fatiga despreciable a leve)
media: trimf  [2, 5, 8]        — peak en 5 (fatiga moderada, entrenar con precaución)
alta:  trapmf [7, 9, 10, 10]   — umbral ≥7 conforme a ROF (fatiga intensa)
```

*Energía — Hooper Index / TQR:*

```
baja:  trapmf [0, 0, 2, 4]    — sin energía, rendimiento comprometido
media: trimf  [2, 5, 8]        — energía parcial, sesión equilibrada
alta:  trapmf [7, 9, 10, 10]   — alta disponibilidad, sesión intensa posible
```

*Días de descanso — UC Health / NASM (2023):*

```
poco:   trapmf [0, 0, 0.5, 2]  — 0 días: zona de riesgo de sobreentrenamiento
normal: trimf  [1, 2, 4]        — peak en 2 días (ventana óptima 24–48 h)
mucho:  trapmf [3, 5, 7, 7]    — ≥3 días: recuperación adecuada (zona plana)
```

*Intensidad recomendada (salida) — ACSM (2024) / Borg CR10:*

```
muy_baja: trapmf [0, 0, 1, 3]   — reposo / ejercicio muy ligero (<3 METs)
baja:     trimf  [1, 3, 5]
moderada: trimf  [3, 5, 7]       — rango moderado (3–6 METs ACSM)
alta:     trimf  [6, 8, 10]
muy_alta: trapmf [8, 9, 10, 10]  — máximo (≥6 METs vigoroso)
```

La variable de salida utiliza 5 niveles (vs. 3 del ACSM) para aportar mayor granularidad, siguiendo el modelo FC-TOF (Abdulaziz et al., 2024) que también adopta 5 categorías de intensidad.

#### Reglas Difusas

El sistema implementa 8 reglas que cubren los escenarios fisiológicos más relevantes:

| Regla | Condición | Consecuente | Fundamento |
|---|---|---|---|
| R1 | fatiga ALTA ∧ energía BAJA | intensidad MUY_BAJA | Combinación de máximo riesgo: cuerpo agotado sin recursos |
| R2 | fatiga ALTA ∧ descanso POCO | intensidad BAJA | Fatiga sin recuperación; prioridad absoluta de descanso |
| R3 | fatiga MEDIA ∧ energía MEDIA | intensidad MODERADA | Estado balanceado; entrenamiento controlado viable |
| R4 | fatiga BAJA ∧ energía ALTA | intensidad ALTA | Condiciones óptimas para sesión exigente |
| R5 | fatiga BAJA ∧ descanso MUCHO | intensidad MUY_ALTA | Máxima recuperación; sesión intensa justificada |
| R6 | fatiga MEDIA ∧ energía ALTA | intensidad ALTA | Energía compensa fatiga moderada |
| R7 | fatiga ALTA ∧ descanso MUCHO | intensidad MODERADA | Descanso prolongado mitiga fatiga; entrenamiento moderado posible |
| R8 | fatiga BAJA ∧ energía BAJA | intensidad MODERADA | Cuerpo descansado pero sin energía; no forzar |

#### Proceso de Inferencia y Defuzzificación

El sistema utiliza el motor de inferencia de Mamdani a través de `scikit-fuzzy`. El proceso sigue tres etapas:

1. **Fuzzificación:** cada valor de entrada se evalúa en todas sus funciones de membresía, produciendo grados de activación por conjunto.
2. **Evaluación de reglas:** cada regla se activa con un grado igual al mínimo de los grados de sus antecedentes (operador AND = mínimo en lógica Mamdani).
3. **Defuzzificación:** el conjunto difuso de salida resultante se convierte a un valor escalar mediante el método del centroide (centro de gravedad), que produce valores continuos y estables.

**Validación con perfiles representativos:**

| Perfil | Fatiga | Energía | Descanso | Intensidad obtenida | Categoría |
|---|---|---|---|---|---|
| Carlos (exhausto) | 8.5 | 2.0 | 0 | **2.15/10** | recuperación |
| Luis (normal) | 4.0 | 6.0 | 1 | **5.00/10** | moderada |
| Ana (óptima) | 1.5 | 9.0 | 2 | **8.00/10** | alta |

Los tres perfiles producen salidas fisiológicamente coherentes: Carlos no debe entrenar intensamente, Ana puede hacerlo al máximo, Luis tiene una sesión equilibrada.

#### Mecanismo de Fallback

En casos extremos o combinaciones de entrada no cubiertas por ninguna regla activa, `scikit-fuzzy` puede no converger. El sistema captura esta excepción y aplica una fórmula de respaldo:

```
intensidad_fallback = clip((energia - fatiga + dias_descanso) / 2.0, 0, 10)
```

Esta fórmula conserva la lógica cualitativa del sistema (más energía sube la intensidad, más fatiga la baja, más descanso la incrementa) sin depender del motor difuso. El fallback es defensivo y sus activaciones deben ser infrecuentes con inputs dentro del rango normal.

---

### 4.3 Módulo de Algoritmo Genético (`modules/genetic_routine.py`)

#### Propósito y Justificación del Enfoque

La generación de una rutina semanal de cinco días es un problema de optimización combinatoria: para cada día se debe elegir un grupo muscular (6 opciones), número de ejercicios (2–6), series (2–5) e intensidad (0–10), produciendo un espacio de búsqueda de aproximadamente 5 × 6 × 5 × 4 × 11 = 6.600 combinaciones por día, o ~1.25 × 10¹⁴ rutinas completas posibles. La búsqueda exhaustiva es inviable; la optimización por gradiente no aplica porque la función de evaluación no es diferenciable. Los algoritmos genéticos son especialmente adecuados para este tipo de espacios discretos combinatorios con funciones de fitness multicriterio (Holland, 1992).

#### Estructura del Cromosoma

Cada individuo es una lista plana de 20 enteros que codifica la rutina completa:

```
[grupo_0, ejercicios_0, series_0, intensidad_0,   ← Lunes
 grupo_1, ejercicios_1, series_1, intensidad_1,   ← Martes
 grupo_2, ejercicios_2, series_2, intensidad_2,   ← Miércoles
 grupo_3, ejercicios_3, series_3, intensidad_3,   ← Jueves
 grupo_4, ejercicios_4, series_4, intensidad_4]   ← Viernes
```

| Atributo | Rango | Codificación |
|---|---|---|
| `grupo_muscular` | 0–5 | índice en `["pecho","espalda","piernas","hombros","brazos","core"]` |
| `ejercicios` | 2–6 | entero (número de ejercicios distintos por día) |
| `series` | 2–5 | entero (series por ejercicio) |
| `intensidad_nivel` | 0–10 | entero (escala subjetiva 0–10) |

#### Operadores Genéticos

| Operador | Implementación | Parámetros |
|---|---|---|
| Selección | Torneo (selTournament, DEAP) | tournsize = 3 |
| Cruce | Dos puntos (cxTwoPoint, DEAP) | prob. por par = 0.80 |
| Mutación | Personalizada por atributo | prob. por individuo = 0.20; prob. por gen = 0.20 |

La mutación personalizada respeta los rangos de cada atributo: al mutar el campo `grupo_muscular`, el nuevo valor es un entero en [0, 5]; al mutar `intensidad_nivel`, el valor es un entero en [0, 10]. Esto evita que la mutación genere individuos fuera del espacio de soluciones válidas.

#### Función de Fitness

La función de evaluación combina tres criterios con pesos ponderados:

```
fitness = 0.6 × alineacion_objetivo + 0.4 × variedad_grupos − penalizaciones
```

**Alineación con objetivo (peso 0.6):**  
Para cada uno de los cinco días, se verifica si las series, ejercicios e intensidad caen dentro del rango ideal del objetivo declarado. Cada criterio cumplido suma 1/3; el score diario varía de 0 a 1; el score total es el promedio de los cinco días.

| Objetivo | Series ideales | Ejercicios ideales | Intensidad ideal |
|---|---|---|---|
| Volumen | 4–5 | 4–6 | 6–10 |
| Definición | 3–4 | 3–5 | 4–7 |
| Resistencia | 2–3 | 2–4 | 3–6 |

Estos rangos están basados en los principios de programación del entrenamiento del ACSM (2024): hipertrofia requiere alto volumen e intensidad; resistencia cardiovascular requiere menor carga con mayor variedad de movimientos.

**Variedad de grupos musculares (peso 0.4):**  
`variedad = grupos_únicos / 6`. Con 5 días y 6 grupos posibles, el máximo alcanzable es 5/6 ≈ 0.833, lo que favorece rutinas que distribuyan el trabajo entre distintos grupos musculares.

**Penalizaciones:**

| Penalización | Magnitud | Criterio |
|---|---|---|
| Exceso de intensidad sobre límite difuso | 0.05 por punto de exceso | `intensidad_gen > intensidad_max` |
| Grupo muscular repetido en días consecutivos | 0.15 por ocurrencia | Misma etiqueta en día i e i+1 |
| Reglas activas con consecuente de sobreentrenamiento | 0.10 por regla | `fatiga_post_alta`, `rendimiento_bajo`, `dias_consecutivos_4mas` |

La penalización por exceso de intensidad implementa el **límite de seguridad difuso**: si el sistema difuso recomendó intensidad máxima 4.0 y la rutina propone un día con intensidad 7, el exceso (3 puntos) genera una penalización de 0.15. Esto conecta directamente la salida del módulo difuso con la optimización genética.

La penalización por sobreentrenamiento implementa el **límite histórico Apriori**: si las reglas activas indican que el contexto del usuario está asociado con sobreentrenamiento, las rutinas que no lo compensan (con descanso o menor intensidad) reciben penalización adicional.

#### Parámetros del Algoritmo

| Parámetro | Valor | Justificación |
|---|---|---|
| Tamaño de población | 100 | Balance entre diversidad genética y costo computacional |
| Generaciones | 50 | Suficiente para convergencia observable; se verifica mediante historial |
| Prob. cruce por par | 0.80 | Valor estándar recomendado para GAs sobre espacios continuos/discretos (De Jong, 1975) |
| Prob. mutación por individuo | 0.20 | Evita convergencia prematura sin dominar la exploración |
| Prob. mutación por gen (indpb) | 0.20 | ~4 genes mutados por individuo en promedio (de 20) |
| Selección | Torneo tamaño 3 | Presión selectiva moderada; preserva diversidad mejor que ruleta pura |
| Semilla aleatoria | 42 | Reproducibilidad de resultados para fines de demostración |
| Hall of Fame | 1 individuo | Elitismo: garantiza que el mejor individuo nunca se pierde |

---

### 4.4 Módulo de Generación de Dataset (`data/historial.csv`)

#### Descripción

El dataset contiene 500 registros de sesiones de entrenamiento distribuidas entre 20 usuarios ficticios (`u001`–`u020`). Cada registro representa una sesión con las siguientes columnas:

| Columna | Tipo | Descripción |
|---|---|---|
| `usuario_id` | categórico | Identificador del usuario (`u001`–`u020`) |
| `dia_semana` | categórico | Día de la sesión (lunes–sábado) |
| `grupo_muscular` | categórico | Grupo trabajado (pecho/espalda/piernas/hombros/brazos/core) |
| `intensidad` | categórico | Nivel de intensidad registrado (baja/media/alta) |
| `duracion_min` | entero | Duración de la sesión en minutos |
| `fatiga_post` | entero [1–10] | Fatiga percibida post-sesión (escala ROF) |
| `rendimiento` | categórico | Autoevaluación del rendimiento (bajo/normal/alto) |
| `dias_consecutivos` | entero | Días consecutivos entrenando sin descanso |
| `objetivo` | categórico | Objetivo del usuario (volumen/definicion/resistencia) |

#### Patrones Intencionados

El dataset fue diseñado para incorporar cuatro patrones con soporte estadístico verificable:

1. **Fatiga alta → días consecutivos prolongados:** Sesiones con `fatiga_post` ≥ 7 se concentran en registros con `dias_consecutivos` ≥ 4.
2. **Piernas + alta intensidad → fatiga alta:** El grupo `piernas` con `intensidad = alta` produce consistentemente `fatiga_post` elevada.
3. **4+ días consecutivos → rendimiento bajo:** Rachas de entrenamiento sin descanso correlacionan con `rendimiento = bajo`.
4. **Objetivo volumen + intensidad alta → fatiga alta:** La combinación de objetivo de volumen con sesiones intensas produce alta fatiga post-entrenamiento.

Estos patrones son extraídos satisfactoriamente por Apriori con confianza ≥ 0.70, confirmando que el algoritmo funciona correctamente sobre el dataset.

#### Validez Académica del Dataset Sintético

El uso de datos sintéticos es válido bajo las siguientes condiciones, todas cumplidas en este proyecto:

1. **Transparencia:** el origen sintético se declara explícitamente en la documentación.
2. **Fundamentación:** los patrones se basan en principios fisiológicos publicados, no en valores arbitrarios.
3. **Verificabilidad:** los patrones son extraídos con métricas estadísticas medibles (soporte, confianza, lift), por lo que el correcto funcionamiento del algoritmo puede comprobarse independientemente.
4. **Propósito limitado:** el dataset sirve para demostrar el funcionamiento del pipeline, no para hacer inferencias poblacionales.

Esta aproximación es análoga al uso de benchmarks sintéticos en la evaluación de algoritmos de clasificación o clustering, práctica estándar en investigación de aprendizaje automático.

---

### 4.5 Interfaz de Usuario (`app.py`)

#### Propósito

La interfaz Streamlit tiene como objetivo principal hacer visible el razonamiento interno del agente. No es simplemente una capa de presentación: es la herramienta que transforma el pipeline computacional en un sistema explicable e interactivo.

#### Flujo de Interacción

1. El usuario configura su perfil en el panel izquierdo (nombre, nivel, objetivo, fatiga, energía, descanso, grupo planificado) o carga uno de los tres casos de demostración.
2. Al presionar **Ejecutar Agente**, el orquestador procesa el perfil completo.
3. Los resultados se presentan en secciones expandibles en el panel derecho, en el mismo orden lógico del pipeline:
   - **Paso 1:** gráfico de barras de confianza por regla activa + tabla de reglas.
   - **Paso 2:** gauge de intensidad recomendada + gráficos de funciones de membresía + tabla de grados de activación.
   - **Paso 3:** curva de evolución del fitness por generación + distribución de grupos musculares (donut chart) + intensidad por día (barras).
4. Debajo de los pasos, se muestra la rutina semanal en tabla con indicadores visuales de intensidad.
5. La recomendación final consolida el razonamiento en lenguaje natural.

#### Visualización del Sistema Difuso

Las funciones de membresía se grafican en tiempo real con los valores del usuario marcados mediante línea vertical punteada. Los conjuntos con mayor grado de activación se resaltan con mayor opacidad. Los puntos de intersección entre el valor del usuario y cada función de membresía muestran el grado exacto de pertenencia en el hover. Esta visualización hace pedagógicamente claro el concepto de pertenencia parcial simultánea a múltiples conjuntos difusos.

Las definiciones de funciones de membresía utilizadas para la visualización se importan directamente desde `modules/fuzzy_state.py` (constantes `DEFS_FATIGA`, `DEFS_ENERGIA`, `DEFS_DESCANSO`, `DEFS_INTENSIDAD`), garantizando que lo que se muestra al usuario corresponde exactamente a los parámetros usados en el cómputo real.

#### Comparación de Perfiles

Un expander adicional ejecuta los tres perfiles de demostración (Carlos, Ana, Luis) simultáneamente y produce una vista comparativa con tabla de métricas y gráfico de barras agrupado. Este componente permite validar visualmente que el agente produce respuestas diferenciadas ante perfiles fisiológicamente distintos, lo cual es esencial para demostrar la sensibilidad del sistema.

---

## 5. Integración entre Módulos

La integración es funcional en dos niveles:

### 5.1 Integración Apriori → Genético (señales de riesgo)

Las reglas activas de Apriori incluyen consecuentes como `fatiga_post_alta`, `rendimiento_bajo` y `dias_consecutivos_4mas`. El módulo genético mantiene un conjunto `CONSECUENTES_SOBREENTRENAMIENTO` con estos tokens. Durante la evaluación de fitness, si alguna regla activa tiene un consecuente perteneciente a este conjunto, se aplica una penalización de 0.10 por regla. Esto implementa el principio de que el conocimiento histórico sobre el usuario debe limitar la agresividad de la rutina propuesta.

### 5.2 Integración Difuso → Genético (límite de intensidad)

La salida del módulo difuso (`intensidad_recomendada`) se pasa al genético como `intensidad_max`. Cualquier gen cuya `intensidad_nivel` supere este límite recibe una penalización proporcional al exceso (0.05 por punto). Así, la evaluación fisiológica del estado del usuario se traduce directamente en una restricción blanda de la optimización combinatoria.

### 5.3 Coherencia del Pipeline

La integración es coherente porque las tres técnicas actúan sobre dimensiones complementarias del problema:

- **Apriori** razona sobre el *historial colectivo*: qué patrones emergieron de sesiones pasadas.
- **Difusa** razona sobre el *estado presente*: qué puede tolerar el cuerpo del usuario hoy.
- **Genético** razona sobre el *futuro inmediato*: qué rutina maximiza los objetivos sin violar las restricciones anteriores.

Esta separación de responsabilidades evita que cualquiera de las tres técnicas deba resolver un problema para el que no fue diseñada.

---

## 6. Decisiones de Diseño y Justificación Técnica

### 6.1 Uso de Dataset Sintético

| | |
|---|---|
| **Decisión** | Generar 500 sesiones sintéticas con patrones intencionados basados en fisiología del ejercicio. |
| **Razón** | No se disponía de datos reales de usuarios de gimnasio con las columnas requeridas. Los datos reales de entrenamiento físico son privados y difíciles de obtener en volumen suficiente para Apriori. |
| **Alternativa** | Eliminar Apriori y usar únicamente reglas expertas codificadas manualmente. |
| **Por qué no se usó** | Un sistema con solo dos técnicas (difuso + genético) pierde la capa de descubrimiento automático de patrones. Apriori aporta la capacidad de identificar asociaciones no obvias con métricas estadísticas verificables, lo que no se puede replicar con reglas escritas a mano. |
| **Tipo de decisión** | Simplificación práctica con validez académica, documentada y justificada. |

### 6.2 Mecanismo de Fallback en Apriori

| | |
|---|---|
| **Decisión** | Implementar dos niveles de coincidencia: estricta y parcial con anti-contradicción. |
| **Razón** | En espacios de contexto pequeños, la coincidencia estricta puede producir cero reglas para combinaciones de perfil poco frecuentes en el historial. El fallback garantiza que el genético siempre reciba alguna señal contextual. |
| **Alternativa** | Retornar lista vacía cuando no hay coincidencia estricta. |
| **Por qué no se usó** | Con lista vacía, el genético opera sin penalización de sobreentrenamiento en perfiles atípicos, precisamente los que más riesgo tienen. El fallback parcial es más robusto. |
| **Tipo de decisión** | Decisión de ingeniería práctica basada en análisis de casos extremos. |

### 6.3 Discretización Manual en Apriori

| | |
|---|---|
| **Decisión** | Discretizar `fatiga_post` con umbrales fijos en 3 y 6 (baja ≤3, media 4–6, alta ≥7). |
| **Razón** | Los umbrales coinciden con la escala ROF (Micklewright et al., 2017), que fue validada clínicamente mediante correlaciones con lactato y frecuencia cardíaca. |
| **Alternativa** | Discretización automática por percentiles o k-means. |
| **Por qué no se usó** | La discretización automática podría producir umbrales sin significado fisiológico, debilitando la interpretabilidad de las reglas generadas. La discretización manual con respaldo clínico es más defendible académicamente. |
| **Tipo de decisión** | Decisión respaldada académicamente. |

### 6.4 Penalizaciones Blandas vs. Restricciones Duras en el GA

| | |
| --- | --- |
| **Decisión** | Implementar límites difusos e históricos como penalizaciones sobre el fitness (restricciones blandas) en lugar de rechazar individuos inválidos (restricciones duras). |
| **Razón** | Las restricciones duras reducen drásticamente el espacio de búsqueda y pueden causar que el algoritmo no encuentre ninguna solución válida en generaciones tempranas. Las restricciones blandas permiten exploración amplia mientras presionan hacia soluciones que respetan los límites. |
| **Alternativa** | Rechazar cualquier individuo con intensidad > intensidad_max. |
| **Por qué no se usó** | Con restricciones duras, si `intensidad_max` es muy baja (por ejemplo 2.15 en el caso Carlos), casi todos los individuos iniciales serían inválidos y el algoritmo degeneraría. |
| **Tipo de decisión** | Decisión de ingeniería de algoritmos genéticos con base en literatura estándar (Deb, 2000). |

### 6.5 Semilla Aleatoria Fija

| | |
| --- | --- |
| **Decisión** | Fijar `random.seed(42)` al inicio de `evolucionar_rutina`. |
| **Razón** | Reproducibilidad completa de resultados para los casos de demostración. El mismo perfil de usuario produce siempre la misma rutina. |
| **Alternativa** | No fijar semilla (resultados estocásticos). |
| **Por qué no se usó** | En un contexto de demostración académica, la reproducibilidad es más valiosa que la variabilidad entre ejecuciones. Una semilla fija también facilita la depuración y comparación de experimentos. |
| **Tipo de decisión** | Decisión de ingeniería para contexto de demostración. En producción, se eliminaría o haría configurable. |

---

## 7. Limitaciones del Sistema

### 7.1 Dataset Sintético

La limitación más importante del sistema es que el módulo Apriori aprende de datos que fueron diseñados para ser aprendidos. Esto significa que las reglas extraídas reflejan los patrones que se insertaron intencionalmente, no patrones descubiertos en datos de usuarios reales. En un entorno de producción, el valor de Apriori aumentaría significativamente al trabajar con datos históricos auténticos que podrían revelar asociaciones no anticipadas.

### 7.2 Ausencia de Retroalimentación

El sistema no aprende de sus propias recomendaciones. Si un usuario ejecuta el agente repetidamente, las rutinas generadas no incorporan el historial de las sesiones previas del mismo usuario. El dataset es estático desde el inicio de la sesión.

### 7.3 Representación Simplificada de la Rutina

El cromosoma del algoritmo genético codifica la rutina solo mediante grupo muscular, número de ejercicios, series e intensidad. No incluye el nombre específico de ejercicios, tiempos de descanso entre series, variaciones de tempo ni progresión entre semanas. La rutina generada es un esquema de planificación, no un programa completo de entrenamiento.

### 7.4 Subjetividad de los Inputs del Usuario

El sistema acepta valores de fatiga y energía como inputs del usuario sin mecanismo de calibración. Dos usuarios que reportan fatiga = 7 pueden tener condiciones fisiológicas objetivamente distintas. La escala es subjetiva por diseño (escala ROF/CR10), lo cual introduce variabilidad de medición que el sistema no puede compensar.

### 7.5 Semilla Aleatoria Fija

La fijación de `random.seed(42)` garantiza reproducibilidad pero elimina la variabilidad estocástica del algoritmo genético. El sistema siempre producirá la misma rutina para el mismo perfil, independientemente del número de ejecuciones. En un uso real, esta característica podría no ser deseable.

### 7.6 Convergencia del Sistema Difuso

El fallback de `evaluar_estado` captura `Exception` de forma amplia, lo que incluye potencialmente errores de programación además de fallos de convergencia del motor difuso. Esto puede ocultar bugs en inputs extremos o modificaciones futuras del sistema de reglas.

---

## 8. Posibles Mejoras Futuras

### 8.1 Integración con Datos Reales

La mejora de mayor impacto sería reemplazar el dataset sintético con registros reales de sesiones de entrenamiento, anonimizados y con consentimiento. Esto permitiría a Apriori descubrir patrones auténticos y potencialmente contra-intuitivos, como asociaciones entre tipos de ejercicio y lesiones, o entre objetivos y adherencia al entrenamiento.

### 8.2 Actualización Dinámica del Historial

El sistema podría registrar las sesiones completadas por el usuario y agregarlas al historial después de cada semana. Esto convertiría a Apriori en un módulo que aprende incrementalmente del comportamiento del usuario específico, no solo de una población genérica.

### 8.3 Ajuste Automático de Parámetros del AG

Los pesos de la función de fitness (0.6 para alineación, 0.4 para variedad) y los rangos ideales por objetivo están fijados manualmente. Una mejora sería ajustarlos mediante optimización de hiperparámetros con retroalimentación del usuario (por ejemplo, valoración de la rutina generada), convirtiendo el sistema en un aprendiz activo.

### 8.4 Expansión del Modelo Difuso

El sistema actual usa tres variables de entrada. Podría enriquecerse con variables adicionales como frecuencia cardíaca en reposo, calidad del sueño (escala Hooper) o Percepción de Esfuerzo acumulada semanal, todas con respaldo en la literatura de ciencias del deporte.

### 8.5 Recomendaciones de Ejercicios Específicos

El algoritmo genético actualmente optimiza el esquema de la rutina (grupos, series, intensidad). Una extensión natural sería ampliar el cromosoma para incluir ejercicios concretos de una base de datos de movimientos, con restricciones adicionales de equipamiento disponible y nivel técnico del usuario.

### 8.6 Interfaz de Seguimiento Semanal

La aplicación Streamlit actual está diseñada para consultas individuales. Podría extenderse para gestionar múltiples semanas, visualizar la progresión del fitness a lo largo del tiempo y mostrar alertas si el historial del usuario presenta señales de sobreentrenamiento acumulado.

---

## 9. Validación Conceptual del Sistema

### 9.1 Coherencia Computacional

El pipeline respeta las precondiciones de cada técnica:

- Apriori recibe transacciones discretas (cumple el supuesto del algoritmo).
- El sistema difuso recibe valores continuos dentro de los universos definidos (entradas clampadas antes de la evaluación).
- El algoritmo genético recibe parámetros cuantitativos correctamente tipados (los genes son enteros con rangos válidos garantizados por la mutación personalizada).

No existen incompatibilidades de interfaz entre módulos. El orquestador maneja todos los tipos de datos intermedios sin conversiones implícitas.

### 9.2 Coherencia Lógica

El flujo de información tiene dirección única: cada módulo produce outputs que restringen o informan al siguiente, sin retroalimentación ni dependencias circulares. La división de responsabilidades es limpia:

- Apriori responde: *¿qué señales históricas son relevantes para este contexto?*
- Difuso responde: *¿qué puede tolerar este cuerpo hoy?*
- Genético responde: *¿qué rutina maximiza el objetivo respetando esas restricciones?*

Ninguno de los tres módulos intenta resolver la pregunta de otro, lo que garantiza que la arquitectura no tenga redundancias ni contradicciones internas.

### 9.3 Coherencia Fisiológica

Las decisiones de diseño de los tres módulos están ancladas en literatura de ciencias del ejercicio:

- Los umbrales de las funciones de membresía difusa (escala ROF, guías ACSM, ventana de recuperación 24–48h) tienen respaldo en estudios clínicos publicados.
- Los rangos de series, ejercicios e intensidad por objetivo en la función de fitness del GA corresponden a protocolos de entrenamiento establecidos en la práctica del fitness.
- Los patrones del dataset (sobreentrenamiento por días consecutivos, fatiga diferencial por grupo muscular) son consistentes con la fisiología del ejercicio documentada.

El sistema es, por tanto, académicamente defendible no solo desde la perspectiva computacional (algoritmos correctamente implementados) sino también desde la perspectiva del dominio de aplicación (los criterios de evaluación reflejan conocimiento experto en entrenamiento físico).

---

## 10. Conclusiones

Este proyecto demuestra que es posible construir un sistema de planificación de entrenamiento físico que sea simultáneamente **técnicamente riguroso, fisiológicamente fundamentado y pedagógicamente transparente**.

El principal aporte arquitectónico es la integración funcional de tres paradigmas de inteligencia artificial en una cadena de razonamiento donde cada técnica complementa las limitaciones de las demás. Apriori detecta patrones estadísticos que el experto humano puede no haber anticipado; la lógica difusa modela la incertidumbre inherente a la percepción subjetiva del esfuerzo; el algoritmo genético resuelve la optimización combinatoria respetando las restricciones producidas por los módulos anteriores. Ninguna de las tres técnicas podría abordar el problema de forma completa por sí sola.

El enfoque de **razonamiento visible** implementado en la interfaz no es una característica cosmética: refleja un principio de diseño fundamental para sistemas de apoyo a la toma de decisiones en salud y deporte. Un usuario que entiende por qué el agente recomienda una intensidad baja (porque su fatiga es 8.5/10 y no ha descansado en días) tiene mayor probabilidad de confiar y seguir la recomendación que si recibe solo el resultado final sin explicación.

Las limitaciones identificadas — principalmente el dataset sintético y la ausencia de retroalimentación — son inherentes al estado de prototipo del sistema y no comprometen la validez del enfoque. El dataset sintético demuestra que los algoritmos funcionan correctamente sobre datos con estructura conocida; el paso a datos reales es una extensión natural que no requeriría cambios arquitectónicos.

El sistema constituye una base sólida y extensible para trabajos futuros en sistemas híbridos de IA aplicados al entrenamiento físico personalizado, con una arquitectura modular que facilita la sustitución o mejora de cualquiera de sus componentes de forma independiente.

---

## Referencias

1. Borg, G. (1990). Psychophysical scaling with applications in physical work and the perception of exertion. *Scandinavian Journal of Work, Environment & Health*, 16(Suppl 1), 55–58.

2. Micklewright, D., St Clair Gibson, A., Gladwell, V., & Al Salman, A. (2017). Development and Validity of the Rating-of-Fatigue Scale. *Sports Medicine*, 47(11), 2375–2393. <https://doi.org/10.1007/s40279-017-0711-5>

3. American College of Sports Medicine. (2024). *ACSM's Guidelines for Exercise Testing and Prescription* (11th ed.). Wolters Kluwer.

4. Novatchkov, H., & Baca, A. (2013). Fuzzy Logic in Sports: A Review and an Illustrative Case Study in Strength Training. *International Journal of Computer Applications*, 71(6), 8–14.

5. Abdulaziz, M., Al-Dossary, S., & colleagues. (2024). Optimization of Sports Training Intensity Based on Fuzzy Control (FC-TOF). *International Journal of Computational Intelligence Systems*, Springer Nature. <https://doi.org/10.1007/s44196-026-01158-6>

6. UC Health / NASM. (2023). *Rest and Recovery for Athletes*. Retrieved from <https://www.uchealth.org/today/rest-and-recovery-for-athletes>

7. Holland, J. H. (1992). *Adaptation in Natural and Artificial Systems*. MIT Press.

8. Deb, K. (2000). An efficient constraint handling method for genetic algorithms. *Computer Methods in Applied Mechanics and Engineering*, 186(2–4), 311–338.

9. De Jong, K. A. (1975). *Analysis of the Behavior of a Class of Genetic Adaptive Systems*. Doctoral Dissertation, University of Michigan.
