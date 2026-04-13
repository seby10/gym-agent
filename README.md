# Agente Inteligente de Entrenamiento Personalizado

Proyecto académico — Inteligencia Artificial, UTA 2025

Sistema de IA que genera rutinas de entrenamiento semanales personalizadas encadenando tres técnicas de razonamiento: **Apriori**, **Lógica Difusa** y un **Algoritmo Genético**. El énfasis está en el razonamiento visible paso a paso, no solo en el resultado final.

---

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

Se abrirá automáticamente en `http://localhost:8501`.

---

## Arquitectura del agente

```
Usuario (Streamlit)
    └─▶ GymAgent.razonar()
            │
            ├─ Paso 1: apriori_rules.py
            │   Extrae reglas del historial (mlxtend)
            │   Filtra las que aplican al contexto actual
            │
            ├─ Paso 2: fuzzy_state.py
            │   Evalúa fatiga / energía / descanso (scikit-fuzzy)
            │   Devuelve intensidad recomendada [0–10]
            │
            └─ Paso 3: genetic_routine.py
                Genera rutina de 5 días (DEAP)
                Fitness = alineación objetivo + variedad − penalizaciones
```

### Módulos

| Archivo                      | Técnica              | Librería           |
| ---------------------------- | -------------------- | ------------------ |
| `modules/apriori_rules.py`   | Reglas de asociación | mlxtend            |
| `modules/fuzzy_state.py`     | Lógica difusa        | scikit-fuzzy       |
| `modules/genetic_routine.py` | Algoritmo genético   | DEAP               |
| `agent/agent.py`             | Orquestador          | —                  |
| `app.py`                     | Interfaz             | Streamlit + Plotly |

### Dataset

`data/historial.csv` — 200+ sesiones de entrenamiento sintéticas con patrones intencionales:

- Piernas + intensidad alta → fatiga post alta
- 4+ días consecutivos → rendimiento bajo
- Resistencia + intensidad media → rendimiento alto

Generado con `data/generar_dataset.py`.

---

## Casos de demostración

### Caso 1 — Usuario exhausto (Carlos)

| Campo             | Valor      |
| ----------------- | ---------- |
| Nivel             | Intermedio |
| Objetivo          | Volumen    |
| Fatiga            | 8.5 / 10   |
| Energía           | 2.0 / 10   |
| Días descanso     | 0          |
| Grupo planificado | Piernas    |

**Resultado esperado:** Estado = recuperación activa. El AG genera rutina de baja intensidad a pesar del objetivo de volumen. Las reglas Apriori activan alertas de sobreentrenamiento.

---

### Caso 2 — Usuario en condiciones óptimas (Ana)

| Campo             | Valor      |
| ----------------- | ---------- |
| Nivel             | Avanzado   |
| Objetivo          | Definición |
| Fatiga            | 1.5 / 10   |
| Energía           | 9.0 / 10   |
| Días descanso     | 2          |
| Grupo planificado | Pecho      |

**Resultado esperado:** Estado = alta intensidad. Rutina variada con intensidad 6–8, bien distribuida entre grupos musculares.

---

### Caso 3 — Principiante con estado normal (Luis)

| Campo             | Valor        |
| ----------------- | ------------ |
| Nivel             | Principiante |
| Objetivo          | Resistencia  |
| Fatiga            | 4.0 / 10     |
| Energía           | 6.0 / 10     |
| Días descanso     | 1            |
| Grupo planificado | Core         |

**Resultado esperado:** Estado = moderado. Rutina equilibrada, series cortas (2–3), intensidad controlada (3–5).

---

## Cómo usar la comparación en la app

En `app.py` puedes cargar cualquiera de los 3 casos del README desde el selector **Cargar caso de demostración**. Además, la sección **Comparar los 3 casos del README** ejecuta automáticamente a Carlos, Ana y Luis para mostrar:

- Intensidad recomendada por lógica difusa.
- Fitness final del algoritmo genético.
- Diferencias de comportamiento entre usuario exhausto, óptimo y normal.

Esto sirve para exponer claramente cómo cambia la salida del sistema según el estado del usuario.

---

## Razonamiento del agente — qué mostrar en la demo

1. **Paso 1 (Apriori):** mostrar que las reglas cambian según el perfil. Carlos activa alertas de fatiga; Ana activa patrones de alto rendimiento.
2. **Paso 2 (Difuso):** el gauge de intensidad baja drásticamente entre Carlos y Ana. Explicar las reglas difusas en voz.
3. **Paso 3 (AG):** la gráfica muestra cómo el fitness mejora generación a generación. La rutina de Carlos tiene intensidades 0–2; la de Ana tiene 6–8.
4. **Recomendación final:** texto generado automáticamente con el razonamiento completo del agente.

---

## Estructura de archivos

```
gym-agent/
├── app.py                      # Interfaz Streamlit
├── requirements.txt
├── PLAN.md
├── README.md
├── agent/
│   ├── __init__.py
│   └── agent.py                # Orquestador GymAgent
├── modules/
│   ├── __init__.py
│   ├── apriori_rules.py        # Módulo 1 — Apriori
│   ├── fuzzy_state.py          # Módulo 2 — Lógica Difusa
│   └── genetic_routine.py      # Módulo 3 — Algoritmo Genético
└── data/
    ├── historial.csv           # Dataset de entrenamiento
    └── generar_dataset.py      # Script generador
```
