"""
app.py — Interfaz Streamlit del Agente Inteligente de Entrenamiento
Punto de entrada: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from agent.agent import GymAgent


# ── Helper ────────────────────────────────────────────────────────────────────
def _color_intensidad(nivel: str) -> str:
    return {"recuperacion": "#ff6b6b", "moderada": "#ffd93d", "alta": "#6bcb77"}.get(
        nivel, "#aaa"
    )


# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Agente de Entrenamiento",
    page_icon="🏋️",
    layout="wide",
)


# ── Cargar agente (cacheado para no re-entrenar Apriori en cada interacción) ──
@st.cache_resource
def cargar_agente():
    return GymAgent("data/historial.csv")


agente = cargar_agente()

# ── Título ────────────────────────────────────────────────────────────────────
st.title("🏋️ Agente Inteligente de Entrenamiento Personalizado")
st.caption(
    "Razonamiento visible paso a paso: Apriori → Lógica Difusa → Algoritmo Genético"
)
st.divider()

# ── Layout: columna izquierda (inputs) | columna derecha (resultados) ─────────
col_inputs, col_resultados = st.columns([1, 2], gap="large")

with col_inputs:
    st.subheader("Perfil del Usuario")

    nombre = st.text_input("Nombre", value="Carlos")

    nivel = st.selectbox(
        "Nivel físico",
        ["principiante", "intermedio", "avanzado"],
        index=1,
    )

    objetivo = st.selectbox(
        "Objetivo",
        ["volumen", "definicion", "resistencia"],
        index=0,
    )

    fatiga = st.slider(
        "Fatiga actual", min_value=0.0, max_value=10.0, value=5.0, step=0.5
    )

    energia = st.slider(
        "Energía actual", min_value=0.0, max_value=10.0, value=6.0, step=0.5
    )

    dias_descanso = st.slider(
        "Días de descanso", min_value=0, max_value=7, value=1, step=1
    )

    grupo_planificado = st.selectbox(
        "Grupo muscular planificado hoy",
        ["pecho", "espalda", "piernas", "hombros", "brazos", "core"],
    )

    ejecutar = st.button("▶ Ejecutar Agente", type="primary", width="stretch")

# ── Panel de resultados ────────────────────────────────────────────────────────
with col_resultados:
    if not ejecutar:
        st.info("Configura tu perfil y presiona **Ejecutar Agente** para comenzar.")
    else:
        perfil = {
            "nombre": nombre,
            "nivel": nivel,
            "objetivo": objetivo,
            "fatiga_actual": fatiga,
            "energia_actual": energia,
            "dias_descanso": dias_descanso,
            "grupo_planificado": grupo_planificado,
        }

        with st.spinner("El agente está razonando..."):
            resultado = agente.razonar(perfil)

        # ── Paso 1 — Apriori ─────────────────────────────────────────────────
        with st.expander("🔵 Paso 1 — Reglas activas (Apriori)", expanded=True):
            reglas = resultado["paso1_reglas"]
            if reglas:
                df_reglas = pd.DataFrame(reglas)[["regla_texto", "confidence"]]
                df_reglas.columns = ["Regla detectada", "Confianza"]
                df_reglas["Confianza"] = df_reglas["Confianza"].map("{:.1%}".format)
                st.dataframe(df_reglas, width="stretch", hide_index=True)
            else:
                st.write("No se encontraron reglas aplicables al contexto actual.")

        # ── Paso 2 — Lógica Difusa ────────────────────────────────────────────
        with st.expander(
            "🟢 Paso 2 — Estado físico actual (Lógica Difusa)", expanded=True
        ):
            estado = resultado["paso2_estado"]
            intensidad_val = estado["intensidad_recomendada"]
            nivel_difuso = estado["nivel"]

            color_nivel = {"recuperacion": "🔴", "moderada": "🟡", "alta": "🟢"}
            etiqueta = color_nivel.get(nivel_difuso, "⚪")

            m1, m2, m3 = st.columns(3)
            m1.metric("Fatiga", f"{fatiga:.1f}/10")
            m2.metric("Energía", f"{energia:.1f}/10")
            m3.metric("Descanso", f"{dias_descanso}d")

            st.markdown(
                f"**Intensidad recomendada:** {intensidad_val:.1f}/10 &nbsp; "
                f"{etiqueta} **{nivel_difuso.capitalize()}**"
            )

            # Gauge de intensidad
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=intensidad_val,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Intensidad recomendada"},
                    gauge={
                        "axis": {"range": [0, 10]},
                        "bar": {"color": _color_intensidad(nivel_difuso)},
                        "steps": [
                            {"range": [0, 3.5], "color": "#ffcccc"},
                            {"range": [3.5, 6.5], "color": "#fff3cc"},
                            {"range": [6.5, 10], "color": "#ccffcc"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(height=250, margin=dict(t=40, b=0))
            st.plotly_chart(fig_gauge, width="stretch")

            st.caption(estado["razonamiento"])

        # ── Paso 3 — Algoritmo Genético ───────────────────────────────────────
        with st.expander("🔴 Paso 3 — Evolución del Algoritmo Genético", expanded=True):
            historial = resultado["paso3_rutina"]["historial_fitness"]
            fitness_final = resultado["paso3_rutina"]["fitness_final"]

            fig_evol = go.Figure()
            fig_evol.add_trace(
                go.Scatter(
                    x=list(range(1, len(historial) + 1)),
                    y=historial,
                    mode="lines+markers",
                    name="Mejor fitness",
                    line=dict(color="#4f8ef7", width=2),
                    marker=dict(size=4),
                )
            )
            fig_evol.update_layout(
                title=f"Evolución del fitness — {len(historial)} generaciones",
                xaxis_title="Generación",
                yaxis_title="Fitness (0–1)",
                yaxis=dict(range=[0, 1]),
                height=300,
                margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig_evol, width="stretch")
            st.metric("Fitness final", f"{fitness_final:.4f}")

        # ── Rutina final ──────────────────────────────────────────────────────
        st.subheader("📋 Rutina Semanal Generada")
        rutina = resultado["paso3_rutina"]["rutina"]

        df_rutina = pd.DataFrame(rutina)
        df_rutina.columns = [
            "Día",
            "Grupo Muscular",
            "Ejercicios",
            "Series",
            "Intensidad",
        ]
        df_rutina["Intensidad"] = df_rutina["Intensidad"].apply(
            lambda v: f"{v}/10 {'🔴' if v >= 7 else '🟡' if v >= 4 else '🟢'}"
        )

        st.dataframe(
            df_rutina,
            width="stretch",
            hide_index=True,
        )

        # ── Recomendación final ───────────────────────────────────────────────
        st.info(f"💬 **Recomendación del agente:** {resultado['recomendacion_final']}")

        # ── Razonamiento completo (expander) ──────────────────────────────────
        with st.expander("🔍 Ver razonamiento completo del agente"):
            st.json(
                {
                    "perfil": perfil,
                    "paso1_reglas": resultado["paso1_reglas"],
                    "paso2_estado": resultado["paso2_estado"],
                    "fitness_ag": resultado["paso3_rutina"]["fitness_final"],
                }
            )
