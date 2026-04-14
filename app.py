"""
app.py — Interfaz Streamlit del Agente Inteligente de Entrenamiento
Punto de entrada: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import skfuzzy as fuzz

from agent.agent import GymAgent


# ── Helper ────────────────────────────────────────────────────────────────────
def _color_intensidad(nivel: str) -> str:
    return {"recuperacion": "#ff6b6b", "moderada": "#ffd93d", "alta": "#6bcb77"}.get(
        nivel, "#aaa"
    )


def _membf(universo: np.ndarray, params: list[float]) -> np.ndarray:
    """Aplica trimf (3 params) o trapmf (4 params) según longitud de params."""
    if len(params) == 4:
        return fuzz.trapmf(universo, params)
    return fuzz.trimf(universo, params)


def _grados_triangulares(
    valor: float, definiciones: dict[str, list[float]]
) -> dict[str, float]:
    """Calcula grados de pertenencia (soporta trimf y trapmf)."""
    grados = {}
    max_val = max(p[-1] for p in definiciones.values())
    universo = np.arange(0, max_val + 0.5, 0.5)
    for etiqueta, params in definiciones.items():
        grados[etiqueta] = float(
            fuzz.interp_membership(universo, _membf(universo, params), valor)
        )
    return grados


_PALETTE = [
    (79, 142, 247),   # azul
    (239, 71, 111),   # rojo
    (6, 214, 160),    # verde
    (255, 217, 61),   # amarillo
    (167, 139, 250),  # violeta
]


def _rgba(rgb: tuple, alpha: float) -> str:
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"


def _plot_membresias(
    titulo: str,
    valor: float,
    universo: np.ndarray,
    definiciones: dict[str, list[float]],
) -> go.Figure:
    """
    Grafica funciones de membresía con area rellena.
    El set con mayor grado de activacion se resalta (opacidad alta).
    Un punto en la interseccion muestra el grado exacto de cada set activado.
    """
    fig = go.Figure()

    # Calcular grados de activacion para resaltar el dominante
    grados = {}
    for etiqueta, params in definiciones.items():
        y = _membf(universo, params)
        grados[etiqueta] = float(fuzz.interp_membership(universo, y, valor))

    dominante = max(grados, key=grados.get)

    for i, (etiqueta, params) in enumerate(definiciones.items()):
        rgb = _PALETTE[i % len(_PALETTE)]
        y = _membf(universo, params)
        grado = grados[etiqueta]
        es_dominante = etiqueta == dominante and grado > 0.01

        fill_alpha = 0.45 if es_dominante else 0.10
        line_width = 2.5 if es_dominante else 1.5

        label = f"{etiqueta}  ({grado:.2f})" if grado > 0.01 else etiqueta

        fig.add_trace(
            go.Scatter(
                x=universo,
                y=y,
                mode="lines",
                name=label,
                fill="tozeroy",
                fillcolor=_rgba(rgb, fill_alpha),
                line=dict(color=_rgba(rgb, 0.9), width=line_width),
            )
        )

        # Punto en la interseccion solo si el set esta activado
        if grado > 0.01:
            fig.add_trace(
                go.Scatter(
                    x=[valor],
                    y=[grado],
                    mode="markers",
                    marker=dict(color=_rgba(rgb, 1.0), size=9, line=dict(color="white", width=1.5)),
                    showlegend=False,
                    hovertemplate=f"{etiqueta}: {grado:.3f}<extra></extra>",
                )
            )

    fig.add_vline(
        x=valor,
        line_width=2,
        line_dash="dash",
        line_color="#333",
        annotation_text=f"{valor:.1f}",
        annotation_position="top right",
    )

    fig.update_layout(
        title=titulo,
        xaxis_title="Valor",
        yaxis_title="Grado de pertenencia",
        yaxis=dict(range=[0, 1.15]),
        height=280,
        margin=dict(t=45, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


CASOS_DEMO = {
    "Caso 1 - Carlos (exhausto)": {
        "nombre": "Carlos",
        "nivel": "intermedio",
        "objetivo": "volumen",
        "fatiga_actual": 8.5,
        "energia_actual": 2.0,
        "dias_descanso": 0,
        "grupo_planificado": "piernas",
    },
    "Caso 2 - Ana (óptima)": {
        "nombre": "Ana",
        "nivel": "avanzado",
        "objetivo": "definicion",
        "fatiga_actual": 1.5,
        "energia_actual": 9.0,
        "dias_descanso": 2,
        "grupo_planificado": "pecho",
    },
    "Caso 3 - Luis (normal)": {
        "nombre": "Luis",
        "nivel": "principiante",
        "objetivo": "resistencia",
        "fatiga_actual": 4.0,
        "energia_actual": 6.0,
        "dias_descanso": 1,
        "grupo_planificado": "core",
    },
}


def _resumen_resultado(perfil: dict, resultado: dict) -> dict:
    """Resume el resultado del agente para la vista comparativa."""
    estado = resultado["paso2_estado"]
    rutina = resultado["paso3_rutina"]["rutina"]
    intensidades = [fila["intensidad_nivel"] for fila in rutina]
    return {
        "Nombre": perfil["nombre"],
        "Objetivo": perfil["objetivo"],
        "Fatiga": perfil["fatiga_actual"],
        "Energía": perfil["energia_actual"],
        "Descanso": perfil["dias_descanso"],
        "Intensidad difusa": estado["intensidad_recomendada"],
        "Nivel": estado["nivel"],
        "Fitness AG": resultado["paso3_rutina"]["fitness_final"],
        "Intensidad media rutina": round(float(np.mean(intensidades)), 2),
    }


# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Agente de Entrenamiento",
    layout="wide",
)


# ── Cargar agente (cacheado para no re-entrenar Apriori en cada interacción) ──
@st.cache_resource
def cargar_agente():
    return GymAgent("data/historial.csv")


agente = cargar_agente()

# ── Título ────────────────────────────────────────────────────────────────────
st.title("Agente Inteligente de Entrenamiento Personalizado")
st.caption(
    "Razonamiento visible paso a paso: Apriori → Lógica Difusa → Algoritmo Genético"
)
st.divider()

# ── Layout: columna izquierda (inputs) | columna derecha (resultados) ─────────
col_inputs, col_resultados = st.columns([1, 2], gap="large")

with col_inputs:
    st.subheader("Perfil del Usuario")

    caso_demo = st.selectbox(
        "Cargar caso de demostración",
        ["Personalizado"] + list(CASOS_DEMO.keys()),
    )

    perfil_demo = CASOS_DEMO.get(caso_demo, {})
    nombre_default = perfil_demo.get("nombre", "Carlos")
    nivel_default = perfil_demo.get("nivel", "intermedio")
    objetivo_default = perfil_demo.get("objetivo", "volumen")
    fatiga_default = perfil_demo.get("fatiga_actual", 5.0)
    energia_default = perfil_demo.get("energia_actual", 6.0)
    descanso_default = perfil_demo.get("dias_descanso", 1)
    grupo_default = perfil_demo.get("grupo_planificado", "core")

    nombre = st.text_input("Nombre", value=nombre_default)

    nivel = st.selectbox(
        "Nivel físico",
        ["principiante", "intermedio", "avanzado"],
        index=["principiante", "intermedio", "avanzado"].index(nivel_default),
    )

    objetivo = st.selectbox(
        "Objetivo",
        ["volumen", "definicion", "resistencia"],
        index=["volumen", "definicion", "resistencia"].index(objetivo_default),
    )

    fatiga = st.slider(
        "Fatiga actual",
        min_value=0.0,
        max_value=10.0,
        value=float(fatiga_default),
        step=0.5,
    )

    energia = st.slider(
        "Energía actual",
        min_value=0.0,
        max_value=10.0,
        value=float(energia_default),
        step=0.5,
    )

    dias_descanso = st.slider(
        "Días de descanso",
        min_value=0,
        max_value=7,
        value=int(descanso_default),
        step=1,
    )

    grupo_planificado = st.selectbox(
        "Grupo muscular planificado hoy",
        ["pecho", "espalda", "piernas", "hombros", "brazos", "core"],
        index=["pecho", "espalda", "piernas", "hombros", "brazos", "core"].index(
            grupo_default
        ),
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

        st.info(
            f"Demo activa: {caso_demo if caso_demo != 'Personalizado' else 'perfil manual'}"
        )

        with st.expander("Comparar los 3 casos del README", expanded=False):
            st.write(
                "Esta comparación usa los tres perfiles documentados en el README para mostrar cómo cambia la salida del agente según el estado del usuario."
            )

            with st.spinner("Generando comparación de los 3 casos..."):
                comparaciones = []
                resultados_demo = []
                for etiqueta_caso, perfil_base in CASOS_DEMO.items():
                    resultado_caso = agente.razonar(perfil_base)
                    resultados_demo.append((etiqueta_caso, perfil_base, resultado_caso))
                    comparaciones.append(
                        _resumen_resultado(perfil_base, resultado_caso)
                    )

            df_comparacion = pd.DataFrame(comparaciones)
            st.dataframe(df_comparacion, width="stretch", hide_index=True)

            fig_comp = go.Figure()
            fig_comp.add_trace(
                go.Bar(
                    x=df_comparacion["Nombre"],
                    y=df_comparacion["Intensidad difusa"],
                    name="Intensidad difusa",
                    marker=dict(color="#4f8ef7"),
                )
            )
            fig_comp.add_trace(
                go.Bar(
                    x=df_comparacion["Nombre"],
                    y=df_comparacion["Fitness AG"],
                    name="Fitness AG",
                    marker=dict(color="#ef476f"),
                )
            )
            fig_comp.update_layout(
                title="Comparación entre los 3 casos",
                barmode="group",
                yaxis_title="Valor",
                height=320,
                margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig_comp, width="stretch")

            c1, c2, c3 = st.columns(3)
            for columna, (etiqueta_caso, perfil_base, resultado_caso) in zip(
                [c1, c2, c3], resultados_demo
            ):
                with columna:
                    estado_caso = resultado_caso["paso2_estado"]
                    st.markdown(f"**{etiqueta_caso}**")
                    st.metric(
                        "Intensidad recomendada",
                        f"{estado_caso['intensidad_recomendada']:.1f}",
                    )
                    st.metric(
                        "Fitness final",
                        f"{resultado_caso['paso3_rutina']['fitness_final']:.4f}",
                    )
                    st.caption(estado_caso["razonamiento"])

        st.markdown("### Qué hace el agente, paso a paso")
        tab_guion, tab_tecnico = st.tabs(["Guion para exponer", "Detalle técnico IA"])

        with tab_guion:
            st.markdown(
                """
1. El sistema toma tu perfil actual: objetivo, fatiga, energía, descanso y grupo muscular.
2. Con esos datos, consulta patrones históricos con Apriori para detectar riesgos o señales útiles.
3. Luego estima tu estado real con lógica difusa y produce una intensidad recomendada entre 0 y 10.
4. Esa intensidad actúa como límite de seguridad para el algoritmo genético.
5. El algoritmo genético evoluciona rutinas durante varias generaciones y conserva la de mejor fitness.
6. Finalmente, el agente entrega una rutina semanal y una explicación en lenguaje natural.
                """
            )

            df_flujo = pd.DataFrame(
                [
                    {
                        "Paso": "1. Apriori",
                        "Entrada": "Grupo, objetivo, intensidad estimada, días consecutivos",
                        "Proceso": "Filtra reglas de asociación cuyo antecedente coincide con el contexto",
                        "Salida": "Reglas activas con confianza",
                    },
                    {
                        "Paso": "2. Lógica difusa",
                        "Entrada": "Fatiga, energía, descanso",
                        "Proceso": "Aplica membresías triangulares + reglas difusas + defuzzificación",
                        "Salida": "Intensidad recomendada y nivel (recuperación/moderada/alta)",
                    },
                    {
                        "Paso": "3. Algoritmo genético",
                        "Entrada": "Objetivo, intensidad máxima, reglas activas",
                        "Proceso": "Selecciona, cruza y muta poblaciones de rutinas por generaciones",
                        "Salida": "Rutina semanal óptima + historial de fitness",
                    },
                ]
            )
            st.dataframe(df_flujo, width="stretch", hide_index=True)

        with tab_tecnico:
            st.markdown("**Función objetivo del AG (idea general):**")
            st.latex(
                r"fitness = 0.6\cdot alineacion\_objetivo + 0.4\cdot variedad - penalizaciones"
            )
            st.markdown(
                """
- Penaliza intensidades por encima del límite difuso.
- Penaliza repetir el mismo grupo muscular en días consecutivos.
- Penaliza si las reglas activas sugieren sobreentrenamiento.

**Cómo conectar los 3 módulos al explicarlo:**
- Apriori aporta conocimiento histórico.
- Difuso aporta adaptación al estado fisiológico actual.
- Genético optimiza la planificación semanal respetando restricciones.
                """
            )

        # ── Paso 1 — Apriori ─────────────────────────────────────────────────
        with st.expander("🔵 Paso 1 — Reglas activas (Apriori)", expanded=True):
            reglas = resultado["paso1_reglas"]
            contexto = {
                "grupo_muscular": grupo_planificado,
                "intensidad": "alta"
                if energia - fatiga >= 4
                else "media"
                if energia - fatiga >= 0
                else "baja",
                "objetivo": objetivo,
                "dias_consecutivos": 5
                if dias_descanso == 0
                else 3
                if dias_descanso == 1
                else 1,
            }

            st.caption(
                f"Contexto construido para Apriori: grupo={contexto['grupo_muscular']} | "
                f"intensidad={contexto['intensidad']} | objetivo={contexto['objetivo']} | "
                f"dias_consecutivos={contexto['dias_consecutivos']}"
            )

            if reglas:
                df_reglas = pd.DataFrame(reglas)[["regla_texto", "confidence"]]
                df_reglas = df_reglas.rename(
                    columns={
                        "regla_texto": "Regla detectada",
                        "confidence": "Confianza",
                    }
                )

                fig_conf = go.Figure(
                    go.Bar(
                        x=df_reglas["Confianza"],
                        y=df_reglas["Regla detectada"],
                        orientation="h",
                        marker=dict(color="#4f8ef7"),
                    )
                )
                fig_conf.update_layout(
                    title="Confianza por regla activa",
                    xaxis_title="Confianza",
                    yaxis_title="Reglas",
                    xaxis=dict(range=[0, 1], tickformat=".0%"),
                    height=260,
                    margin=dict(t=50, b=40),
                )
                st.plotly_chart(fig_conf, width="stretch")

                df_reglas_show = df_reglas.copy()
                df_reglas_show["Confianza"] = df_reglas_show["Confianza"].map(
                    "{:.1%}".format
                )
                st.dataframe(df_reglas_show, width="stretch", hide_index=True)
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

            # Parámetros sincronizados con modules/fuzzy_state.py
            # Sets de extremo → trapmf (4 puntos); sets de centro → trimf (3 puntos)
            # Novatchkov & Baca (2013); MDPI cardiorespiratory (2019)
            defs_fatiga = {
                "baja":  [0, 0, 2, 4],    # trapmf — plano 0-2 (ROF baja)
                "media": [2, 5, 8],        # trimf   — peak en 5
                "alta":  [7, 9, 10, 10],   # trapmf — plano 9-10, umbral ≥7 (ROF)
            }
            defs_energia = {
                "baja":  [0, 0, 2, 4],    # trapmf — plano 0-2
                "media": [2, 5, 8],        # trimf   — peak en 5
                "alta":  [7, 9, 10, 10],   # trapmf — plano 9-10 (TQR/Hooper)
            }
            defs_descanso = {
                "poco":   [0, 0, 0.5, 2],  # trapmf — plano en 0 (sin descanso)
                "normal": [1, 2, 4],        # trimf   — peak 2d (24-48h óptimo)
                "mucho":  [3, 5, 7, 7],    # trapmf — plano 5-7 (≥3d adecuado)
            }
            defs_intensidad = {
                "muy_baja": [0, 0, 1, 3],   # trapmf — plano 0-1
                "baja":     [1, 3, 5],       # trimf
                "moderada": [3, 5, 7],       # trimf
                "alta":     [6, 8, 10],      # trimf
                "muy_alta": [8, 9, 10, 10],  # trapmf — plano 9-10
            }

            tab_fuzzy_1, tab_fuzzy_2 = st.tabs(
                [
                    "Funciones de membresía",
                    "Grados activados",
                ]
            )

            with tab_fuzzy_1:
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        _plot_membresias(
                            "Fatiga", fatiga, np.arange(0, 10.5, 0.5), defs_fatiga
                        ),
                        width="stretch",
                    )
                    st.plotly_chart(
                        _plot_membresias(
                            "Descanso",
                            dias_descanso,
                            np.arange(0, 7.5, 0.5),
                            defs_descanso,
                        ),
                        width="stretch",
                    )
                with c2:
                    st.plotly_chart(
                        _plot_membresias(
                            "Energía", energia, np.arange(0, 10.5, 0.5), defs_energia
                        ),
                        width="stretch",
                    )
                    st.plotly_chart(
                        _plot_membresias(
                            "Salida: intensidad recomendada",
                            intensidad_val,
                            np.arange(0, 10.5, 0.5),
                            defs_intensidad,
                        ),
                        width="stretch",
                    )

            with tab_fuzzy_2:
                g_f = _grados_triangulares(fatiga, defs_fatiga)
                g_e = _grados_triangulares(energia, defs_energia)
                g_d = _grados_triangulares(dias_descanso, defs_descanso)
                g_i = _grados_triangulares(intensidad_val, defs_intensidad)

                df_grados = pd.DataFrame(
                    [
                        {"Variable": "Fatiga", "Etiqueta": k, "Grado": v}
                        for k, v in g_f.items()
                    ]
                    + [
                        {"Variable": "Energía", "Etiqueta": k, "Grado": v}
                        for k, v in g_e.items()
                    ]
                    + [
                        {"Variable": "Descanso", "Etiqueta": k, "Grado": v}
                        for k, v in g_d.items()
                    ]
                    + [
                        {"Variable": "Intensidad", "Etiqueta": k, "Grado": v}
                        for k, v in g_i.items()
                    ]
                )

                fig_grados = go.Figure()
                for var in df_grados["Variable"].unique():
                    data_var = df_grados[df_grados["Variable"] == var]
                    fig_grados.add_trace(
                        go.Bar(
                            x=[f"{var}:{x}" for x in data_var["Etiqueta"]],
                            y=data_var["Grado"],
                            name=var,
                        )
                    )

                fig_grados.update_layout(
                    title="Grados de pertenencia activados",
                    yaxis=dict(range=[0, 1.05]),
                    yaxis_title="Grado",
                    xaxis_title="Etiquetas difusas",
                    barmode="group",
                    height=320,
                    margin=dict(t=45, b=30),
                )
                st.plotly_chart(fig_grados, width="stretch")

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

            rutina_prev = resultado["paso3_rutina"]["rutina"]
            df_prev = pd.DataFrame(rutina_prev)

            cg1, cg2 = st.columns(2)
            with cg1:
                fig_int = go.Figure(
                    go.Bar(
                        x=df_prev["dia"],
                        y=df_prev["intensidad_nivel"],
                        marker=dict(color="#ef476f"),
                        text=df_prev["intensidad_nivel"],
                        textposition="outside",
                    )
                )
                fig_int.update_layout(
                    title="Intensidad por día (rutina final)",
                    xaxis_title="Día",
                    yaxis_title="Intensidad",
                    yaxis=dict(range=[0, 10]),
                    height=280,
                    margin=dict(t=45, b=20),
                )
                st.plotly_chart(fig_int, width="stretch")

            with cg2:
                dist = df_prev["grupo_muscular"].value_counts().reset_index()
                dist.columns = ["grupo", "dias"]
                fig_dist = go.Figure(
                    go.Pie(
                        labels=dist["grupo"],
                        values=dist["dias"],
                        hole=0.45,
                    )
                )
                fig_dist.update_layout(
                    title="Distribución de grupos musculares",
                    height=280,
                    margin=dict(t=45, b=20),
                )
                st.plotly_chart(fig_dist, width="stretch")

        # ── Rutina final ──────────────────────────────────────────────────────
        st.subheader("Rutina Semanal Generada")
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
