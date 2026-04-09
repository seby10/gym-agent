"""
Agente Orquestador — GymAgent
Conecta los 3 módulos en secuencia y produce un resultado con razonamiento explicable.
"""

import pandas as pd

from modules.apriori_rules import extraer_reglas, reglas_aplicables
from modules.fuzzy_state import evaluar_estado
from modules.genetic_routine import evolucionar_rutina


class GymAgent:
    """
    Agente inteligente de entrenamiento. Orquesta los 3 módulos de IA
    y produce una rutina con razonamiento paso a paso.
    """

    def __init__(self, path_historial: str):
        self.df = pd.read_csv(path_historial)
        self.reglas = extraer_reglas(self.df)
        # Sistema difuso se instancia en evaluar_estado (stateless por diseño)

    def razonar(self, perfil_usuario: dict) -> dict:
        """
        Ejecuta el pipeline completo dado un perfil de usuario.

        perfil_usuario = {
            "nombre":           str,
            "nivel":            "principiante" | "intermedio" | "avanzado",
            "objetivo":         "volumen" | "definicion" | "resistencia",
            "fatiga_actual":    float,   # 0-10
            "energia_actual":   float,   # 0-10
            "dias_descanso":    float,   # dias sin entrenar
            "grupo_planificado": str     # grupo muscular que queria trabajar hoy
        }

        Retorna:
        {
            "paso1_reglas":        list[dict],
            "paso2_estado":        dict,
            "paso3_rutina":        dict,
            "recomendacion_final": str
        }
        """
        nombre   = perfil_usuario.get("nombre", "Usuario")
        objetivo = perfil_usuario.get("objetivo", "resistencia")
        nivel    = perfil_usuario.get("nivel", "principiante")
        fatiga   = float(perfil_usuario.get("fatiga_actual", 5))
        energia  = float(perfil_usuario.get("energia_actual", 5))
        descanso = float(perfil_usuario.get("dias_descanso", 1))
        grupo    = perfil_usuario.get("grupo_planificado", "core")

        # ── Paso 1: Reglas de Asociación ──────────────────────────────────────
        contexto_apriori = {
            "grupo_muscular":    grupo,
            "intensidad":        self._intensidad_label(fatiga, energia),
            "objetivo":          objetivo,
            "dias_consecutivos": self._dias_consecutivos(descanso),
        }
        reglas_activas = reglas_aplicables(self.reglas, contexto_apriori)

        # ── Paso 2: Lógica Difusa ─────────────────────────────────────────────
        estado_difuso = evaluar_estado(fatiga, energia, descanso)
        intensidad_max = estado_difuso["intensidad_recomendada"]

        # ── Paso 3: Algoritmo Genético ────────────────────────────────────────
        resultado_ag = evolucionar_rutina(
            objetivo=objetivo,
            intensidad_max=intensidad_max,
            reglas_activas=reglas_activas,
            generaciones=50,
        )

        # ── Recomendación final en texto ──────────────────────────────────────
        recomendacion = self._generar_recomendacion(
            nombre, nivel, objetivo, estado_difuso, reglas_activas, resultado_ag
        )

        return {
            "paso1_reglas":        reglas_activas,
            "paso2_estado":        estado_difuso,
            "paso3_rutina":        resultado_ag,
            "recomendacion_final": recomendacion,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _dias_consecutivos(self, descanso: float) -> int:
        """
        Estima días consecutivos de entrenamiento a partir de días de descanso.
        Más descanso = menos días consecutivos activos.
        """
        if descanso == 0:
            return 5   # sin descanso → riesgo de racha larga
        elif descanso == 1:
            return 3   # descanso reciente → racha moderada
        else:
            return 1   # 2+ días de descanso → sin racha activa

    def _intensidad_label(self, fatiga: float, energia: float) -> str:
        """Aproxima la etiqueta de intensidad actual a partir de fatiga y energía."""
        score = energia - fatiga
        if score >= 4:
            return "alta"
        elif score >= 0:
            return "media"
        return "baja"

    def _generar_recomendacion(self, nombre: str, nivel: str, objetivo: str,
                                estado: dict, reglas: list, rutina: dict) -> str:
        """Construye el texto de recomendación final del agente."""
        nivel_difuso = estado["nivel"]
        intensidad   = estado["intensidad_recomendada"]
        fitness      = rutina["fitness_final"]

        # Apertura personalizada por nivel
        apertura = {
            "principiante": f"{nombre}, como principiante es clave no excederte.",
            "intermedio":   f"{nombre}, tu nivel intermedio te permite una carga equilibrada.",
            "avanzado":     f"{nombre}, tu nivel avanzado soporta rutinas exigentes.",
        }.get(nivel, f"Hola {nombre}.")

        # Estado físico
        estado_texto = {
            "recuperacion": "Tu estado físico actual requiere recuperacion activa — prioriza descanso y movilidad.",
            "moderada":     "Tu estado físico es moderado — entrena con intensidad controlada.",
            "alta":         "Tu estado físico es óptimo — puedes exigirte al máximo hoy.",
        }[nivel_difuso]

        # Alertas de reglas
        alertas = ""
        if reglas:
            top = reglas[0]
            alertas = f" El historial advierte: {top['regla_texto']} (confianza {top['confidence']})."

        # Objetivo
        obj_texto = {
            "volumen":     "La rutina prioriza series altas y cargas pesadas para hipertrofia.",
            "definicion":  "La rutina combina variedad muscular con intensidad moderada-alta.",
            "resistencia": "La rutina busca el acondicionamiento general con series cortas y variedad.",
        }.get(objetivo, "")

        # Calidad de la rutina generada
        if fitness >= 0.8:
            calidad = "El agente encontró una rutina de alta calidad"
        elif fitness >= 0.6:
            calidad = "El agente generó una rutina sólida"
        else:
            calidad = "El agente generó una rutina conservadora adaptada a tu estado"

        return (
            f"{apertura} {estado_texto}{alertas} "
            f"{obj_texto} {calidad} "
            f"(fitness {fitness:.2f}, intensidad máx. {intensidad:.1f}/10)."
        )
