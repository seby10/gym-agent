"""
Módulo 2 — Lógica Difusa para evaluación del estado físico del usuario.
Inputs: fatiga [0-10], energia [0-10], dias_descanso [0-7]
Output: intensidad_recomendada [0-10] + categoría + texto de razonamiento

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFERENCIAS CIENTÍFICAS — Justificación de funciones de membresía
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1] Borg, G. (1990). Psychophysical scaling with applications in physical
    work and the perception of exertion. Scand J Work Environ Health,
    16(Suppl 1), 55-58. — Escala CR10 (0-10): ligero 1-3, moderado 4-6,
    alto ≥7. Fundamenta los umbrales de fatiga y salida de intensidad.

[2] Micklewright, D. et al. (2017). Development and Validity of the
    Rating-of-Fatigue Scale. Sports Medicine, 47(11), 2375-2393.
    https://doi.org/10.1007/s40279-017-0711-5
    — Valida escala ROF (0-10) para fatiga percibida; alta correlación
    con FC, lactato y VO2. Umbral "fatiga alta" ≥ 7, no 6.

[3] American College of Sports Medicine (ACSM, 2024). Guidelines for
    Exercise Testing and Prescription, 11th ed.
    — Clasifica intensidad: ligera <3 METs, moderada 3-6, vigorosa ≥6.
    Mapeado a escala subjetiva 0-10 coincide con los sets de intensidad
    de salida: muy_baja (0-3), moderada (3-6), alta (6-10).

[4] Novatchkov, H. & Baca, A. (2013). Fuzzy Logic in Sports: A Review
    and an Illustrative Case Study in Strength Training. IJCA, 71(6).
    — Justifica funciones triangulares (trimf) en variables fisiológicas
    deportivas por su simplicidad y adecuación a percepciones subjetivas.

[5] Abdulaziz, M. et al. (2024). Optimization of Sports Training Intensity
    Based on Fuzzy Control (FC-TOF). Int. J. Computational Intelligence
    Systems, Springer Nature. https://doi.org/10.1007/s44196-026-01158-6
    — Sistema difuso con inputs fatiga, RPE y volumen; valida inputs y
    estructura de reglas similares a los adoptados en este módulo.

[6] UC Health / NASM (2023). Rest and Recovery for Athletes.
    https://www.uchealth.org/today/rest-and-recovery-for-athletes
    — 24-48 h (1-2 días) entre sesiones = recuperación normal;
    ≥3 días = recuperación adecuada. Fundamento de dias_descanso.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ── Definiciones de funciones de membresía ────────────────────────────────────
# Fuente única de verdad: usadas tanto en crear_sistema_difuso() como en app.py
# para visualización. Modificar aquí actualiza ambos automáticamente.
# trapmf → 4 puntos (sets de borde); trimf → 3 puntos (sets centrales) [Ref.4]
DEFS_FATIGA = {
    "baja": [0, 0, 2, 4],  # plano 0-2 [Ref.4]
    "media": [2, 5, 8],  # peak en 5
    "alta": [7, 9, 10, 10],  # plano 9-10, umbral ≥7 [Ref.2]
}
DEFS_ENERGIA = {
    "baja": [0, 0, 2, 4],  # plano 0-2
    "media": [2, 5, 8],  # peak en 5
    "alta": [7, 9, 10, 10],  # plano 9-10 [Ref.5]
}
DEFS_DESCANSO = {
    "poco": [0, 0, 0.5, 2],  # plano en 0 [Ref.6]
    "normal": [1, 2, 4],  # peak 2d (24-48h) [Ref.6]
    "mucho": [3, 5, 7, 7],  # plano 5-7 [Ref.6]
}
DEFS_INTENSIDAD = {
    "muy_baja": [0, 0, 1, 3],  # plano 0-1
    "baja": [1, 3, 5],
    "moderada": [3, 5, 7],
    "alta": [6, 8, 10],
    "muy_alta": [8, 9, 10, 10],  # plano 9-10
}


def crear_sistema_difuso():
    """
    Define universos, funciones de membresía y reglas difusas.
    Retorna el sistema de control listo para simular (ControlSystemSimulation).
    """
    # ── Variables de entrada ──────────────────────────────────────────────────
    # Se define universos de las variables [0, 0.5, 1, ..., 10] para fatiga y energía, y [0, 0.5, ..., 7] para descanso
    # ctrl.Antecedent() — viene de skfuzzy control. Crea una variable de entrada del sistema difuso. Recibe el universo y un nombre interno. Fatiga, energía y descanso son entradas porque son los datos que el usuario ingresa.
    fatiga = ctrl.Antecedent(np.arange(0, 11, 0.5), "fatiga")
    energia = ctrl.Antecedent(np.arange(0, 11, 0.5), "energia")
    descanso = ctrl.Antecedent(np.arange(0, 8, 0.5), "dias_descanso")

    # ── Variable de salida ────────────────────────────────────────────────────
    # ctrl.Consequent() — viene de skfuzzy control. Crea una variable de salida. La intensidad es salida porque es lo que el sistema calcula.
    intensidad = ctrl.Consequent(np.arange(0, 11, 0.5), "intensidad")

    # ── Funciones de membresía — usan DEFS_* para mantener sincronía con app.py ─
    def _aplicar(variable, defs):
        for etiqueta, params in defs.items():
            if len(params) == 4:
                variable[etiqueta] = fuzz.trapmf(variable.universe, params)
            else:
                variable[etiqueta] = fuzz.trimf(variable.universe, params)

    _aplicar(fatiga, DEFS_FATIGA)
    _aplicar(energia, DEFS_ENERGIA)
    _aplicar(descanso, DEFS_DESCANSO)
    _aplicar(intensidad, DEFS_INTENSIDAD)

    # ── Reglas difusas (≥ 5 requeridas) ──────────────────────────────────────
    # Regla 1: fatiga alta + energía baja → recuperación total
    r1 = ctrl.Rule(fatiga["alta"] & energia["baja"], intensidad["muy_baja"])

    # Regla 2: fatiga alta + poco descanso → intensidad baja
    r2 = ctrl.Rule(fatiga["alta"] & descanso["poco"], intensidad["baja"])

    # Regla 3: fatiga media + energía media → intensidad moderada
    r3 = ctrl.Rule(fatiga["media"] & energia["media"], intensidad["moderada"])

    # Regla 4: fatiga baja + energía alta → intensidad alta
    r4 = ctrl.Rule(fatiga["baja"] & energia["alta"], intensidad["alta"])

    # Regla 5: fatiga baja + mucho descanso → máximo rendimiento
    r5 = ctrl.Rule(fatiga["baja"] & descanso["mucho"], intensidad["muy_alta"])

    # Regla 6: fatiga media + energía alta → moderada-alta
    r6 = ctrl.Rule(fatiga["media"] & energia["alta"], intensidad["alta"])

    # Regla 7: fatiga alta + mucho descanso → moderada (recuperado parcialmente)
    r7 = ctrl.Rule(fatiga["alta"] & descanso["mucho"], intensidad["moderada"])

    # Regla 8: fatiga baja + energía baja → moderada (descansado pero sin energía)
    r8 = ctrl.Rule(fatiga["baja"] & energia["baja"], intensidad["moderada"])

    # ctrl.ControlSystem() — viene de skfuzzy control. Agrupa todas las reglas en un sistema cohesivo.
    sistema = ctrl.ControlSystem([r1, r2, r3, r4, r5, r6, r7, r8])
    # ctrl.ControlSystemSimulation() — viene de skfuzzy control. Envuelve el sistema y lo deja listo para recibir valores y calcular resultados. Es lo que se devuelve para usar en la siguiente función.
    return ctrl.ControlSystemSimulation(sistema)


def evaluar_estado(fatiga: float, energia: float, dias_descanso: float) -> dict:
    """
    Simula el sistema difuso y retorna resultado con razonamiento en texto.

    Retorna:
    {
        "intensidad_recomendada": float,   # valor entre 0 y 10
        "nivel": str,                      # "recuperacion" | "moderada" | "alta"
        "razonamiento": str                # explicación para mostrar al usuario
    }
    """
    # Clamping de entradas al rango válido
    # np.clip() — método de numpy. Fuerza un valor a quedarse dentro de un rango. Si alguien ingresa fatiga = 15, np.clip(15, 0, 10) devuelve 10. Si ingresa -3, devuelve 0. Es una protección de seguridad.
    fatiga = float(np.clip(fatiga, 0, 10))
    energia = float(np.clip(energia, 0, 10))
    dias_descanso = float(np.clip(dias_descanso, 0, 7))

    sim = crear_sistema_difuso()
    # sim.input[] — es la forma de skfuzzy de alimentar valores al sistema. Usas el nombre que le diste a la variable al crearla con Antecedent.
    sim.input["fatiga"] = fatiga
    sim.input["energia"] = energia
    sim.input["dias_descanso"] = dias_descanso

    try:
        # sim.compute() — método de skfuzzy. Ejecuta todo el proceso difuso internamente en tres pasos que suceden automáticamente:
        sim.compute()
        intensidad_val = float(
            sim.output["intensidad"]
        )  # Nuemro real despues de la defuzzificación
    except Exception:
        # Fallback si el sistema no converge (valores extremos)
        intensidad_val = _fallback_intensidad(fatiga, energia, dias_descanso)

    # Categoría según el valor de salida
    if intensidad_val < 3.5:
        nivel = "recuperacion"
    elif intensidad_val < 6.5:
        nivel = "moderada"
    else:
        nivel = "alta"

    razonamiento = _generar_razonamiento(
        fatiga, energia, dias_descanso, intensidad_val, nivel
    )

    return {
        "intensidad_recomendada": round(intensidad_val, 2),
        "nivel": nivel,
        "razonamiento": razonamiento,
    }


def _fallback_intensidad(fatiga: float, energia: float, descanso: float) -> float:
    """Cálculo simple cuando el sistema difuso no converge."""
    # Es un cálculo manual simple que se usa solo cuando el sistema difuso falla. La fórmula tiene sentido intuitivo: más energía sube el score, más fatiga lo baja, más descanso lo sube. np.clip garantiza que el resultado quede entre 0 y 10.
    score = (energia - fatiga + descanso) / 2.0
    return float(np.clip(score, 0, 10))


def _generar_razonamiento(
    fatiga: float, energia: float, descanso: float, intensidad: float, nivel: str
) -> str:
    """Genera texto explicativo del razonamiento difuso."""
    lineas = []

    # Análisis de fatiga
    if fatiga >= 7:
        lineas.append(f"Fatiga elevada ({fatiga}/10): el cuerpo necesita recuperación.")
    elif fatiga >= 4:
        lineas.append(
            f"Fatiga moderada ({fatiga}/10): hay margen para entrenar con cuidado."
        )
    else:
        lineas.append(
            f"Fatiga baja ({fatiga}/10): el cuerpo está listo para el esfuerzo."
        )

    # Análisis de energía
    if energia >= 7:
        lineas.append(f"Energía alta ({energia}/10): excelente para sesión intensa.")
    elif energia >= 4:
        lineas.append(f"Energía media ({energia}/10): sesión equilibrada recomendada.")
    else:
        lineas.append(
            f"Energía baja ({energia}/10): limita la capacidad de rendimiento."
        )

    # Análisis de descanso
    if descanso == 0:
        lineas.append("Sin días de descanso recientes: riesgo de sobreentrenamiento.")
    elif descanso >= 4:
        lineas.append(
            f"{int(descanso)} días de descanso: músculo y sistema nervioso recuperados."
        )
    else:
        lineas.append(f"{int(descanso)} día(s) de descanso: recuperación parcial.")

    # Conclusión
    nivel_texto = {
        "recuperacion": "recuperación activa (ejercicio suave o descanso activo)",
        "moderada": "entrenamiento moderado (intensidad controlada)",
        "alta": "entrenamiento de alta intensidad (sesión completa)",
    }[nivel]

    lineas.append(f"→ Intensidad recomendada: {intensidad:.1f}/10 — {nivel_texto}.")

    return " | ".join(lineas)
