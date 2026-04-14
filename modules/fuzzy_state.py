"""
Módulo 2 — Lógica Difusa para evaluación del estado físico del usuario.
Inputs: fatiga [0-10], energia [0-10], dias_descanso [0-7]
Output: intensidad_recomendada [0-10] + categoría + texto de razonamiento
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


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

    # ── Funciones de membresía — Fatiga ───────────────────────────────────────
    # Aqui se arma la función trimf, todas son triangulares
    fatiga["baja"] = fuzz.trimf(fatiga.universe, [0, 0, 4])
    fatiga["media"] = fuzz.trimf(fatiga.universe, [2, 5, 8])
    fatiga["alta"] = fuzz.trimf(fatiga.universe, [6, 10, 10])

    # ── Funciones de membresía — Energía ──────────────────────────────────────
    energia["baja"] = fuzz.trimf(energia.universe, [0, 0, 4])
    energia["media"] = fuzz.trimf(energia.universe, [2, 5, 8])
    energia["alta"] = fuzz.trimf(energia.universe, [6, 10, 10])

    # ── Funciones de membresía — Días de descanso ─────────────────────────────
    descanso["poco"] = fuzz.trimf(descanso.universe, [0, 0, 2])
    descanso["normal"] = fuzz.trimf(descanso.universe, [1, 3, 5])
    descanso["mucho"] = fuzz.trimf(descanso.universe, [4, 7, 7])

    # ── Funciones de membresía — Intensidad recomendada ───────────────────────
    intensidad["muy_baja"] = fuzz.trimf(intensidad.universe, [0, 0, 3])
    intensidad["baja"] = fuzz.trimf(intensidad.universe, [1, 3, 5])
    intensidad["moderada"] = fuzz.trimf(intensidad.universe, [3, 5, 7])
    intensidad["alta"] = fuzz.trimf(intensidad.universe, [6, 8, 10])
    intensidad["muy_alta"] = fuzz.trimf(intensidad.universe, [8, 10, 10])

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
