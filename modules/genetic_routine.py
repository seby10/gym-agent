"""
Módulo 3 — Algoritmo Genético para generación de rutina semanal.
Cromosoma: lista de 5 genes (un gen por día de entrenamiento).
Cada gen codificado como [grupo_idx, ejercicios, series, intensidad_nivel].
"""

import random
import numpy as np
from deap import base, creator, tools, algorithms

# ── Constantes ────────────────────────────────────────────────────────────────
GRUPOS = ['pecho', 'espalda', 'piernas', 'hombros', 'brazos', 'core']
DIAS   = ['lunes', 'martes', 'miercoles', 'jueves', 'viernes']

# Rangos de cada atributo del gen (min, max inclusive)
RANGO_GRUPO      = (0, 5)    # índice en GRUPOS
RANGO_EJERCICIOS = (2, 6)
RANGO_SERIES     = (2, 5)
RANGO_INTENSIDAD = (0, 10)

# Atributos de fitness por objetivo
PERFIL_OBJETIVO = {
    'volumen': {
        'series_ideal': (4, 5),
        'ejercicios_ideal': (4, 6),
        'intensidad_ideal': (6, 10),
    },
    'definicion': {
        'series_ideal': (3, 4),
        'ejercicios_ideal': (3, 5),
        'intensidad_ideal': (4, 7),
    },
    'resistencia': {
        'series_ideal': (2, 3),
        'ejercicios_ideal': (2, 4),
        'intensidad_ideal': (3, 6),
    },
}

# Señales de sobreentrenamiento en los consecuentes de reglas Apriori
CONSECUENTES_SOBREENTRENAMIENTO = {
    'fatiga_post_alta', 'rendimiento_bajo', 'dias_consecutivos_4mas'
}


# ── Configuración DEAP (guardada en registro global para evitar re-crear) ─────
def _setup_deap():
    """Registra tipos y operadores en DEAP (idempotente)."""
    if not hasattr(creator, 'FitnessMaxGym'):
        creator.create('FitnessMaxGym', base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'IndividualGym'):
        creator.create('IndividualGym', list, fitness=creator.FitnessMaxGym)

    tb = base.Toolbox()

    def gen_atributo(lo, hi):
        return random.randint(lo, hi)

    def gen_individuo():
        ind = []
        for _ in range(5):   # 5 días
            ind.append(gen_atributo(*RANGO_GRUPO))
            ind.append(gen_atributo(*RANGO_EJERCICIOS))
            ind.append(gen_atributo(*RANGO_SERIES))
            ind.append(gen_atributo(*RANGO_INTENSIDAD))
        return creator.IndividualGym(ind)

    tb.register('individual', gen_individuo)
    tb.register('population', tools.initRepeat, list, tb.individual)
    tb.register('mate',   tools.cxTwoPoint)
    tb.register('select', tools.selTournament, tournsize=3)
    return tb


# ── Decodificación ────────────────────────────────────────────────────────────
def _decodificar(individuo: list) -> list[dict]:
    """Convierte la lista plana de 20 enteros en la lista de 5 genes (dicts)."""
    genes = []
    for i, dia in enumerate(DIAS):
        base_idx = i * 4
        grupo_idx     = int(np.clip(individuo[base_idx],     *RANGO_GRUPO))
        ejercicios    = int(np.clip(individuo[base_idx + 1], *RANGO_EJERCICIOS))
        series        = int(np.clip(individuo[base_idx + 2], *RANGO_SERIES))
        intensidad    = int(np.clip(individuo[base_idx + 3], *RANGO_INTENSIDAD))
        genes.append({
            'dia':              dia,
            'grupo_muscular':   GRUPOS[grupo_idx],
            'ejercicios':       ejercicios,
            'series':           series,
            'intensidad_nivel': intensidad,
        })
    return genes


# ── Función de fitness ────────────────────────────────────────────────────────
def calcular_fitness(individuo: list, objetivo: str,
                     intensidad_max: float, reglas_activas: list) -> tuple[float]:
    """
    Evalúa qué tan buena es una rutina. Retorna tupla (score,) para DEAP.
    Score entre 0.0 y 1.0 (mayor = mejor).

    Criterios positivos:
    + Alineación con objetivo (series, ejercicios, intensidad ideales)
    + Variedad de grupos musculares

    Penalizaciones:
    - Intensidad por encima del límite difuso
    - Grupos musculares repetidos en días consecutivos
    - Reglas activas que indican sobreentrenamiento
    """
    genes = _decodificar(individuo)
    objetivo_key = objetivo if objetivo in PERFIL_OBJETIVO else 'resistencia'
    perfil = PERFIL_OBJETIVO[objetivo_key]

    score_alineacion  = 0.0
    score_variedad    = 0.0
    penalizacion      = 0.0

    grupos_usados = [g['grupo_muscular'] for g in genes]

    for i, gen in enumerate(genes):
        # Alineación con objetivo ──────────────────────────────────────────────
        s_lo, s_hi = perfil['series_ideal']
        e_lo, e_hi = perfil['ejercicios_ideal']
        i_lo, i_hi = perfil['intensidad_ideal']

        alin = 0.0
        if s_lo <= gen['series'] <= s_hi:
            alin += 1.0
        if e_lo <= gen['ejercicios'] <= e_hi:
            alin += 1.0
        if i_lo <= gen['intensidad_nivel'] <= i_hi:
            alin += 1.0
        score_alineacion += alin / 3.0

        # Penalizar intensidad por encima del tope difuso ─────────────────────
        exceso = gen['intensidad_nivel'] - intensidad_max
        if exceso > 0:
            penalizacion += exceso * 0.05   # 0.05 por punto de exceso

        # Penalizar grupo repetido en día consecutivo ─────────────────────────
        if i > 0 and grupos_usados[i] == grupos_usados[i - 1]:
            penalizacion += 0.15

    # Variedad: grupos únicos en los 5 días ───────────────────────────────────
    grupos_unicos = len(set(grupos_usados))
    score_variedad = grupos_unicos / len(GRUPOS)   # máx 1.0 si usa 6 distintos

    # Penalizar si reglas activas señalan sobreentrenamiento ──────────────────
    for regla in reglas_activas:
        consecuente = regla.get('consecuente', '').lower().replace(' ', '_')
        if any(sena in consecuente for sena in CONSECUENTES_SOBREENTRENAMIENTO):
            penalizacion += 0.10

    # Score total ──────────────────────────────────────────────────────────────
    score_alineacion_norm = score_alineacion / 5.0   # promedio sobre 5 días
    total = 0.6 * score_alineacion_norm + 0.4 * score_variedad - penalizacion
    total = float(np.clip(total, 0.0, 1.0))

    return (total,)


# ── Mutación personalizada ────────────────────────────────────────────────────
def _mutar(individuo: list, indpb: float = 0.2) -> tuple:
    """
    Muta aleatoriamente cada atributo del individuo con probabilidad indpb,
    respetando los rangos de cada campo.
    """
    rangos = [RANGO_GRUPO, RANGO_EJERCICIOS, RANGO_SERIES, RANGO_INTENSIDAD] * 5
    for i in range(len(individuo)):
        if random.random() < indpb:
            individuo[i] = random.randint(*rangos[i])
    return (individuo,)


# ── Evolución principal ───────────────────────────────────────────────────────
def evolucionar_rutina(objetivo: str, intensidad_max: float,
                       reglas_activas: list, generaciones: int = 50) -> dict:
    """
    Ejecuta el algoritmo genético y retorna la mejor rutina encontrada.

    Parámetros AG:
    - Población: 100 individuos
    - Generaciones: 50
    - Crossover: cxTwoPoint, prob=0.8
    - Mutación: personalizada, prob=0.2
    - Selección: selTournament, tournsize=3

    Retorna:
    {
        "rutina": list[dict],            # 5 genes decodificados
        "fitness_final": float,
        "historial_fitness": list[float] # mejor fitness por generación
    }
    """
    random.seed(42)

    tb = _setup_deap()

    # Registrar evaluación con los parámetros del usuario
    tb.register('evaluate', calcular_fitness,
                objetivo=objetivo,
                intensidad_max=intensidad_max,
                reglas_activas=reglas_activas)
    tb.register('mutate', _mutar, indpb=0.2)

    # Crear población inicial
    poblacion = tb.population(n=100)

    historial_fitness = []
    hof = tools.HallOfFame(1)   # guarda el mejor individuo visto

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register('max', np.max)

    # Evaluación inicial
    fits = list(map(tb.evaluate, poblacion))
    for ind, fit in zip(poblacion, fits):
        ind.fitness.values = fit

    hof.update(poblacion)
    historial_fitness.append(float(hof[0].fitness.values[0]))

    # Ciclo generacional
    for gen in range(1, generaciones):
        # Selección
        descendientes = tb.select(poblacion, len(poblacion))
        descendientes = list(map(tb.clone, descendientes))

        # Crossover
        for hijo1, hijo2 in zip(descendientes[::2], descendientes[1::2]):
            if random.random() < 0.8:
                tb.mate(hijo1, hijo2)
                del hijo1.fitness.values
                del hijo2.fitness.values

        # Mutación
        for mutante in descendientes:
            if random.random() < 0.2:
                tb.mutate(mutante)
                del mutante.fitness.values

        # Re-evaluar los que cambiaron
        invalidos = [ind for ind in descendientes if not ind.fitness.valid]
        fits = list(map(tb.evaluate, invalidos))
        for ind, fit in zip(invalidos, fits):
            ind.fitness.values = fit

        poblacion[:] = descendientes
        hof.update(poblacion)
        historial_fitness.append(float(hof[0].fitness.values[0]))

    mejor = hof[0]
    rutina = _decodificar(mejor)

    return {
        'rutina':            rutina,
        'fitness_final':     round(float(mejor.fitness.values[0]), 4),
        'historial_fitness': historial_fitness,
    }
