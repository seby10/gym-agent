"""
Genera historial.csv — dataset simulado de sesiones de entrenamiento.

Patrones intencionales embebidos (fisiológicamente justificados):
  P1.  piernas + alta        -> fatiga_post muy alta (7-10)
         Mayor masa muscular -> mayor demanda metabólica y neural.
  P2.  espalda + alta        -> fatiga_post alta (6-9)
         Cadena posterior (peso muerto, remo) = alta demanda del SNC.
  P3.  brazos/core + baja   -> fatiga_post baja (1-3)
         Músculos pequeños/aislados a baja intensidad = mínimo impacto sistémico.
  P4.  4+ días consecutivos  -> fatiga_post alta (6-10)
         Fatiga acumulada por sobreentrenamiento sin recuperación.
  P5.  volumen + alta        -> rendimiento alto
         Entrenamiento de hipertrofia a alta intensidad es el estímulo óptimo.
  P6.  resistencia + media   -> rendimiento alto
         La zona aeróbica moderada (60-75% FC) es sostenible y productiva.
  P7.  definicion + media    -> rendimiento alto
         Intensidad moderada en déficit calórico es sostenible sin catabolismo.
  P8.  dias_consecutivos=1 + alta -> rendimiento alto
         Atleta descansado (fosfocretina y glucógeno repuestos) = rendimiento pico.
  P9.  4+ días consecutivos  -> rendimiento bajo
         Fatiga acumulada degrada el rendimiento (sobreentrenamiento).
  P10. alta intensidad + fatiga_post_alta -> rendimiento bajo
         Sesión agotadora ya fatigada = rendimiento comprometido.
"""

import random
import pandas as pd

random.seed(42)

USUARIOS = [f"u{i:03d}" for i in range(1, 21)]
DIAS_SEMANA = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
GRUPOS = ["pecho", "espalda", "piernas", "hombros", "brazos", "core"]
INTENSIDADES = ["baja", "media", "alta"]
OBJETIVOS = ["volumen", "definicion", "resistencia"]


def generar_sesion(usuario_id: str, dias_consecutivos: int, objetivo: str) -> dict:
    dia = random.choice(DIAS_SEMANA)
    grupo = random.choice(GRUPOS)
    intensidad = random.choice(INTENSIDADES)
    duracion = random.randint(30, 90)

    # ── Fatiga post-sesión ────────────────────────────────────────────────────
    # P1: Piernas + alta -> fatiga muy alta
    #     Piernas concentran ~50% de la masa muscular corporal; ejercicios como
    #     sentadilla y peso muerto rumano generan la mayor demanda metabólica.
    if grupo == "piernas" and intensidad == "alta":
        fatiga_post = random.randint(7, 10)

    # P2: Espalda + alta -> fatiga alta
    #     Movimientos de tirón pesado (peso muerto, dominadas, remo) activan la
    #     cadena posterior completa con alta demanda del sistema nervioso central.
    elif grupo == "espalda" and intensidad == "alta":
        fatiga_post = random.randint(6, 9)

    # P3: Brazos/core + baja -> fatiga baja
    #     Grupos musculares pequeños o aislados (bíceps, tríceps, abdominales)
    #     a baja intensidad producen mínimo estrés sistémico y metabólico.
    elif grupo in ["brazos", "core"] and intensidad == "baja":
        fatiga_post = random.randint(1, 3)

    # P4: 4+ días consecutivos -> fatiga acumulada
    #     Sin días de descanso suficientes, la fatiga se acumula progresivamente
    #     afectando al sistema nervioso central y la recuperación muscular.
    elif dias_consecutivos >= 4:
        fatiga_post = random.randint(6, 10)

    # P11: Definicion + alta -> fatiga_post media (sin sobreentrenamiento)
    #      En fase de definicion, el entrenamiento de alta intensidad (HIIT,
    #      circuitos pesados) produce fatiga moderada gestionable; el déficit
    #      calórico limita ligeramente la recuperación pero no la impide.
    elif objetivo == "definicion" and intensidad == "alta" and dias_consecutivos < 4:
        fatiga_post = random.randint(4, 6)  # cap at 6 => always fatiga_post_media

    else:
        fatiga_post = random.randint(1, 6)

    # ── Rendimiento ───────────────────────────────────────────────────────────
    # P5: Volumen + alta (sin sobreentrenamiento) -> rendimiento alto
    #     La sobrecarga progresiva a alta intensidad es el estímulo óptimo para
    #     hipertrofia; el rendimiento es alto cuando no hay fatiga acumulada.
    if objetivo == "volumen" and intensidad == "alta" and dias_consecutivos < 4:
        rendimiento = random.choices(["bajo", "normal", "alto"], weights=[5, 20, 75])[0]

    # P6: Resistencia + media -> rendimiento alto
    #     La zona aeróbica moderada (~60-75% FCmax) permite sesiones largas y
    #     sostenidas con alto rendimiento, evitando el umbral anaeróbico.
    elif objetivo == "resistencia" and intensidad == "media":
        rendimiento = random.choices(["bajo", "normal", "alto"], weights=[5, 30, 65])[0]

    # P7: Definicion + media -> rendimiento alto
    #     En fase de déficit calórico, la intensidad moderada evita el catabolismo
    #     muscular excesivo y es sostenible manteniendo buen rendimiento.
    elif objetivo == "definicion" and intensidad == "media":
        rendimiento = random.choices(["bajo", "normal", "alto"], weights=[5, 25, 70])[0]

    # P8: Fresco (1 día consecutivo) + alta -> rendimiento alto
    #     Con reservas de fosfocretatina y glucógeno repuestas y sistema nervioso
    #     recuperado, el atleta alcanza su rendimiento pico a alta intensidad.
    #     Pesos ajustados a 95% para garantizar confianza >= 0.70 en Apriori.
    elif dias_consecutivos == 1 and intensidad == "alta":
        rendimiento = random.choices(["bajo", "normal", "alto"], weights=[1, 4, 95])[0]

    # P9: 4+ días consecutivos -> rendimiento bajo (fatiga acumulada)
    elif dias_consecutivos >= 4:
        rendimiento = random.choices(["bajo", "normal", "alto"], weights=[65, 30, 5])[0]

    # P10: Alta intensidad + fatiga post alta -> rendimiento bajo
    elif intensidad == "alta" and fatiga_post >= 7:
        rendimiento = random.choices(["bajo", "normal", "alto"], weights=[55, 35, 10])[
            0
        ]

    else:
        rendimiento = random.choices(["bajo", "normal", "alto"], weights=[20, 50, 30])[
            0
        ]

    return {
        "usuario_id": usuario_id,
        "dia_semana": dia,
        "grupo_muscular": grupo,
        "intensidad": intensidad,
        "duracion_min": duracion,
        "fatiga_post": fatiga_post,
        "rendimiento": rendimiento,
        "dias_consecutivos": dias_consecutivos,
        "objetivo": objetivo,
    }


def generar_dataset(n_filas: int = 500) -> pd.DataFrame:
    filas = []
    for _ in range(n_filas):
        usuario = random.choice(USUARIOS)
        objetivo = random.choice(OBJETIVOS)
        # días consecutivos: mayoría 1-3, minoría 4-7 para activar patrones de fatiga
        dias_cons = random.choices(range(1, 8), weights=[25, 22, 18, 12, 10, 8, 5])[0]
        filas.append(generar_sesion(usuario, dias_cons, objetivo))

    return pd.DataFrame(filas)


if __name__ == "__main__":
    df = generar_dataset(500)
    out_path = "data/historial.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset generado: {len(df)} filas -> {out_path}")

    print("\n-- Distribucion --")
    print("Intensidad:\n", df["intensidad"].value_counts().to_string())
    print("Rendimiento:\n", df["rendimiento"].value_counts().to_string())
    print("Objetivo:\n", df["objetivo"].value_counts().to_string())

    print("\n-- Verificacion de patrones --")
    checks = [
        (
            "P1  piernas+alta  -> fatiga_post >=7",
            (df.grupo_muscular == "piernas") & (df.intensidad == "alta"),
            lambda s: s["fatiga_post"].mean(),
        ),
        (
            "P2  espalda+alta  -> fatiga_post >=6",
            (df.grupo_muscular == "espalda") & (df.intensidad == "alta"),
            lambda s: s["fatiga_post"].mean(),
        ),
        (
            "P3  brazos/core+baja -> fatiga_post <=3",
            (df.grupo_muscular.isin(["brazos", "core"])) & (df.intensidad == "baja"),
            lambda s: s["fatiga_post"].mean(),
        ),
        (
            "P4  4+dias -> fatiga_post >=6",
            df.dias_consecutivos >= 4,
            lambda s: s["fatiga_post"].mean(),
        ),
        (
            "P5  volumen+alta -> rendimiento alto %",
            (df.objetivo == "volumen")
            & (df.intensidad == "alta")
            & (df.dias_consecutivos < 4),
            lambda s: (s["rendimiento"] == "alto").mean() * 100,
        ),
        (
            "P6  resistencia+media -> rendimiento alto %",
            (df.objetivo == "resistencia") & (df.intensidad == "media"),
            lambda s: (s["rendimiento"] == "alto").mean() * 100,
        ),
        (
            "P7  definicion+media -> rendimiento alto %",
            (df.objetivo == "definicion") & (df.intensidad == "media"),
            lambda s: (s["rendimiento"] == "alto").mean() * 100,
        ),
        (
            "P8  dias_cons=1+alta -> rendimiento alto %",
            (df.dias_consecutivos == 1) & (df.intensidad == "alta"),
            lambda s: (s["rendimiento"] == "alto").mean() * 100,
        ),
        (
            "P9  4+dias -> rendimiento bajo %",
            df.dias_consecutivos >= 4,
            lambda s: (s["rendimiento"] == "bajo").mean() * 100,
        ),
    ]
    for desc, mask, fn in checks:
        subset = df[mask]
        val = fn(subset)
        print(f"  {desc:<45} n={mask.sum():>3}  val={val:.1f}")
