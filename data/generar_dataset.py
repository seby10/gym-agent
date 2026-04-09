"""
Genera historial.csv — dataset simulado de sesiones de entrenamiento.
Incluye patrones intencionales para que Apriori los descubra:
  - piernas + alta intensidad  → fatiga_post alta (7–10)
  - 4+ días consecutivos       → rendimiento bajo
  - resistencia + media        → rendimiento alto
"""

import random
import pandas as pd

random.seed(42)

USUARIOS      = [f"u{i:03d}" for i in range(1, 21)]          # 20 usuarios
DIAS_SEMANA   = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
GRUPOS        = ["pecho", "espalda", "piernas", "hombros", "brazos", "core"]
INTENSIDADES  = ["baja", "media", "alta"]
OBJETIVOS     = ["volumen", "definicion", "resistencia"]

def generar_sesion(usuario_id: str, dias_consecutivos: int, objetivo: str) -> dict:
    dia         = random.choice(DIAS_SEMANA)
    grupo       = random.choice(GRUPOS)
    intensidad  = random.choice(INTENSIDADES)
    duracion    = random.randint(30, 90)

    # --- Patrones intencionales ---

    # 1. piernas + alta → fatiga_post alta
    if grupo == "piernas" and intensidad == "alta":
        fatiga_post = random.randint(7, 10)
    # 2. 4+ días consecutivos → fatiga post alta también
    elif dias_consecutivos >= 4:
        fatiga_post = random.randint(6, 10)
    else:
        fatiga_post = random.randint(1, 6)

    # 3. resistencia + media → rendimiento alto
    if objetivo == "resistencia" and intensidad == "media":
        rendimiento = random.choices(["bajo", "normal", "alto"], weights=[5, 30, 65])[0]
    # 4. 4+ días consecutivos → rendimiento bajo
    elif dias_consecutivos >= 4:
        rendimiento = random.choices(["bajo", "normal", "alto"], weights=[65, 30, 5])[0]
    # 5. alta intensidad + fatiga alta → rendimiento bajo
    elif intensidad == "alta" and fatiga_post >= 7:
        rendimiento = random.choices(["bajo", "normal", "alto"], weights=[55, 35, 10])[0]
    else:
        rendimiento = random.choices(["bajo", "normal", "alto"], weights=[20, 50, 30])[0]

    return {
        "usuario_id":        usuario_id,
        "dia_semana":        dia,
        "grupo_muscular":    grupo,
        "intensidad":        intensidad,
        "duracion_min":      duracion,
        "fatiga_post":       fatiga_post,
        "rendimiento":       rendimiento,
        "dias_consecutivos": dias_consecutivos,
        "objetivo":          objetivo,
    }


def generar_dataset(n_filas: int = 300) -> pd.DataFrame:
    filas = []
    for _ in range(n_filas):
        usuario    = random.choice(USUARIOS)
        objetivo   = random.choice(OBJETIVOS)
        # días consecutivos: mayoría 1–3, algunos 4–7 para activar el patrón
        dias_cons  = random.choices(range(1, 8), weights=[25, 22, 18, 12, 10, 8, 5])[0]
        filas.append(generar_sesion(usuario, dias_cons, objetivo))

    return pd.DataFrame(filas)


if __name__ == "__main__":
    df = generar_dataset(300)
    out_path = "data/historial.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset generado: {len(df)} filas -> {out_path}")
    print("\nDistribución de intensidad:")
    print(df["intensidad"].value_counts())
    print("\nDistribución de rendimiento:")
    print(df["rendimiento"].value_counts())
    print("\nFatiga promedio (piernas + alta):")
    mask = (df["grupo_muscular"] == "piernas") & (df["intensidad"] == "alta")
    print(f"  {df.loc[mask, 'fatiga_post'].mean():.2f}  (esperado >= 7)")
    print("\nRendimiento bajo con 4+ dias consecutivos:")
    mask2 = df["dias_consecutivos"] >= 4
    pct = (df.loc[mask2, "rendimiento"] == "bajo").mean() * 100
    print(f"  {pct:.1f}%  (esperado > 50%)")
