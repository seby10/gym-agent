"""
Módulo 1 — Reglas de Asociación con Apriori
Extrae patrones del historial de entrenamiento y los aplica al contexto actual del usuario.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def _discretizar_fila(row: pd.Series) -> list[str]:
    """Convierte una fila del CSV en una lista de ítems para Apriori."""
    items = []

    # Grupo muscular
    items.append(f"grupo_{row['grupo_muscular']}")

    # Intensidad
    items.append(f"intensidad_{row['intensidad']}")

    # Objetivo
    items.append(f"objetivo_{row['objetivo']}")

    # Fatiga post (discretizada)
    f = row['fatiga_post']
    if f <= 3:
        items.append("fatiga_post_baja")
    elif f <= 6:
        items.append("fatiga_post_media")
    else:
        items.append("fatiga_post_alta")

    # Rendimiento
    items.append(f"rendimiento_{row['rendimiento']}")

    # Días consecutivos
    d = row['dias_consecutivos']
    if d >= 4:
        items.append("dias_consecutivos_4mas")
    elif d >= 2:
        items.append("dias_consecutivos_2a3")
    else:
        items.append("dias_consecutivos_1")

    return items


def extraer_reglas(df: pd.DataFrame, min_support: float = 0.1,
                   min_confidence: float = 0.7) -> pd.DataFrame:
    """
    Convierte el historial en transacciones y extrae reglas de asociación.
    Retorna DataFrame con columnas: antecedents, consequents, confidence, lift.
    """
    transacciones = df.apply(_discretizar_fila, axis=1).tolist()

    te = TransactionEncoder()
    te_array = te.fit_transform(transacciones)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    itemsets_frecuentes = apriori(df_encoded, min_support=min_support,
                                  use_colnames=True)

    if itemsets_frecuentes.empty:
        return pd.DataFrame(columns=['antecedents', 'consequents',
                                     'confidence', 'lift'])

    reglas = association_rules(itemsets_frecuentes, metric="confidence",
                               min_threshold=min_confidence)

    return reglas[['antecedents', 'consequents', 'confidence', 'lift']].reset_index(drop=True)


def reglas_aplicables(reglas: pd.DataFrame, contexto: dict) -> list[dict]:
    """
    Dado el contexto actual del usuario, retorna las reglas cuyo antecedente
    coincide con algún ítem del contexto.

    contexto esperado:
    {
        "grupo_muscular": str,     # pecho, espalda, piernas, ...
        "intensidad": str,         # baja, media, alta
        "objetivo": str,           # volumen, definicion, resistencia
        "dias_consecutivos": int   # número de días consecutivos entrenando
    }

    Retorna lista de dicts: [{"regla_texto": str, "consecuente": str, "confidence": float}]
    """
    if reglas.empty:
        return []

    # Construir ítems del contexto actual
    items_contexto = set()
    if "grupo_muscular" in contexto:
        items_contexto.add(f"grupo_{contexto['grupo_muscular']}")
    if "intensidad" in contexto:
        items_contexto.add(f"intensidad_{contexto['intensidad']}")
    if "objetivo" in contexto:
        items_contexto.add(f"objetivo_{contexto['objetivo']}")
    if "dias_consecutivos" in contexto:
        d = contexto["dias_consecutivos"]
        if d >= 4:
            items_contexto.add("dias_consecutivos_4mas")
        elif d >= 2:
            items_contexto.add("dias_consecutivos_2a3")
        else:
            items_contexto.add("dias_consecutivos_1")

    resultado = []
    for _, row in reglas.iterrows():
        antecedente = set(row['antecedents'])
        # La regla aplica si su antecedente es subconjunto del contexto
        if antecedente.issubset(items_contexto):
            consecuente = ", ".join(sorted(row['consequents']))
            antecedente_texto = " + ".join(sorted(antecedente))
            resultado.append({
                "regla_texto": f"{antecedente_texto} → {consecuente}",
                "consecuente": consecuente,
                "confidence": round(float(row['confidence']), 3)
            })

    # Ordenar por confianza descendente
    resultado.sort(key=lambda x: x['confidence'], reverse=True)
    return resultado
