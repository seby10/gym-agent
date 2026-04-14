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
    f = row["fatiga_post"]
    if f <= 3:
        items.append("fatiga_post_baja")
    elif f <= 6:
        items.append("fatiga_post_media")
    else:
        items.append("fatiga_post_alta")

    # Rendimiento
    items.append(f"rendimiento_{row['rendimiento']}")

    # Días consecutivos
    d = row["dias_consecutivos"]
    if d >= 4:
        items.append("dias_consecutivos_4mas")
    elif d >= 2:
        items.append("dias_consecutivos_2a3")
    else:
        items.append("dias_consecutivos_1")

    return items


def extraer_reglas(
    df: pd.DataFrame, min_support: float = 0.04, min_confidence: float = 0.7
) -> pd.DataFrame:
    """
    Convierte el historial en transacciones y extrae reglas de asociación.
    Retorna DataFrame con columnas: antecedents, consequents, confidence, lift.
    """
    transacciones = df.apply(_discretizar_fila, axis=1).tolist()

    te = TransactionEncoder()
    te_array = te.fit_transform(transacciones)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    itemsets_frecuentes = apriori(
        df_encoded, min_support=min_support, use_colnames=True
    )

    if itemsets_frecuentes.empty:
        return pd.DataFrame(
            columns=["antecedents", "consequents", "confidence", "lift"]
        )

    reglas = association_rules(
        itemsets_frecuentes, metric="confidence", min_threshold=min_confidence
    )

    return reglas[["antecedents", "consequents", "confidence", "lift"]].reset_index(
        drop=True
    )


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

    # Prefijos de ítems que pueden aparecer en el contexto del usuario.
    # Solo las reglas con antecedentes formados exclusivamente por estos
    # prefijos son relevantes: las que incluyen fatiga_post o rendimiento
    # en el antecedente nunca podrán ser activadas por inputs del usuario.
    _PREFIJOS_CONTEXTO = ("grupo_", "intensidad_", "objetivo_", "dias_consecutivos_")

    def _es_antecedente_contextual(antecedente: set) -> bool:
        return all(
            any(item.startswith(p) for p in _PREFIJOS_CONTEXTO) for item in antecedente
        )

    def _construir_entrada(antecedente: set, row) -> dict:
        consecuente = ", ".join(sorted(row["consequents"]))
        antecedente_texto = " + ".join(sorted(antecedente))
        return {
            "regla_texto": f"{antecedente_texto} -> {consecuente}",
            "consecuente": consecuente,
            "confidence": round(float(row["confidence"]), 3),
        }

    resultado = []
    for _, row in reglas.iterrows():
        antecedente = set(row["antecedents"])
        # Coincidencia estricta: antecedente completo ⊆ contexto
        if antecedente.issubset(items_contexto):
            resultado.append(_construir_entrada(antecedente, row))

    if resultado:
        resultado.sort(key=lambda x: x["confidence"], reverse=True)
        return resultado

    # Fallback — coincidencia parcial sin contradicción:
    # Si no hubo coincidencia estricta, buscar reglas cuyo antecedente:
    #   a) esté formado solo por ítems de tipo contexto (no outcomes históricos)
    #   b) comparta al menos 1 ítem con el contexto actual
    #   c) NO contenga ítems que contradigan el contexto: p.ej. si el contexto
    #      tiene "dias_consecutivos_1", descartar reglas con "dias_consecutivos_4mas"
    #      porque pertenecen al mismo tipo pero representan condiciones opuestas.
    def _contradice_contexto(antecedente: set) -> bool:
        for item in antecedente:
            if item in items_contexto:
                continue  # coincide, no contradice
            # ítem no está en contexto — verificar si su tipo ya está representado
            for prefijo in _PREFIJOS_CONTEXTO:
                if item.startswith(prefijo):
                    if any(c.startswith(prefijo) and c != item for c in items_contexto):
                        return True  # mismo tipo, valor diferente = contradicción
        return False

    for _, row in reglas.iterrows():
        antecedente = set(row["antecedents"])
        if (
            _es_antecedente_contextual(antecedente)
            and antecedente & items_contexto
            and not _contradice_contexto(antecedente)
        ):
            resultado.append(_construir_entrada(antecedente, row))

    resultado.sort(key=lambda x: x["confidence"], reverse=True)
    return resultado
