"""Proyecciones semánticas sobre el espacio Word2Vec (proyecto LuxuryHotels)."""
import numpy as np


def eje_semantico(wv, polo_positivo, polo_negativo):
    """Dirección unitaria que va del polo negativo al positivo: promedio(positivo) − promedio(negativo).
    Las palabras ancla que no están en el vocabulario del modelo se ignoran."""
    pos = [p for p in polo_positivo if p in wv]
    neg = [p for p in polo_negativo if p in wv]
    eje = wv[pos].mean(axis=0) - wv[neg].mean(axis=0)
    return eje / np.linalg.norm(eje)


def proyeccion_global(matriz, eje):
    """Similitud coseno entre el vector completo de cada reseña y el eje."""
    normas = np.linalg.norm(matriz, axis=1, keepdims=True).clip(min=1e-9)
    return (matriz / normas) @ eje


def proyeccion_aspecto(texto, wv, terminos_aspecto, eje, ventana=4):
    """Similitud coseno entre el eje y el promedio de las palabras que rodean (± ventana)
    a los términos del aspecto. Devuelve NaN si la reseña no menciona el aspecto."""
    palabras = texto.split()
    posiciones = [i for i, p in enumerate(palabras) if p in terminos_aspecto]
    if not posiciones:
        return np.nan
    contexto = set()
    for i in posiciones:
        contexto.update(range(max(0, i - ventana), min(len(palabras), i + ventana + 1)))
    vecinas = [palabras[j] for j in sorted(contexto)
               if palabras[j] in wv and palabras[j] not in terminos_aspecto]
    if not vecinas:
        return np.nan
    v = wv[vecinas].mean(axis=0)
    return float(v @ eje / np.linalg.norm(v))