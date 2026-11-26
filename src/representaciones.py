"""Representaciones distribuidas: vectores de reseña a partir de Word2Vec (proyecto LuxuryHotels)."""
import numpy as np


def vector_resena(texto, wv, pesos=None):
    """Promedio de los vectores Word2Vec de las palabras de una reseña.

    Si se pasan `pesos` (diccionario palabra -> IDF), el promedio se pondera por la rareza de cada palabra.
    Las palabras fuera del vocabulario del modelo se ignoran; si ninguna está, devuelve un vector de ceros.
    """
    palabras = [p for p in texto.split() if p in wv]
    if not palabras:
        return np.zeros(wv.vector_size)
    w = np.array([pesos.get(p, 1.0) for p in palabras]) if pesos else None
    return np.average(wv[palabras], axis=0, weights=w)


def matriz_resenas(textos, wv, pesos=None):
    """Aplica `vector_resena` a una colección de textos y devuelve una matriz (n_reseñas × dimensión)."""
    return np.vstack([vector_resena(t, wv, pesos) for t in textos])
