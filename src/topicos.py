"""Modelado de tópicos: métricas de calidad y utilidades (proyecto LuxuryHotels)."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def terminos_principales(componentes, vocabulario, n=10):
    """Índices y palabras de los n términos con mayor peso en cada tópico."""
    indices = np.argsort(-componentes, axis=1)[:, :n]
    return indices, [[vocabulario[j] for j in fila] for fila in indices]


def npmi_documental(indices_top, matriz_binaria):
    """NPMI promedio de los pares de términos principales de cada tópico, con coocurrencia a nivel de documento.
    Devuelve el promedio general y el valor de cada tópico. Rango: −1 (nunca juntos) a +1 (siempre juntos)."""
    X = (matriz_binaria > 0).astype(np.float64).tocsc()
    n_docs = X.shape[0]
    por_topico = []
    for fila in indices_top:
        sub = X[:, fila]
        p = np.asarray(sub.sum(axis=0)).ravel() / n_docs
        conjunta = (sub.T @ sub).toarray() / n_docs
        valores = []
        for a in range(len(fila)):
            for b in range(a + 1, len(fila)):
                pab = conjunta[a, b]
                if pab == 0:
                    valores.append(-1.0)
                else:
                    valores.append(np.log(pab / (p[a] * p[b])) / -np.log(pab))
        por_topico.append(np.mean(valores))
    return float(np.mean(por_topico)), por_topico


def diversidad(indices_top):
    """Proporción de términos únicos entre los términos principales de todos los tópicos (1 = sin repeticiones)."""
    return len(set(np.ravel(indices_top))) / np.size(indices_top)


def estabilidad_jaccard(indices_a, indices_b):
    """Empareja los tópicos de dos ejecuciones (algoritmo húngaro) maximizando la similitud de Jaccard
    de sus términos principales y devuelve la similitud promedio de las parejas."""
    conjuntos_a = [set(f) for f in indices_a]
    conjuntos_b = [set(f) for f in indices_b]
    jac = np.array([[len(a & b) / len(a | b) for b in conjuntos_b] for a in conjuntos_a])
    filas, columnas = linear_sum_assignment(-jac)
    return float(jac[filas, columnas].mean())
