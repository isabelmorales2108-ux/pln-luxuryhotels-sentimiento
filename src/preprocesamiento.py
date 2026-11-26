"""Normalización y lematización del texto de las reseñas (proyecto LuxuryHotels).

Funciones reutilizadas por todos los notebooks del proyecto.
"""
import re

import spacy

# Palabras de negación: se conservan aunque spaCy las marque como palabras vacías,
# porque invierten la polaridad ("not clean", "never again").
NEGACIONES = {"no", "not", "never", "nor", "neither", "none", "nothing", "nobody", "nowhere", "without"}


def cargar_nlp(modelo="en_core_web_sm"):
    """Carga el modelo de spaCy sin el analizador sintáctico ni el reconocedor de entidades,
    que no se necesitan para lematizar y harían el proceso varias veces más lento."""
    return spacy.load(modelo, disable=["parser", "ner"])


def preparar_texto(texto):
    """Limpieza previa con expresiones regulares, antes de pasar el texto a spaCy."""
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)          # URL
    texto = re.sub(r"\S+@\S+", " ", texto)                         # correos
    texto = re.sub(r"([.,!?;:)])(?=[A-Za-z])", r"\1 ", texto)      # "Everyone,If" -> "Everyone, If"
    texto = re.sub(r"(\w)\1{2,}", r"\1\1", texto)                  # "sooooo" -> "soo"
    texto = re.sub(r"\s+", " ", texto)                             # espacios múltiples
    return texto.strip()


def lematizar_corpus(textos, nlp, palabras_extra=(), batch_size=500):
    """Devuelve una lista de textos normalizados: lemas en minúscula, sin palabras vacías
    (excepto negaciones), sin números ni puntuación y sin tokens de una letra."""
    vacias = (nlp.Defaults.stop_words - NEGACIONES) | set(palabras_extra)
    resultado = []
    for doc in nlp.pipe((preparar_texto(t) for t in textos), batch_size=batch_size):
        lemas = []
        for tok in doc:
            if tok.lower_ in ("n't", "nt"):          # didn't -> did + n't -> "not"
                lemas.append("not")
                continue
            if not tok.is_alpha:
                continue
            lema = tok.lemma_.lower()
            if lema in vacias or tok.lower_ in vacias or len(lema) < 2:
                continue
            lemas.append(lema)
        resultado.append(" ".join(lemas))
    return resultado