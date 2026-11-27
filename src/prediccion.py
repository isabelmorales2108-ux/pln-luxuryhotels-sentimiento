"""Función reutilizable para clasificar el sentimiento de una reseña nueva (proyecto LuxuryHotels)."""
from pathlib import Path

import joblib

from src.preprocesamiento import cargar_nlp, lematizar_corpus

_RAIZ = Path(__file__).resolve().parent.parent
_modelo, _nlp = None, None


def predict_sentiment(texto, ruta_modelo=_RAIZ / "models" / "tfidf_logistica.joblib"):
    """Normaliza una reseña en inglés con el mismo proceso del notebook 01 y devuelve
    la clase predicha y la probabilidad de que sea negativa."""
    global _modelo, _nlp
    if _modelo is None:
        _modelo = joblib.load(ruta_modelo)
        _nlp = cargar_nlp()
    limpio = lematizar_corpus([texto], _nlp)
    clase = _modelo.predict(limpio)[0]
    p_negativo = _modelo.predict_proba(limpio)[0][list(_modelo.classes_).index("negativo")]
    return {"clase": clase, "probabilidad_negativo": round(float(p_negativo), 3), "texto_normalizado": limpio[0]}