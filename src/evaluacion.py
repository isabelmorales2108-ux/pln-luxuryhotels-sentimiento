"""Métricas de evaluación para clasificación desbalanceada (proyecto LuxuryHotels)."""
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)


def metricas_clasificacion(y_real, y_pred, clase_minoritaria="negativo"):
    """Devuelve un diccionario con las métricas del proyecto. La exactitud se reporta solo como referencia."""
    return {
        "F1-macro": f1_score(y_real, y_pred, average="macro"),
        "Exactitud balanceada": balanced_accuracy_score(y_real, y_pred),
        f"Precisión ({clase_minoritaria})": precision_score(y_real, y_pred, pos_label=clase_minoritaria, zero_division=0),
        f"Recall ({clase_minoritaria})": recall_score(y_real, y_pred, pos_label=clase_minoritaria),
        "Exactitud (referencia)": accuracy_score(y_real, y_pred),
    }
    