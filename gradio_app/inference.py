"""
gradio_app/inference.py — Lógica de inferencia sin PySpark
Espejo de Phase 2 implementado en Python puro para el entorno de HuggingFace Spaces.
"""

import re
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).parent / "sentiment_model.joblib"

_model = None


def load_model():
    """Carga el modelo una sola vez (singleton). Llamar antes de predecir."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado en {MODEL_PATH}. "
                "Ejecuta primero: docker-compose run --rm export"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def clean_and_combine(title: str, text: str) -> str:
    """
    Aplica el mismo preprocesamiento de Phase 2 en Python puro:
    lower → sin HTML → solo alfanumérico → colapsar espacios → trim.
    Combina título + cuerpo como hace _combine_text_fields de Phase 2.
    """
    def _clean(s: str) -> str:
        s = s.lower()
        s = re.sub(r"<[^>]+>", " ", s)
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    return f"{_clean(title)} {_clean(text)}".strip()


def predict_single(title: str, text: str) -> dict:
    """
    Clasifica una review individual.

    Returns:
        Dict con 'sentiment', 'label', 'confidence' y 'probabilities'.
    """
    model = load_model()
    combined = clean_and_combine(title, text)

    proba = model.predict_proba([combined])[0]  # [P(neg), P(pos)]
    label = int(model.predict([combined])[0])

    prob_pos = round(float(proba[1]), 4)
    prob_neg = round(float(proba[0]), 4)
    confidence = prob_pos if label == 1 else prob_neg

    return {
        "sentiment": "positive" if label == 1 else "negative",
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {"Positive ⭐": prob_pos, "Negative ❌": prob_neg},
    }


def predict_batch(reviews: list[dict]) -> list[dict]:
    """
    Clasifica una lista de reviews en un solo llamado al modelo.

    Args:
        reviews: Lista de dicts con claves 'title' y 'text'.

    Returns:
        Lista de resultados en el mismo orden que la entrada.
    """
    model = load_model()
    texts = [clean_and_combine(r.get("title", ""), r.get("text", "")) for r in reviews]

    probas = model.predict_proba(texts)
    labels = model.predict(texts)

    results = []
    for i, (label, proba) in enumerate(zip(labels, probas)):
        label = int(label)
        prob_pos = round(float(proba[1]), 4)
        prob_neg = round(float(proba[0]), 4)
        results.append({
            "sentiment": "positive" if label == 1 else "negative",
            "label": label,
            "confidence": round(prob_pos if label == 1 else prob_neg, 4),
            "probabilities": {"Positive ⭐": prob_pos, "Negative ❌": prob_neg},
        })

    return results
