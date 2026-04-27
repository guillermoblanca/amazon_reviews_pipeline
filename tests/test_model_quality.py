"""
=============================================================================
tests/test_model_quality.py — Tests de regresión del modelo (Golden Set)
=============================================================================
Verifica que el modelo en producción mantiene un nivel mínimo de calidad
sobre un conjunto de reviews con etiquetas conocidas (golden set).

¿Por qué son necesarios estos tests?
  · Un modelo re-entrenado con nuevos datos puede ser más preciso en métricas
    globales pero degradar en casos específicos importantes (regresión).
  · Estos tests capturan esas regresiones antes de que lleguen a producción.
  · Son el contrato de calidad mínima del modelo: si falla, no se despliega.

Umbrales de calidad:
  Los umbrales están definidos conservadoramente por debajo de los resultados
  reales del modelo (94.32% accuracy, 89.55% F1) para tolerar variabilidad
  natural entre re-entrenamientos sin disparar falsas alarmas.

  · Accuracy mínima sobre golden set: 80%  (modelo real: 94%)
  · Todas las reviews "clear_positive/clear_negative" deben clasificarse bien
  · La confianza en casos claros debe ser > 65%

En CI/CD, estos tests se ejecutan después de cada re-entrenamiento para
decidir automáticamente si el nuevo modelo se promueve a producción (gate de calidad).
=============================================================================
"""

import pytest
from fastapi.testclient import TestClient

# Umbrales de calidad mínima (más bajos que el modelo real para tolerar varianza)
MIN_ACCURACY_GOLDEN_SET = 0.80
MIN_CONFIDENCE_CLEAR_CASES = 0.65


class TestGoldenSet:
    """Regresión sobre el golden set completo."""

    def test_overall_accuracy_on_golden_set(
        self, api_client: TestClient, sample_reviews: list[dict]
    ):
        """
        La accuracy sobre el golden set debe superar el umbral mínimo.

        Si este test falla, significa que el modelo ha degradado respecto
        al baseline y NO debe desplegarse a producción.
        """
        correct = 0
        total = len(sample_reviews)

        for review in sample_reviews:
            payload = {"title": review["title"], "text": review["text"]}
            data = api_client.post("/predict", json=payload).json()
            if data["label"] == review["expected_label"]:
                correct += 1

        accuracy = correct / total
        assert accuracy >= MIN_ACCURACY_GOLDEN_SET, (
            f"Accuracy sobre golden set: {accuracy:.2%} < umbral mínimo {MIN_ACCURACY_GOLDEN_SET:.0%}. "
            f"El modelo ha regresado. No desplegar."
        )

    def test_all_clear_positives_classified_correctly(
        self, api_client: TestClient, sample_reviews: list[dict]
    ):
        """
        Todas las reviews claramente positivas (category=clear_positive)
        deben clasificarse como positivas.

        Estas reviews usan lenguaje inequívocamente positivo. Si el modelo
        falla en alguna, indica un problema grave de calibración.
        """
        clear_positives = [r for r in sample_reviews if r["category"] == "clear_positive"]
        assert len(clear_positives) > 0, "El golden set debe contener reviews clear_positive."

        for review in clear_positives:
            payload = {"title": review["title"], "text": review["text"]}
            data = api_client.post("/predict", json=payload).json()
            assert data["label"] == 1, (
                f"Review clara positiva (id={review['id']}) clasificada como negativa. "
                f"Confianza: {data['confidence']:.2%}. Texto: '{review['text'][:80]}...'"
            )

    def test_all_clear_negatives_classified_correctly(
        self, api_client: TestClient, sample_reviews: list[dict]
    ):
        """
        Todas las reviews claramente negativas (category=clear_negative)
        deben clasificarse como negativas.
        """
        clear_negatives = [r for r in sample_reviews if r["category"] == "clear_negative"]
        assert len(clear_negatives) > 0, "El golden set debe contener reviews clear_negative."

        for review in clear_negatives:
            payload = {"title": review["title"], "text": review["text"]}
            data = api_client.post("/predict", json=payload).json()
            assert data["label"] == 0, (
                f"Review clara negativa (id={review['id']}) clasificada como positiva. "
                f"Confianza: {data['confidence']:.2%}. Texto: '{review['text'][:80]}...'"
            )

    def test_confidence_on_clear_cases(
        self, api_client: TestClient, sample_reviews: list[dict]
    ):
        """
        En casos claros (no edge cases), la confianza del modelo debe ser alta.

        Baja confianza en casos obvios indica que el modelo ha perdido
        poder discriminativo, posiblemente por over-regularización o datos
        de entrenamiento contaminados.
        """
        clear_cases = [
            r for r in sample_reviews
            if r["category"] in ("clear_positive", "clear_negative")
        ]

        for review in clear_cases:
            payload = {"title": review["title"], "text": review["text"]}
            data = api_client.post("/predict", json=payload).json()
            assert data["confidence"] >= MIN_CONFIDENCE_CLEAR_CASES, (
                f"Confianza baja en caso claro (id={review['id']}): "
                f"{data['confidence']:.2%} < {MIN_CONFIDENCE_CLEAR_CASES:.0%}"
            )


class TestEdgeCases:
    """Verifica el comportamiento del modelo en casos límite."""

    def test_review_without_title(self, api_client: TestClient, sample_reviews: list[dict]):
        """Una review sin título (solo texto) debe clasificarse correctamente."""
        no_title = next(r for r in sample_reviews if r["category"] == "edge_case_no_title")
        payload = {"title": no_title["title"], "text": no_title["text"]}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == no_title["expected_label"]

    def test_short_review(self, api_client: TestClient, sample_reviews: list[dict]):
        """Una review muy corta pero con palabras claras debe ser clasificable."""
        short = next(r for r in sample_reviews if r["category"] == "edge_case_short")
        payload = {"title": short["title"], "text": short["text"]}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == short["expected_label"]

    def test_long_review(self, api_client: TestClient):
        """Una review muy larga no debe causar errores ni timeouts."""
        long_text = "Great product. " * 200  # ~3000 palabras
        payload = {"title": "Excellent", "text": long_text}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["sentiment"] == "positive"

    def test_non_english_text(self, api_client: TestClient):
        """
        Texto en otro idioma (el modelo fue entrenado en inglés).
        El test verifica que no lanza excepción — la predicción puede ser incorrecta
        pero el servicio debe seguir en pie (degradación elegante).
        """
        payload = {
            "title": "Producto horrible",
            "text": "Muy mal producto, completamente roto desde el primer día. No lo compres."
        }
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 200  # No debe fallar, aunque la predicción sea incierta

    def test_batch_with_mixed_sentiments(self, api_client: TestClient, sample_reviews: list[dict]):
        """
        Un batch con mix de positivos y negativos debe tener positive_ratio entre 0 y 1.
        """
        clear = [r for r in sample_reviews if r["category"] in ("clear_positive", "clear_negative")]
        payload = {"reviews": [{"title": r["title"], "text": r["text"]} for r in clear]}
        data = api_client.post("/predict/batch", json=payload).json()
        assert 0.0 < data["positive_ratio"] < 1.0, (
            "Un batch con mix de positivos y negativos no debe tener ratio extremo (0 o 1)."
        )
