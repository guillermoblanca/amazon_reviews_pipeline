"""
=============================================================================
tests/test_api.py — Tests de integración de la API
=============================================================================
Verifica que cada endpoint de la API responde correctamente: códigos HTTP,
estructura del JSON, validación de entrada y comportamiento con casos límite.

Estos tests se ejecutan con TestClient (sin servidor real) y garantizan:
  · El contrato de la API (esquema request/response) no se rompe
  · Las validaciones de entrada funcionan (422 para datos inválidos)
  · Los endpoints operacionales (health, info) están disponibles
  · El modelo clasifica correctamente casos claros de sentimiento

En CI/CD, estos tests se ejecutan en cada PR antes de mergear a main.
Si alguno falla, el despliegue a producción queda bloqueado.
=============================================================================
"""

import pytest
from fastapi.testclient import TestClient


# ── Health check ─────────────────────────────────────────────────────────────

class TestHealth:
    """Verifica que el liveness probe responde correctamente."""

    def test_health_returns_200(self, api_client: TestClient):
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self, api_client: TestClient):
        data = api_client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "spark_active" in data

    def test_health_model_is_loaded(self, api_client: TestClient):
        data = api_client.get("/health").json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["spark_active"] is True


# ── Información del modelo ────────────────────────────────────────────────────

class TestModelInfo:
    """Verifica el endpoint de auditoría del modelo."""

    def test_model_info_returns_200(self, api_client: TestClient):
        response = api_client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_has_pipeline_stages(self, api_client: TestClient):
        data = api_client.get("/model/info").json()
        stages = data["pipeline_stages"]
        assert isinstance(stages, list)
        assert len(stages) == 5  # Tokenizer, StopWordsRemover, HashingTF, IDF, LogReg

    def test_model_info_has_training_metrics(self, api_client: TestClient):
        data = api_client.get("/model/info").json()
        metrics = data["training_metrics"]
        assert "f1_score" in metrics
        assert "auc_roc" in metrics
        assert "accuracy" in metrics

    def test_model_info_pipeline_order(self, api_client: TestClient):
        data = api_client.get("/model/info").json()
        stages = data["pipeline_stages"]
        assert stages[0] == "Tokenizer"
        assert stages[-1] == "LogisticRegressionModel"


# ── Predicción individual ─────────────────────────────────────────────────────

class TestPredict:
    """Verifica el endpoint /predict para reviews individuales."""

    def test_predict_positive_review(self, api_client: TestClient, positive_review: dict):
        response = api_client.post("/predict", json=positive_review)
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "positive"
        assert data["label"] == 1
        assert data["confidence"] > 0.7  # El modelo debe ser seguro en casos claros

    def test_predict_negative_review(self, api_client: TestClient, negative_review: dict):
        response = api_client.post("/predict", json=negative_review)
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "negative"
        assert data["label"] == 0
        assert data["confidence"] > 0.7

    def test_predict_response_schema(self, api_client: TestClient, positive_review: dict):
        """Verifica que el esquema de respuesta tiene todos los campos requeridos."""
        data = api_client.post("/predict", json=positive_review).json()
        assert "sentiment" in data
        assert "label" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "positive" in data["probabilities"]
        assert "negative" in data["probabilities"]

    def test_predict_probabilities_sum_to_one(self, api_client: TestClient, positive_review: dict):
        """Las probabilidades deben sumar 1 (o muy cerca, por redondeo de floats)."""
        data = api_client.post("/predict", json=positive_review).json()
        total = data["probabilities"]["positive"] + data["probabilities"]["negative"]
        assert abs(total - 1.0) < 0.01

    def test_predict_confidence_matches_sentiment(self, api_client: TestClient, positive_review: dict):
        """La confianza debe ser la probabilidad de la clase predicha."""
        data = api_client.post("/predict", json=positive_review).json()
        if data["label"] == 1:
            assert abs(data["confidence"] - data["probabilities"]["positive"]) < 0.01
        else:
            assert abs(data["confidence"] - data["probabilities"]["negative"]) < 0.01

    def test_predict_without_title(self, api_client: TestClient):
        """El título es opcional; solo con texto debe funcionar."""
        payload = {"text": "Great product, very happy with my purchase."}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_missing_text_returns_422(self, api_client: TestClient):
        """Sin el campo 'text' (requerido), la API debe devolver 422."""
        response = api_client.post("/predict", json={"title": "Great"})
        assert response.status_code == 422

    def test_predict_empty_text_returns_422(self, api_client: TestClient):
        """Un texto vacío o solo espacios debe ser rechazado con 422."""
        response = api_client.post("/predict", json={"text": "   "})
        assert response.status_code == 422

    def test_predict_very_short_text_returns_422(self, api_client: TestClient):
        """Texto menor a 5 caracteres debe ser rechazado (demasiado corto para inferir)."""
        response = api_client.post("/predict", json={"text": "bad"})
        assert response.status_code == 422

    def test_predict_label_is_binary(self, api_client: TestClient, positive_review: dict):
        """El label solo puede ser 0 o 1."""
        data = api_client.post("/predict", json=positive_review).json()
        assert data["label"] in (0, 1)

    def test_predict_sentiment_is_valid_string(self, api_client: TestClient, negative_review: dict):
        """El sentiment solo puede ser 'positive' o 'negative'."""
        data = api_client.post("/predict", json=negative_review).json()
        assert data["sentiment"] in ("positive", "negative")


# ── Predicción en batch ───────────────────────────────────────────────────────

class TestPredictBatch:
    """Verifica el endpoint /predict/batch para múltiples reviews."""

    def test_batch_basic(self, api_client: TestClient, positive_review: dict, negative_review: dict):
        payload = {"reviews": [positive_review, negative_review]}
        response = api_client.post("/predict/batch", json=payload)
        assert response.status_code == 200

    def test_batch_response_count(self, api_client: TestClient, positive_review: dict, negative_review: dict):
        """La respuesta debe contener tantas predicciones como reviews enviadas."""
        payload = {"reviews": [positive_review, negative_review]}
        data = api_client.post("/predict/batch", json=payload).json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_counts_are_consistent(self, api_client: TestClient, positive_review: dict, negative_review: dict):
        """positive_count + negative_count debe ser igual a total."""
        payload = {"reviews": [positive_review, negative_review]}
        data = api_client.post("/predict/batch", json=payload).json()
        assert data["positive_count"] + data["negative_count"] == data["total"]

    def test_batch_positive_ratio_range(self, api_client: TestClient, positive_review: dict):
        """La ratio de positivos debe estar entre 0.0 y 1.0."""
        payload = {"reviews": [positive_review]}
        data = api_client.post("/predict/batch", json=payload).json()
        assert 0.0 <= data["positive_ratio"] <= 1.0

    def test_batch_empty_list_returns_422(self, api_client: TestClient):
        """Una lista vacía de reviews debe ser rechazada."""
        response = api_client.post("/predict/batch", json={"reviews": []})
        assert response.status_code == 422

    def test_batch_single_review(self, api_client: TestClient, positive_review: dict):
        """Un batch de una sola review debe funcionar correctamente."""
        payload = {"reviews": [positive_review]}
        response = api_client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
