"""
=============================================================================
src/schemas.py — Contratos de la API (Pydantic v2)
=============================================================================
Define los tipos de entrada y salida de cada endpoint de la API de inferencia.

En producción, los schemas son el contrato entre el cliente (front-end,
microservicio, equipo de datos) y el modelo. Cualquier cambio en los schemas
es un cambio de API y requiere versionado o backwards-compatibility.

Pydantic valida automáticamente los tipos, lanza 422 Unprocessable Entity
si la petición no cumple el esquema, y genera la documentación OpenAPI
de forma automática (disponible en /docs cuando la API está levantada).
=============================================================================
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator


# ── Peticiones (Request) ────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    """Cuerpo de petición para clasificar una sola review."""

    title: str = Field(
        default="",
        description="Título de la review de Amazon (opcional).",
        examples=["Great product, totally worth it"]
    )
    text: str = Field(
        description="Texto completo de la review.",
        min_length=5,
        examples=["I bought this two months ago and it still works perfectly. Highly recommend."]
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("El texto de la review no puede estar vacío.")
        return v.strip()

    @field_validator("title")
    @classmethod
    def normalize_title(cls, v: str) -> str:
        return v.strip()


class BatchReviewRequest(BaseModel):
    """Cuerpo de petición para clasificar múltiples reviews en una sola llamada."""

    reviews: list[ReviewRequest] = Field(
        description="Lista de reviews a clasificar.",
        min_length=1,
        max_length=100  # Límite de seguridad: PySpark necesita lanzar una tarea por batch
    )


# ── Respuestas (Response) ────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Respuesta del modelo para una sola review."""

    sentiment: str = Field(
        description="Clase predicha: 'positive' o 'negative'.",
        examples=["positive"]
    )
    label: int = Field(
        description="Etiqueta numérica: 1 (positivo) o 0 (negativo).",
        examples=[1]
    )
    confidence: float = Field(
        description="Confianza del modelo en la predicción (0.0–1.0). "
                    "Es la probabilidad de la clase predicha.",
        ge=0.0,
        le=1.0,
        examples=[0.923]
    )
    probabilities: dict[str, float] = Field(
        description="Probabilidades para cada clase.",
        examples=[{"positive": 0.923, "negative": 0.077}]
    )


class BatchPredictionResponse(BaseModel):
    """Respuesta del modelo para un batch de reviews."""

    predictions: list[PredictionResponse]
    total: int = Field(description="Total de reviews procesadas.")
    positive_count: int = Field(description="Número de reviews clasificadas como positivas.")
    negative_count: int = Field(description="Número de reviews clasificadas como negativas.")
    positive_ratio: float = Field(description="Proporción de reviews positivas (0.0–1.0).")


class HealthResponse(BaseModel):
    """Respuesta del health check. Estándar para liveness probes en Kubernetes."""

    status: str = Field(examples=["ok"])
    model_loaded: bool
    spark_active: bool


class ModelInfoResponse(BaseModel):
    """Metadatos del modelo en producción. Útil para auditorías y debugging."""

    model_path: str
    pipeline_stages: list[str]
    training_metrics: dict
    version: str
