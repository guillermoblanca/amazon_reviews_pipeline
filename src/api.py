"""
=============================================================================
src/api.py — API de Inferencia en Producción (FastAPI)
=============================================================================
Expone el modelo entrenado como un servicio REST con los endpoints estándar
que se esperan en un entorno de producción empresarial:

  GET  /health          → liveness probe (Kubernetes/ECS/health checks)
  GET  /model/info      → metadatos del modelo para auditoría
  POST /predict         → clasificación de una review individual
  POST /predict/batch   → clasificación de hasta 100 reviews en paralelo

Protocolo de producción implementado:
  · Modelo cargado una sola vez en el startup (patrón singleton)
  · SparkSession persistente entre peticiones (evita overhead de JVM)
  · Validación de esquema en entrada via Pydantic (422 automático)
  · Logging estructurado de cada petición con tiempo de respuesta
  · Gestión de ciclo de vida (lifespan) para startup/shutdown limpio
  · Documentación OpenAPI auto-generada en /docs y /redoc

Para levantar el servicio:
  docker-compose up api
  Acceder a: http://localhost:8000/docs
=============================================================================
"""

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, functions as F

from src.phase1_ingest import create_spark_session
from src.schemas import (
    BatchPredictionResponse,
    BatchReviewRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReviewRequest,
)

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/sentiment_model")
METRICS_PATH = Path("/app/outputs/metrics.json")
API_VERSION = "1.0.0"

# Estado global del proceso (singleton): SparkSession y modelo cargados una vez
_spark: SparkSession | None = None
_model: PipelineModel | None = None


# ── Ciclo de vida de la aplicación ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el startup y shutdown de la aplicación.

    El patrón lifespan (FastAPI >= 0.93) reemplaza los decoradores
    @app.on_event("startup") y @app.on_event("shutdown"), que están
    deprecados. Garantiza que los recursos se liberan aunque la app
    termine de forma inesperada.

    En el startup:
      1. Inicia la SparkSession (lenta la primera vez: ~15s por la JVM)
      2. Carga el modelo serializado desde disco (rápido: ~2s)

    En el shutdown:
      · Detiene la SparkSession y libera los recursos de JVM
    """
    global _spark, _model

    logger.info("Iniciando API de inferencia...")

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Modelo no encontrado en '{MODEL_PATH}'. "
            "Ejecuta primero el pipeline de entrenamiento: docker-compose run pipeline"
        )

    logger.info("Iniciando SparkSession para inferencia...")
    _spark = create_spark_session("SentimentAPI")

    logger.info(f"Cargando modelo desde: {MODEL_PATH}")
    _model = PipelineModel.load(MODEL_PATH)

    stages = [type(s).__name__ for s in _model.stages]
    logger.info(f"Modelo cargado. Etapas: {stages}")
    logger.info("API lista para recibir peticiones.")

    yield  # La aplicación está corriendo

    logger.info("Deteniendo API y liberando recursos...")
    if _spark:
        _spark.stop()
    logger.info("SparkSession detenida. API cerrada correctamente.")


# ── Aplicación FastAPI ───────────────────────────────────────────────────────

app = FastAPI(
    title="Amazon Reviews — Sentiment Analysis API",
    description=(
        "API de inferencia en producción para el modelo de análisis de sentimiento "
        "entrenado sobre ~21.000 reseñas de Amazon con PySpark MLlib."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Middleware: logging de peticiones ────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Registra cada petición HTTP con su tiempo de respuesta.

    En producción, este middleware alimenta sistemas de observabilidad
    (Datadog, CloudWatch, Prometheus) para monitorizar latencia y errores.
    """
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000  # ms

    logger.info(
        f"{request.method} {request.url.path} → "
        f"{response.status_code} ({elapsed:.1f}ms)"
    )
    return response


# ── Función de inferencia interna ────────────────────────────────────────────

def _run_inference(texts: list[str]) -> list[PredictionResponse]:
    """
    Ejecuta el pipeline de ML sobre una lista de textos y devuelve predicciones.

    Internamente crea un DataFrame PySpark con los textos, aplica el pipeline
    (Tokenizer → TF-IDF → LogReg) y extrae los resultados al driver.

    El UDF extrae `probability[1]` (P(positivo)) del vector de probabilidades
    de PySpark, que tiene la forma [P(negativo), P(positivo)].

    Args:
        texts: Lista de strings con el texto combinado (título + cuerpo).

    Returns:
        Lista de PredictionResponse con sentimiento, label, confianza y probabilidades.
    """
    rows = [{"review_combined": text} for text in texts]
    df = _spark.createDataFrame(rows)

    extract_positive_prob = F.udf(lambda vector: float(vector[1]))

    predictions = (
        _model.transform(df)
        .withColumn("prob_positive", extract_positive_prob(F.col("probability")))
        .select("prediction", "prob_positive")
        .collect()
    )

    results = []
    for row in predictions:
        label = int(row["prediction"])
        prob_pos = round(float(row["prob_positive"]), 4)
        prob_neg = round(1.0 - prob_pos, 4)
        confidence = prob_pos if label == 1 else prob_neg

        results.append(PredictionResponse(
            sentiment="positive" if label == 1 else "negative",
            label=label,
            confidence=round(confidence, 4),
            probabilities={"positive": prob_pos, "negative": prob_neg},
        ))

    return results


def _build_review_text(review: ReviewRequest) -> str:
    """Combina título y texto de la review igual que la Phase 2 del pipeline de entrenamiento."""
    parts = [review.title, review.text]
    return " ".join(p for p in parts if p).strip()


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Liveness probe estándar para Kubernetes, ECS y load balancers.",
    tags=["Operaciones"],
)
async def health_check() -> HealthResponse:
    """
    Devuelve el estado del servicio: si el modelo está cargado y Spark activo.

    Un status != 'ok' o model_loaded=False indica que el servicio no está
    listo para recibir tráfico (readiness probe fallida).
    """
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        spark_active=_spark is not None and _spark.sparkContext._jsc is not None,
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Metadatos del modelo",
    description="Devuelve información sobre el modelo en producción: etapas, ruta y métricas de entrenamiento.",
    tags=["Operaciones"],
)
async def model_info() -> ModelInfoResponse:
    """
    Endpoint de auditoría: expone la configuración y métricas del modelo activo.

    En producción, este endpoint se usa para verificar qué versión del modelo
    está sirviendo tráfico sin necesidad de acceder al sistema de archivos.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    stages = [type(s).__name__ for s in _model.stages]

    training_metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            training_metrics = json.load(f)

    return ModelInfoResponse(
        model_path=MODEL_PATH,
        pipeline_stages=stages,
        training_metrics=training_metrics,
        version=API_VERSION,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Clasificar una review",
    description="Clasifica una sola review de Amazon como positiva o negativa.",
    tags=["Inferencia"],
)
async def predict(review: ReviewRequest) -> PredictionResponse:
    """
    Clasifica el sentimiento de una review individual.

    El modelo espera texto pre-procesado de la misma forma que el pipeline
    de entrenamiento: título + cuerpo concatenados, sin necesidad de que
    el cliente aplique ninguna normalización.

    **Ejemplo de uso:**
    ```json
    {
      "title": "Best laptop I have ever owned",
      "text": "Fast, reliable and the battery lasts all day. Completely worth the price."
    }
    ```
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    text = _build_review_text(review)
    results = _run_inference([text])
    return results[0]


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Clasificar múltiples reviews",
    description=(
        "Clasifica un batch de hasta 100 reviews en una sola llamada. "
        "Más eficiente que llamadas individuales porque PySpark procesa "
        "todas las filas en paralelo en una sola tarea."
    ),
    tags=["Inferencia"],
)
async def predict_batch(request: BatchReviewRequest) -> BatchPredictionResponse:
    """
    Clasifica múltiples reviews en paralelo aprovechando PySpark.

    Usar este endpoint en lugar de llamadas individuales repetidas cuando
    se procesan múltiples reviews (p. ej. análisis de un período de tiempo,
    moderación de contenido en lote, o migración de datos históricos).
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    texts = [_build_review_text(r) for r in request.reviews]
    predictions = _run_inference(texts)

    positive_count = sum(1 for p in predictions if p.label == 1)
    negative_count = len(predictions) - positive_count

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        positive_count=positive_count,
        negative_count=negative_count,
        positive_ratio=round(positive_count / len(predictions), 4),
    )
