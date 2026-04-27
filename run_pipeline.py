"""
=============================================================================
run_pipeline.py — Orquestador Principal del Pipeline de Análisis de Sentimiento
=============================================================================
Este es el punto de entrada del contenedor Docker. Ejecuta las 4 fases del
pipeline de forma secuencial, pasando los artefactos de cada fase a la siguiente.

Flujo completo:
  Phase 1 (Ingesta)       → DataFrame crudo PySpark
  Phase 2 (Transformación) → DataFrame limpio con etiquetas de sentimiento
  Phase 3 (Entrenamiento)  → PipelineModel + DataFrames de train/test
  Phase 4 (Evaluación)     → Métricas + visualizaciones + reporte

Variables de entorno configurables desde docker-compose.yml:
  DATA_PATH               : ruta al CSV dentro del contenedor
  SPARK_MASTER            : modo de ejecución de Spark (default: local[*])
  POSITIVE_RATING_THRESHOLD: rating mínimo para etiqueta positiva (default: 4)
  NEGATIVE_RATING_THRESHOLD: rating máximo para etiqueta negativa (default: 2)
  TEST_SPLIT_RATIO         : fracción del dataset para test (default: 0.2)
  RANDOM_SEED              : semilla aleatoria para reproducibilidad (default: 42)
  CV_FOLDS                 : número de folds en cross-validation (default: 3)

Para ejecutar el pipeline completo:
  docker-compose up --build
=============================================================================
"""

import os
import sys
import time
from loguru import logger

from src.phase1_ingest import create_spark_session, ingest_data
from src.phase2_transform import transform_data
from src.phase3_train import train_model
from src.phase4_evaluate import evaluate_model


# ── Configuración de logging ────────────────────────────────────────────────
logger.remove()  # Elimina el handler por defecto de loguru
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
    colorize=True
)
logger.add(
    "/app/outputs/pipeline.log",  # Log persistido en el volumen
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    level="INFO",
    rotation="10 MB"
)

DATA_PATH = os.getenv("DATA_PATH", "/app/data/Amazon_Reviews.csv")


def main() -> None:
    """
    Ejecuta el pipeline completo de análisis de sentimiento de principio a fin.

    El pipeline es idempotente: se puede ejecutar múltiples veces sobre el mismo
    dataset y producirá el mismo resultado (dado el mismo RANDOM_SEED).
    Los artefactos de ejecuciones anteriores se sobreescriben automáticamente.
    """
    pipeline_start = time.time()

    logger.info("")
    logger.info("=" * 60)
    logger.info("  AMAZON REVIEWS — SENTIMENT ANALYSIS PIPELINE")
    logger.info("  Tecnologías: PySpark MLlib · TF-IDF · Logistic Regression")
    logger.info("=" * 60)
    logger.info(f"  Dataset   : {DATA_PATH}")
    logger.info(f"  Spark     : {os.getenv('SPARK_MASTER', 'local[*]')}")
    logger.info("=" * 60)
    logger.info("")

    # ── Phase 1: Ingesta ────────────────────────────────────────────────────
    logger.info("▶ PHASE 1 — Ingesta de datos")
    t0 = time.time()

    spark = create_spark_session()
    df_raw = ingest_data(spark, DATA_PATH)

    logger.info(f"  Phase 1 completada en {time.time() - t0:.1f}s\n")

    # ── Phase 2: Transformación ─────────────────────────────────────────────
    logger.info("▶ PHASE 2 — Transformación y feature engineering")
    t0 = time.time()

    df_clean = transform_data(df_raw)

    logger.info(f"  Phase 2 completada en {time.time() - t0:.1f}s\n")

    # ── Phase 3: Entrenamiento ──────────────────────────────────────────────
    logger.info("▶ PHASE 3 — Entrenamiento del modelo (TF-IDF + Logistic Regression)")
    t0 = time.time()

    model, df_train, df_test = train_model(df_clean)

    logger.info(f"  Phase 3 completada en {time.time() - t0:.1f}s\n")

    # ── Phase 4: Evaluación ─────────────────────────────────────────────────
    logger.info("▶ PHASE 4 — Evaluación y generación de reporte")
    t0 = time.time()

    metrics = evaluate_model(model, df_test)

    logger.info(f"  Phase 4 completada en {time.time() - t0:.1f}s\n")

    # ── Resumen final ───────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info(f"  Pipeline completado en {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  F1-Score final: {metrics['f1_score']:.4f}")
    logger.info(f"  AUC-ROC final : {metrics['auc_roc']:.4f}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("  Artefactos disponibles en el host (directorio ./outputs/):")
    logger.info("    · pipeline.log          → log completo de ejecución")
    logger.info("    · metrics.json          → métricas del modelo")
    logger.info("    · confusion_matrix.png  → matriz de confusión")
    logger.info("    · roc_curve.png         → curva ROC")
    logger.info("    · probability_distribution.png → calibración del modelo")
    logger.info("")
    logger.info("  Modelo guardado en ./models/sentiment_model/")
    logger.info("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
