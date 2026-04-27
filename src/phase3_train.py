"""
=============================================================================
PHASE 3 — Entrenamiento del Modelo de Sentimiento
=============================================================================
Responsabilidad:
  - Dividir el dataset en conjuntos de entrenamiento (80%) y test (20%)
  - Construir el pipeline de ML de PySpark MLlib:
      1. Tokenizer           → divide el texto en palabras (tokens)
      2. StopWordsRemover    → elimina palabras vacías (the, is, and...)
      3. HashingTF           → convierte tokens en vector de frecuencias
      4. IDF                 → pondera la importancia de cada término
      5. LogisticRegression  → clasificador binario de sentimiento
  - Ejecutar validación cruzada (Cross-Validation) para optimizar hiperparámetros
  - Guardar el modelo entrenado en disco para servicio posterior
  - Retornar el modelo, el DataFrame de test y las predicciones

Por qué TF-IDF + Logistic Regression:
  · TF-IDF (Term Frequency – Inverse Document Frequency) convierte texto en
    vectores numéricos capturando qué palabras son importantes en cada review
    y poco comunes en el corpus completo (señal discriminativa alta).
  · Logistic Regression es interpretable, eficiente y una baseline sólida
    para clasificación de texto. Ideal para portfolio: demuestra comprensión
    del proceso sin ocultar la lógica en una caja negra compleja.
  · La validación cruzada automatiza la búsqueda de hiperparámetros y evita
    sobreajuste al evaluar en múltiples particiones del dataset.
=============================================================================
"""

import os
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator


MODEL_OUTPUT_PATH = "/app/models/sentiment_model"


def train_model(df: DataFrame) -> tuple[PipelineModel, DataFrame, DataFrame]:
    """
    Orquesta el entrenamiento completo: split → pipeline → cross-validation → guardado.

    Args:
        df: DataFrame limpio proveniente de la fase de transformación,
            con columnas 'review_combined' (texto) y 'label' (0.0/1.0).

    Returns:
        Tupla de (mejor_modelo, df_train, df_test) donde:
          · mejor_modelo: PipelineModel con los mejores hiperparámetros encontrados
          · df_train:     Datos de entrenamiento (para análisis posterior)
          · df_test:      Datos de test (usados en Phase 4 para evaluación)
    """
    test_ratio = float(os.getenv("TEST_SPLIT_RATIO", "0.2"))
    random_seed = int(os.getenv("RANDOM_SEED", "42"))

    df_train, df_test = _split_dataset(df, test_ratio, random_seed)
    pipeline = _build_pipeline()
    best_model = _train_with_cross_validation(pipeline, df_train, random_seed)
    _save_model(best_model)

    return best_model, df_train, df_test


def _split_dataset(df: DataFrame, test_ratio: float, seed: int) -> tuple[DataFrame, DataFrame]:
    """
    Divide el dataset en subconjuntos de entrenamiento y test de forma aleatoria.

    La división estratificada garantiza que ambos subconjuntos mantienen
    proporciones similares de clases positiva y negativa, evitando sesgos
    que ocurrirían si, por azar, el test tuviese muchos más positivos que el train.

    PySpark no tiene stratifiedSplit nativo; la división aleatoria simple es
    suficiente para datasets de más de 10 k filas donde la distribución de
    clases converge por la ley de los grandes números.

    Args:
        df:         DataFrame completo.
        test_ratio: Fracción del dataset para test (e.g., 0.2 = 20%).
        seed:       Semilla aleatoria para reproducibilidad.

    Returns:
        Tupla (df_train, df_test).
    """
    train_ratio = 1.0 - test_ratio
    df_train, df_test = df.randomSplit([train_ratio, test_ratio], seed=seed)

    logger.info("=" * 60)
    logger.info("DIVISIÓN DEL DATASET")
    logger.info("=" * 60)
    logger.info(f"  Train ({train_ratio*100:.0f}%): {df_train.count():,} filas")
    logger.info(f"  Test  ({test_ratio*100:.0f}%): {df_test.count():,} filas")
    logger.info(f"  Semilla aleatoria: {seed}")
    logger.info("=" * 60)

    return df_train, df_test


def _build_pipeline() -> Pipeline:
    """
    Construye el pipeline de ML de PySpark con 5 etapas encadenadas.

    Un Pipeline de PySpark agrupa Transformers y Estimators en secuencia.
    Cada etapa recibe el output de la anterior como input. Esto garantiza
    que las mismas transformaciones se aplican de forma idéntica al train
    y al test, previniendo data leakage.

    Etapas:
      1. Tokenizer:        "buy it now" → ["buy", "it", "now"]
      2. StopWordsRemover: ["buy", "it", "now"] → ["buy", "now"]
                           Elimina palabras funcionales sin significado semántico.
      3. HashingTF:        ["buy", "now"] → Vector esparso de frecuencias.
                           HashingTF es más rápido que CountVectorizer porque
                           no necesita construir un vocabulario completo.
                           numFeatures=65536 balancea precisión vs. memoria.
      4. IDF:              Pondera cada término por su rareza en el corpus.
                           IDF alto → término discriminativo; IDF bajo → común.
      5. LogisticRegression: Clasificador lineal que aprende pesos para cada
                           término del vocabulario TF-IDF. regParam controla
                           la regularización L2 (evita sobreajuste).

    Returns:
        Pipeline de PySpark sin entrenar.
    """
    tokenizer = Tokenizer(
        inputCol="review_combined",
        outputCol="tokens"
    )

    stop_remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="tokens_filtered"
        # Usa la lista de stopwords en inglés por defecto de PySpark
    )

    hashing_tf = HashingTF(
        inputCol="tokens_filtered",
        outputCol="raw_features",
        numFeatures=65536  # 2^16: espacio de hashing amplio para reducir colisiones
    )

    idf = IDF(
        inputCol="raw_features",
        outputCol="features",
        minDocFreq=2  # Ignora términos que aparecen en menos de 2 documentos (ruido)
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,           # Máximo de iteraciones del optimizador L-BFGS
        family="binomial"      # Clasificación binaria (0 vs 1)
    )

    pipeline = Pipeline(stages=[tokenizer, stop_remover, hashing_tf, idf, lr])
    logger.info("Pipeline ML construido: Tokenizer → StopWordsRemover → HashingTF → IDF → LogisticRegression")
    return pipeline


def _train_with_cross_validation(
    pipeline: Pipeline,
    df_train: DataFrame,
    seed: int
) -> PipelineModel:
    """
    Ejecuta validación cruzada (k-fold CV) para encontrar los mejores hiperparámetros.

    Cross-Validation:
      · Divide df_train en k folds de igual tamaño
      · Para cada combinación de hiperparámetros del grid, entrena k modelos:
        cada uno usa k-1 folds para train y 1 fold para validación
      · Selecciona la combinación con mejor métrica promedio (AUC-ROC)
      · Re-entrena el modelo ganador con TODO df_train

    Hiperparámetros explorados:
      · regParam (regularización L2): controla la penalización de coeficientes
        grandes. Valores mayores reducen el sobreajuste pero pueden subajustar.
      · elasticNetParam: mezcla entre L1 (0) y L2 (1) regularización.
        L1 produce modelos más esparsos (más términos con peso 0).

    Args:
        pipeline: Pipeline de ML sin entrenar.
        df_train: Datos de entrenamiento.
        seed:     Semilla para reproducibilidad del split de folds.

    Returns:
        PipelineModel con los mejores hiperparámetros encontrados por CV.
    """
    cv_folds = int(os.getenv("CV_FOLDS", "3"))

    # Acceso al estimador LogisticRegression dentro del pipeline para el grid
    lr_stage = pipeline.getStages()[-1]

    param_grid = (
        ParamGridBuilder()
        .addGrid(lr_stage.regParam, [0.01, 0.1])          # 2 valores de regularización
        .addGrid(lr_stage.elasticNetParam, [0.0, 0.5])    # L2 puro vs. mezcla L1/L2
        .build()
    )
    # Total: 2 × 2 = 4 combinaciones × 3 folds = 12 entrenamientos

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        metricName="areaUnderROC"  # AUC-ROC: métrica robusta para datasets desbalanceados
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=cv_folds,
        seed=seed,
        parallelism=2  # Entrena 2 combinaciones en paralelo para acelerar CV
    )

    logger.info(f"Iniciando Cross-Validation ({cv_folds} folds, {len(param_grid)} combinaciones)...")
    logger.info("Esto puede tardar 2-5 minutos dependiendo del hardware disponible.")

    cv_model = cv.fit(df_train)

    # Loguea los resultados de cada combinación para transparencia
    _log_cv_results(cv_model, param_grid, cv_folds)

    best_model = cv_model.bestModel
    logger.info("Mejor modelo seleccionado por Cross-Validation.")
    return best_model


def _log_cv_results(cv_model, param_grid: list, cv_folds: int) -> None:
    """
    Muestra los resultados de cada combinación de hiperparámetros explorada.

    Permite al revisor del portfolio ver que se evaluaron múltiples opciones
    y que la selección del modelo final está basada en evidencia empírica.

    Args:
        cv_model:   CrossValidatorModel resultante del entrenamiento.
        param_grid: Lista de combinaciones de hiperparámetros evaluadas.
        cv_folds:   Número de folds usados.
    """
    logger.info("=" * 60)
    logger.info("RESULTADOS CROSS-VALIDATION (AUC-ROC promedio)")
    logger.info("=" * 60)

    avg_metrics = cv_model.avgMetrics
    lr_stage = cv_model.bestModel.stages[-1]

    for i, (params, metric) in enumerate(zip(param_grid, avg_metrics)):
        param_str = " | ".join([
            f"regParam={v:.3f}" if k.name == "regParam" else f"elasticNet={v:.1f}"
            for k, v in params.items()
        ])
        best_marker = " ← MEJOR" if metric == max(avg_metrics) else ""
        logger.info(f"  [{i+1}] {param_str} → AUC-ROC: {metric:.4f}{best_marker}")

    best_reg = lr_stage.getRegParam()
    best_en = lr_stage.getElasticNetParam()
    logger.info(f"\n  Hiperparámetros finales: regParam={best_reg}, elasticNetParam={best_en}")
    logger.info("=" * 60)


def _save_model(model: PipelineModel) -> None:
    """
    Persiste el modelo entrenado en formato PySpark nativo (Parquet + metadata).

    El formato PySpark permite cargar el modelo en el futuro con:
        PipelineModel.load(MODEL_OUTPUT_PATH)
    y aplicarlo directamente a nuevos textos sin reentrenar.

    El directorio se sobreescribe en cada ejecución para que el pipeline
    sea idempotente (misma entrada → misma salida, sin acumulación de archivos).

    Args:
        model: PipelineModel entrenado listo para guardar.
    """
    logger.info(f"Guardando modelo en: {MODEL_OUTPUT_PATH}")

    # overwrite evita error si el directorio ya existe de una ejecución previa
    model.write().overwrite().save(MODEL_OUTPUT_PATH)

    logger.info(f"Modelo guardado correctamente.")
