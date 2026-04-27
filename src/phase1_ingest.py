"""
=============================================================================
PHASE 1 — Ingesta de Datos
=============================================================================
Responsabilidad:
  - Leer el CSV de Amazon Reviews usando PySpark (tolerante a líneas malformadas)
  - Validar que las columnas esperadas existen en el esquema
  - Registrar estadísticas básicas del dataset crudo (dimensiones, nulos, muestra)
  - Retornar el DataFrame PySpark para la siguiente fase

Por qué PySpark y no Pandas:
  PySpark permite procesar datasets que no caben en RAM distribuyendo el trabajo
  en particiones. Aunque este dataset es pequeño (~21 k filas), la arquitectura
  está pensada para escalar a millones de reviews sin cambiar el código.
=============================================================================
"""

import os
from loguru import logger
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


# Columnas que el pipeline necesita; las demás se conservan pero no son críticas
REQUIRED_COLUMNS = {"Rating", "Review Text", "Review Title"}


def create_spark_session(app_name: str = "AmazonSentimentAnalysis") -> SparkSession:
    """
    Crea y devuelve una SparkSession configurada para ejecución local.

    La SparkSession es el punto de entrada a PySpark. En modo local[*] utiliza
    tantos threads como cores tenga el host, simulando un entorno distribuido
    en una sola máquina.

    Args:
        app_name: Nombre visible en la Spark UI (http://localhost:4040 cuando corre).

    Returns:
        SparkSession activa y lista para usar.
    """
    master = os.getenv("SPARK_MASTER", "local[*]")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        # Reduce logs de Spark a WARNING para que la salida del pipeline sea legible
        .config("spark.ui.showConsoleProgress", "false")
        # Evita que Spark intente resolver hostnames en entornos Docker
        .config("spark.driver.host", "127.0.0.1")
        # Ajusta memoria del driver; suficiente para datasets de hasta ~1 GB
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )

    # Silencia los logs de Spark (INFO es muy verboso para un pipeline de portfolio)
    spark.sparkContext.setLogLevel("WARN")
    return spark


def ingest_data(spark: SparkSession, data_path: str) -> DataFrame:
    """
    Carga el CSV de Amazon Reviews en un DataFrame PySpark.

    PySpark infiere el esquema leyendo una muestra del archivo. La opción
    PERMISSIVE hace que filas con errores de parseo no detengan la carga,
    sino que se registren en la columna especial _corrupt_record.

    Args:
        spark:     SparkSession activa.
        data_path: Ruta absoluta al archivo CSV dentro del contenedor.

    Returns:
        DataFrame con todos los datos crudos del CSV.

    Raises:
        FileNotFoundError: Si la ruta no existe o el volumen no está montado.
        ValueError:        Si el CSV no contiene las columnas requeridas por el pipeline.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset no encontrado en: {data_path}\n"
            "Verifica que el volumen './data' esté montado en docker-compose.yml"
        )

    logger.info(f"Leyendo dataset desde: {data_path}")

    df = (
        spark.read
        .option("header", "true")          # Primera fila = nombres de columnas
        .option("inferSchema", "true")      # PySpark deduce tipos automáticamente
        .option("multiLine", "true")        # Reviews pueden contener saltos de línea
        .option("escape", '"')              # Maneja comillas dentro de campos
        .option("mode", "PERMISSIVE")       # Filas corruptas → columna _corrupt_record
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .csv(data_path)
    )

    _validate_schema(df, data_path)
    _log_dataset_stats(df)

    return df


def _validate_schema(df: DataFrame, data_path: str) -> None:
    """
    Comprueba que el DataFrame contiene todas las columnas que el pipeline requiere.

    Se lanza un error temprano (fail-fast) para detectar cambios en el esquema
    del CSV antes de que el error aparezca más adelante y sea difícil de depurar.

    Args:
        df:        DataFrame recién cargado.
        data_path: Usado solo para el mensaje de error.

    Raises:
        ValueError: Si alguna columna requerida no existe en el DataFrame.
    """
    existing_columns = set(df.columns)
    missing = REQUIRED_COLUMNS - existing_columns

    if missing:
        raise ValueError(
            f"El CSV en '{data_path}' no contiene las columnas requeridas: {missing}\n"
            f"Columnas presentes: {sorted(existing_columns)}"
        )

    logger.info(f"Esquema validado correctamente. Columnas: {df.columns}")


def _log_dataset_stats(df: DataFrame) -> None:
    """
    Imprime estadísticas básicas del dataset crudo para documentar la calidad
    de los datos de entrada antes de cualquier transformación.

    Estas métricas permiten detectar problemas como datasets vacíos,
    exceso de nulos o distribuciones de rating inesperadas.

    Args:
        df: DataFrame crudo recién cargado.
    """
    total_rows = df.count()
    total_cols = len(df.columns)

    logger.info("=" * 60)
    logger.info("ESTADÍSTICAS DEL DATASET CRUDO")
    logger.info("=" * 60)
    logger.info(f"  Filas totales : {total_rows:,}")
    logger.info(f"  Columnas      : {total_cols}")

    # Conteo de nulos por columna relevante
    null_counts = df.select([
        F.count(F.when(F.col(c).isNull() | (F.col(c) == ""), c)).alias(c)
        for c in ["Rating", "Review Text", "Review Title"]
    ]).collect()[0].asDict()

    logger.info("  Valores nulos / vacíos en columnas clave:")
    for col_name, null_n in null_counts.items():
        pct = (null_n / total_rows * 100) if total_rows > 0 else 0
        logger.info(f"    {col_name:<20}: {null_n:,} ({pct:.1f}%)")

    # Distribución de ratings (muestra si el dataset está desbalanceado)
    logger.info("  Distribución de ratings (valores crudos):")
    df.groupBy("Rating").count().orderBy("Rating").show(truncate=False)

    logger.info("=" * 60)
