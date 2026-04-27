"""
=============================================================================
PHASE 2 — Transformación y Feature Engineering
=============================================================================
Responsabilidad:
  - Parsear el rating crudo ("Rated 4 out of 5 stars") → entero 1-5
  - Asignar etiqueta de sentimiento binario: positivo (1) / negativo (0)
      · Positivo: rating >= 4  (el cliente está satisfecho)
      · Negativo: rating <= 2  (el cliente está insatisfecho)
      · Neutral (rating = 3): excluido del modelo para evitar ambigüedad
  - Limpiar y normalizar el texto de las reviews (minúsculas, sin HTML, etc.)
  - Combinar Review Title + Review Text en un campo unificado para el modelo
  - Eliminar filas con texto vacío o etiqueta no asignable
  - Devolver el DataFrame limpio y listo para el entrenamiento

Decisiones de diseño:
  · Clasificación BINARIA: más interpretable y fácil de evaluar en portfolio.
  · Se excluyen las reviews de 3 estrellas porque son ambiguas semánticamente
    y degradan la capacidad del modelo para aprender patrones claros.
  · La combinación título + texto aprovecha toda la información disponible;
    el título suele condensar la opinión en pocas palabras (señal fuerte).
=============================================================================
"""

import os
import re
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType


# ── Constantes de etiquetado ────────────────────────────────────────────────
POSITIVE_THRESHOLD = int(os.getenv("POSITIVE_RATING_THRESHOLD", "4"))
NEGATIVE_THRESHOLD = int(os.getenv("NEGATIVE_RATING_THRESHOLD", "2"))

# Nombre final de la columna de texto unificado
FEATURE_COLUMN = "review_combined"
# Nombre de la columna de etiqueta binaria
LABEL_COLUMN = "label"


def transform_data(df: DataFrame) -> DataFrame:
    """
    Orquesta el pipeline de transformación en el orden correcto.

    Cada paso recibe el DataFrame del anterior, aplicando transformaciones
    encadenadas (estilo pipeline funcional) que PySpark ejecuta de forma
    perezosa (lazy evaluation) hasta que se necesita una acción.

    Args:
        df: DataFrame crudo proveniente de la fase de ingesta.

    Returns:
        DataFrame limpio con columnas `review_combined` (texto) y `label` (0/1).
    """
    logger.info("Iniciando transformaciones del dataset...")

    df = _parse_rating(df)
    df = _assign_sentiment_label(df)
    df = _clean_text(df)
    df = _combine_text_fields(df)
    df = _drop_invalid_rows(df)
    df = _select_final_columns(df)

    _log_transform_stats(df)
    return df


def _parse_rating(df: DataFrame) -> DataFrame:
    """
    Extrae el valor numérico del rating desde su representación textual.

    El campo crudo tiene el formato: "Rated 4 out of 5 stars"
    Se usa una expresión regular para capturar el primer dígito entero,
    que corresponde al rating del 1 al 5.

    Ejemplo:
        "Rated 3 out of 5 stars" → 3

    Args:
        df: DataFrame con columna 'Rating' en formato texto.

    Returns:
        DataFrame con columna adicional 'rating_numeric' (IntegerType).
    """
    # regexp_extract(columna, patrón, grupo_captura):
    # El patrón (\d) captura el primer dígito encontrado en el string
    df = df.withColumn(
        "rating_numeric",
        F.regexp_extract(F.col("Rating"), r"(\d)", 1).cast(IntegerType())
    )

    # Reemplaza 0 (cuando regexp no encuentra dígito) por null para filtrarlo luego
    df = df.withColumn(
        "rating_numeric",
        F.when(F.col("rating_numeric") == 0, None).otherwise(F.col("rating_numeric"))
    )

    logger.info("Rating parseado: columna 'rating_numeric' creada.")
    return df


def _assign_sentiment_label(df: DataFrame) -> DataFrame:
    """
    Asigna la etiqueta binaria de sentimiento según el rating numérico.

    Regla de negocio:
      · rating >= POSITIVE_THRESHOLD (default 4) → label = 1 (Positivo)
      · rating <= NEGATIVE_THRESHOLD (default 2) → label = 0 (Negativo)
      · rating == 3 (neutral)                    → label = null → se filtrará

    Esta separación binaria clara maximiza la señal para el modelo:
    las reviews de 4-5 estrellas contienen lenguaje positivo consistente,
    y las de 1-2 contienen lenguaje negativo consistente.

    Args:
        df: DataFrame con columna 'rating_numeric'.

    Returns:
        DataFrame con columna adicional 'label' (0, 1, o null).
    """
    df = df.withColumn(
        LABEL_COLUMN,
        F.when(F.col("rating_numeric") >= POSITIVE_THRESHOLD, 1)
         .when(F.col("rating_numeric") <= NEGATIVE_THRESHOLD, 0)
         .otherwise(None)  # Neutros → null → excluidos
    )

    logger.info(
        f"Etiquetas asignadas: "
        f">= {POSITIVE_THRESHOLD} stars → Positivo (1), "
        f"<= {NEGATIVE_THRESHOLD} stars → Negativo (0), "
        f"resto → Excluido"
    )
    return df


def _clean_text(df: DataFrame) -> DataFrame:
    """
    Aplica normalización básica al texto de las reviews.

    Pasos:
      1. Convertir a minúsculas: reduce el vocabulario (Apple = apple)
      2. Eliminar etiquetas HTML: algunas reviews provienen de scraping
      3. Reemplazar caracteres no alfanuméricos por espacio: elimina puntuación
         y caracteres especiales que no aportan significado semántico
      4. Colapsar múltiples espacios en uno: limpieza final

    Se usa F.regexp_replace de PySpark, que aplica la regex en paralelo
    sobre todas las particiones del DataFrame sin mover datos al driver.

    Args:
        df: DataFrame con columnas 'Review Text' y 'Review Title'.

    Returns:
        DataFrame con columnas 'text_clean' y 'title_clean'.
    """
    for raw_col, clean_col in [("Review Text", "text_clean"), ("Review Title", "title_clean")]:
        df = (
            df
            # Paso 1: minúsculas
            .withColumn(clean_col, F.lower(F.col(f"`{raw_col}`")))
            # Paso 2: eliminar etiquetas HTML (<br>, <p>, etc.)
            .withColumn(clean_col, F.regexp_replace(F.col(clean_col), r"<[^>]+>", " "))
            # Paso 3: eliminar caracteres no alfanuméricos (mantiene letras y números)
            .withColumn(clean_col, F.regexp_replace(F.col(clean_col), r"[^a-z0-9\s]", " "))
            # Paso 4: colapsar múltiples espacios
            .withColumn(clean_col, F.regexp_replace(F.col(clean_col), r"\s+", " "))
            # Paso 5: eliminar espacios al inicio/fin
            .withColumn(clean_col, F.trim(F.col(clean_col)))
        )

    logger.info("Texto normalizado: columnas 'text_clean' y 'title_clean' creadas.")
    return df


def _combine_text_fields(df: DataFrame) -> DataFrame:
    """
    Concatena el título limpio y el cuerpo de la review en un único campo.

    Justificación:
      El título suele ser la frase más representativa de la opinión del usuario
      (señal fuerte y concisa). Al unirlo con el cuerpo, el modelo recibe
      ambas fuentes de información en un solo vector de features.

    Formato resultante: "<título> <cuerpo>"
    Un espacio simple los separa; el Tokenizer posterior los dividirá en tokens.

    Args:
        df: DataFrame con columnas 'title_clean' y 'text_clean'.

    Returns:
        DataFrame con columna adicional 'review_combined'.
    """
    df = df.withColumn(
        FEATURE_COLUMN,
        F.concat_ws(" ", F.col("title_clean"), F.col("text_clean"))
    )

    logger.info(f"Campos combinados en '{FEATURE_COLUMN}': título + cuerpo de review.")
    return df


def _drop_invalid_rows(df: DataFrame) -> DataFrame:
    """
    Elimina filas que no pueden usarse para entrenar el modelo.

    Condiciones de eliminación:
      · label es null: rating neutral (3 estrellas) o rating no parseable
      · review_combined está vacío: sin texto no hay features posibles

    Esta limpieza garantiza que el modelo solo aprende de ejemplos válidos
    y evita errores en las etapas de vectorización y entrenamiento.

    Args:
        df: DataFrame transformado.

    Returns:
        DataFrame sin filas inválidas.
    """
    before = df.count()

    df = df.filter(
        F.col(LABEL_COLUMN).isNotNull() &
        (F.length(F.col(FEATURE_COLUMN)) > 5)  # Al menos 5 chars para ser útil
    )

    after = df.count()
    dropped = before - after
    logger.info(f"Filas eliminadas (nulas/vacías/neutras): {dropped:,} → Dataset limpio: {after:,} filas")
    return df


def _select_final_columns(df: DataFrame) -> DataFrame:
    """
    Selecciona solo las columnas necesarias para el entrenamiento.

    Retener únicamente las columnas relevantes reduce el uso de memoria
    en las particiones de Spark y acelera las operaciones posteriores.

    Columnas retenidas:
      · review_combined : texto unificado (feature de entrada al modelo)
      · label           : etiqueta binaria de sentimiento (0 o 1)
      · rating_numeric  : valor numérico del rating (útil para análisis)
      · Country         : metadato geográfico (útil para análisis exploratorio)

    Args:
        df: DataFrame con todas las columnas transformadas.

    Returns:
        DataFrame con solo las columnas finales.
    """
    return df.select(
        F.col(FEATURE_COLUMN),
        F.col(LABEL_COLUMN).cast("double"),  # MLlib requiere Double, no Int
        F.col("rating_numeric"),
        F.col("Country")
    )


def _log_transform_stats(df: DataFrame) -> None:
    """
    Registra el balance de clases del dataset transformado.

    Un dataset muy desbalanceado (p. ej. 95% positivo / 5% negativo) afecta
    al rendimiento del modelo y puede requerir técnicas como oversampling.
    Esta función documenta ese estado para el reporte de evaluación.

    Args:
        df: DataFrame transformado y limpio.
    """
    total = df.count()
    class_dist = df.groupBy(LABEL_COLUMN).count().collect()

    logger.info("=" * 60)
    logger.info("ESTADÍSTICAS DESPUÉS DE TRANSFORMACIÓN")
    logger.info("=" * 60)
    logger.info(f"  Total de ejemplos para entrenamiento: {total:,}")
    logger.info("  Distribución de clases:")
    for row in sorted(class_dist, key=lambda r: r[LABEL_COLUMN]):
        label_name = "Positivo" if row[LABEL_COLUMN] == 1.0 else "Negativo"
        pct = row["count"] / total * 100
        logger.info(f"    {label_name} ({int(row[LABEL_COLUMN])}): {row['count']:,} ({pct:.1f}%)")
    logger.info("=" * 60)
