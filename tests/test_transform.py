"""
=============================================================================
tests/test_transform.py — Tests unitarios de Phase 2 (Transformación)
=============================================================================
Verifica las funciones de transformación de datos de forma aislada,
sin depender del modelo entrenado ni de la API.

Tests unitarios vs. tests de integración:
  · Unitarios (este archivo): cada función se prueba de forma aislada.
    Rápidos, deterministas, sin efectos secundarios.
  · Integración (test_api.py): el sistema completo se prueba de extremo a extremo.
    Más lentos, requieren el modelo cargado.

Estos tests capturan errores en la lógica de transformación antes de que
afecten al entrenamiento o a la inferencia. Son especialmente útiles si
el formato del CSV cambia (ej. el rating pasa de texto a número).
=============================================================================
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StructField, StructType, StringType, DoubleType

from src.phase2_transform import (
    FEATURE_COLUMN,
    LABEL_COLUMN,
    _assign_sentiment_label,
    _clean_text,
    _combine_text_fields,
    _drop_invalid_rows,
    _parse_rating,
)


# ── Fixtures locales ─────────────────────────────────────────────────────────

@pytest.fixture
def raw_df(spark: SparkSession):
    """DataFrame con ratings en formato texto (igual que el CSV original)."""
    data = [
        ("Rated 5 out of 5 stars", "Amazing product", "Works perfectly"),
        ("Rated 1 out of 5 stars", "Terrible",        "Broken on arrival"),
        ("Rated 3 out of 5 stars", "Okay",             "Nothing special"),
        ("Rated 4 out of 5 stars", "Good",             "Happy with it"),
        ("Rated 2 out of 5 stars", "Disappointed",    "Not as described"),
        (None,                     "No rating",        "Some text"),
    ]
    schema = StructType([
        StructField("Rating",       StringType(), True),
        StructField("Review Title", StringType(), True),
        StructField("Review Text",  StringType(), True),
    ])
    return spark.createDataFrame(data, schema)


@pytest.fixture
def labeled_df(spark: SparkSession):
    """DataFrame con rating_numeric ya parseado para tests de etiquetado."""
    data = [(5,), (4,), (3,), (2,), (1,), (None,)]
    schema = StructType([StructField("rating_numeric", IntegerType(), True)])
    return spark.createDataFrame(data, schema)


@pytest.fixture
def text_df(spark: SparkSession):
    """DataFrame con texto crudo para tests de limpieza."""
    data = [
        ("GREAT Product!!!", "Works <br>perfectly <p>every</p> day."),
        ("  Bad   quality  ", "Terrible!!!   Awful..."),
        ("",                  "Some text here"),
    ]
    schema = StructType([
        StructField("Review Title", StringType(), True),
        StructField("Review Text",  StringType(), True),
    ])
    return spark.createDataFrame(data, schema)


# ── Tests de _parse_rating ────────────────────────────────────────────────────

class TestParseRating:
    """Verifica la extracción del valor numérico del rating."""

    def test_extracts_digit_from_format(self, raw_df):
        result = _parse_rating(raw_df)
        ratings = {
            row["Rating"]: row["rating_numeric"]
            for row in result.select("Rating", "rating_numeric").collect()
            if row["Rating"] is not None
        }
        assert ratings["Rated 5 out of 5 stars"] == 5
        assert ratings["Rated 1 out of 5 stars"] == 1
        assert ratings["Rated 3 out of 5 stars"] == 3

    def test_null_rating_becomes_null(self, raw_df):
        result = _parse_rating(raw_df)
        null_row = result.filter(F.col("Rating").isNull()).select("rating_numeric").first()
        assert null_row["rating_numeric"] is None

    def test_column_rating_numeric_created(self, raw_df):
        result = _parse_rating(raw_df)
        assert "rating_numeric" in result.columns

    def test_original_columns_preserved(self, raw_df):
        result = _parse_rating(raw_df)
        for col in raw_df.columns:
            assert col in result.columns


# ── Tests de _assign_sentiment_label ─────────────────────────────────────────

class TestAssignSentimentLabel:
    """Verifica la asignación de etiquetas binarias de sentimiento."""

    def test_rating_5_is_positive(self, labeled_df):
        result = _assign_sentiment_label(labeled_df)
        label = result.filter(F.col("rating_numeric") == 5).select(LABEL_COLUMN).first()[LABEL_COLUMN]
        assert label == 1

    def test_rating_4_is_positive(self, labeled_df):
        result = _assign_sentiment_label(labeled_df)
        label = result.filter(F.col("rating_numeric") == 4).select(LABEL_COLUMN).first()[LABEL_COLUMN]
        assert label == 1

    def test_rating_3_is_null(self, labeled_df):
        result = _assign_sentiment_label(labeled_df)
        label = result.filter(F.col("rating_numeric") == 3).select(LABEL_COLUMN).first()[LABEL_COLUMN]
        assert label is None  # Neutros excluidos

    def test_rating_2_is_negative(self, labeled_df):
        result = _assign_sentiment_label(labeled_df)
        label = result.filter(F.col("rating_numeric") == 2).select(LABEL_COLUMN).first()[LABEL_COLUMN]
        assert label == 0

    def test_rating_1_is_negative(self, labeled_df):
        result = _assign_sentiment_label(labeled_df)
        label = result.filter(F.col("rating_numeric") == 1).select(LABEL_COLUMN).first()[LABEL_COLUMN]
        assert label == 0

    def test_null_rating_gives_null_label(self, labeled_df):
        result = _assign_sentiment_label(labeled_df)
        label = result.filter(F.col("rating_numeric").isNull()).select(LABEL_COLUMN).first()[LABEL_COLUMN]
        assert label is None

    def test_only_two_non_null_label_values(self, labeled_df):
        result = _assign_sentiment_label(labeled_df)
        distinct_labels = (
            result.filter(F.col(LABEL_COLUMN).isNotNull())
            .select(LABEL_COLUMN)
            .distinct()
            .rdd.flatMap(lambda x: x)
            .collect()
        )
        assert set(distinct_labels) == {0, 1}


# ── Tests de _clean_text ──────────────────────────────────────────────────────

class TestCleanText:
    """Verifica la normalización del texto de las reviews."""

    def test_converts_to_lowercase(self, text_df):
        result = _clean_text(text_df)
        titles = [row["title_clean"] for row in result.select("title_clean").collect()]
        assert all(t == t.lower() for t in titles if t)

    def test_removes_html_tags(self, text_df):
        result = _clean_text(text_df)
        texts = [row["text_clean"] for row in result.select("text_clean").collect()]
        assert all("<" not in t and ">" not in t for t in texts if t)

    def test_removes_punctuation(self, text_df):
        result = _clean_text(text_df)
        titles = [row["title_clean"] for row in result.select("title_clean").collect()]
        import re
        assert all(not re.search(r"[!?.,;:]", t) for t in titles if t)

    def test_no_multiple_spaces(self, text_df):
        result = _clean_text(text_df)
        for col_name in ("title_clean", "text_clean"):
            values = [row[col_name] for row in result.select(col_name).collect()]
            assert all("  " not in v for v in values if v)

    def test_creates_clean_columns(self, text_df):
        result = _clean_text(text_df)
        assert "title_clean" in result.columns
        assert "text_clean" in result.columns


# ── Tests de _combine_text_fields ─────────────────────────────────────────────

class TestCombineTextFields:
    """Verifica la concatenación de título y cuerpo."""

    def test_combined_column_created(self, spark: SparkSession):
        df = spark.createDataFrame(
            [("great product", "works perfectly")],
            ["title_clean", "text_clean"]
        )
        result = _combine_text_fields(df)
        assert FEATURE_COLUMN in result.columns

    def test_combined_contains_both_parts(self, spark: SparkSession):
        df = spark.createDataFrame(
            [("great product", "works perfectly")],
            ["title_clean", "text_clean"]
        )
        result = _combine_text_fields(df)
        combined = result.select(FEATURE_COLUMN).first()[FEATURE_COLUMN]
        assert "great product" in combined
        assert "works perfectly" in combined

    def test_empty_title_handled(self, spark: SparkSession):
        df = spark.createDataFrame(
            [("", "only body text here")],
            ["title_clean", "text_clean"]
        )
        result = _combine_text_fields(df)
        combined = result.select(FEATURE_COLUMN).first()[FEATURE_COLUMN]
        assert combined.strip() != ""


# ── Tests de _drop_invalid_rows ───────────────────────────────────────────────

class TestDropInvalidRows:
    """Verifica que se eliminan correctamente las filas no utilizables."""

    def test_drops_null_labels(self, spark: SparkSession):
        schema = StructType([
            StructField(LABEL_COLUMN,    DoubleType(), True),
            StructField(FEATURE_COLUMN,  StringType(), True),
        ])
        df = spark.createDataFrame(
            [(1.0, "great product"), (None, "some text"), (0.0, "terrible")],
            schema
        )
        result = _drop_invalid_rows(df)
        assert result.count() == 2

    def test_drops_empty_text(self, spark: SparkSession):
        schema = StructType([
            StructField(LABEL_COLUMN,   DoubleType(), True),
            StructField(FEATURE_COLUMN, StringType(), True),
        ])
        df = spark.createDataFrame(
            [(1.0, "great product"), (0.0, "   "), (1.0, "ok")],
            schema
        )
        result = _drop_invalid_rows(df)
        assert result.count() == 2

    def test_keeps_valid_rows(self, spark: SparkSession):
        schema = StructType([
            StructField(LABEL_COLUMN,   DoubleType(), True),
            StructField(FEATURE_COLUMN, StringType(), True),
        ])
        valid_data = [
            (1.0, "this product is amazing and works great"),
            (0.0, "broken terrible waste of money"),
        ]
        df = spark.createDataFrame(valid_data, schema)
        result = _drop_invalid_rows(df)
        assert result.count() == 2
