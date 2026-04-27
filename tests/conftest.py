"""
=============================================================================
tests/conftest.py — Fixtures compartidas entre todos los tests
=============================================================================
Los fixtures de pytest son funciones que preparan recursos (conexiones,
objetos, datos) y los inyectan en los tests que los declaran como argumentos.

Scopes disponibles:
  · "session"  → el fixture se crea una sola vez para toda la suite de tests.
                  Ideal para SparkSession y modelos (costosos de inicializar).
  · "module"   → una vez por fichero de test.
  · "function" → una vez por test (default). Ideal para datos mutables.

En un entorno de CI/CD, este conftest.py es el punto central de configuración:
si se necesita conectar a una base de datos de test o a un modelo en staging,
se cambia aquí sin tocar los tests individuales.
=============================================================================
"""

import json
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from pyspark.sql import SparkSession


# ── Fixtures de infraestructura ──────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    """
    SparkSession compartida entre todos los tests de la suite.

    Se crea una sola vez (scope="session") para evitar el overhead de
    arrancar y detener la JVM múltiples veces. PySpark reutiliza la misma
    SparkContext si ya existe una activa.

    En CI/CD (GitHub Actions, Jenkins) esta SparkSession corre en el agente
    de build, sin necesidad de un cluster externo.
    """
    session = (
        SparkSession.builder
        .master("local[2]")              # 2 cores suficientes para tests
        .appName("SentimentTests")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")  # Silencia logs de Spark en tests

    yield session

    session.stop()


@pytest.fixture(scope="session")
def api_client() -> Generator[TestClient, None, None]:
    """
    Cliente HTTP de test para la API de FastAPI.

    TestClient usa httpx internamente y arranca la aplicación en un
    thread separado sin necesitar un servidor real. Esto permite tests
    de integración rápidos y sin puertos de red.

    El lifespan de FastAPI (startup + shutdown) se ejecuta completo,
    lo que significa que el modelo PySpark se carga al crear el cliente
    y se descarga al cerrarlo, exactamente igual que en producción.

    IMPORTANTE: requiere que el modelo esté entrenado en ./models/sentiment_model
    Ejecutar primero: docker-compose run --rm pipeline
    """
    from src.api import app
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


# ── Fixtures de datos ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_reviews() -> list[dict]:
    """
    Carga el golden set de reviews desde el archivo JSON de test.

    Este conjunto de reviews con etiquetas conocidas es la fuente de verdad
    para los tests de regresión del modelo. Si el modelo se re-entrena y
    deja de clasificar correctamente alguna de estas reviews, el test falla
    y alerta al equipo antes de llegar a producción.
    """
    data_path = Path(__file__).parent / "data" / "sample_reviews.json"
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def positive_review() -> dict:
    """Una review claramente positiva para tests individuales."""
    return {
        "title": "Best product ever",
        "text": "This is absolutely fantastic. Works perfectly and the quality is outstanding. Highly recommend.",
    }


@pytest.fixture
def negative_review() -> dict:
    """Una review claramente negativa para tests individuales."""
    return {
        "title": "Terrible quality",
        "text": "Broke after two days. Complete waste of money. Worst purchase I have ever made. Avoid at all costs.",
    }
