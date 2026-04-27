# =============================================================================
# Dockerfile — Amazon Reviews Sentiment Analysis Pipeline
# Base: Python 3.11 slim + OpenJDK 17 (requerido por PySpark)
# =============================================================================

FROM python:3.11-slim

LABEL maintainer="Portfolio Project — Sentiment Analysis"
LABEL description="PySpark-based sentiment analysis pipeline for Amazon Reviews"

# ---------------------------------------------------------------------------
# 1. Dependencias del sistema: Java (requerido por Spark) y utilidades
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Variables de entorno para PySpark
# JAVA_HOME NO se hardcodea aquí porque la ruta cambia según la arquitectura
# del host (amd64 → java-17-openjdk-amd64, arm64 → java-17-openjdk-arm64).
# El entrypoint script lo detecta dinámicamente en tiempo de ejecución.
# ---------------------------------------------------------------------------
ENV PYSPARK_PYTHON=python3
ENV SPARK_LOCAL_IP=127.0.0.1

# ---------------------------------------------------------------------------
# 3. Directorio de trabajo dentro del contenedor
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# 4. Instalar dependencias Python (cacheadas en una capa separada)
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# 5. Copiar el código fuente del proyecto y el entrypoint
# ---------------------------------------------------------------------------
COPY src/ ./src/
COPY run_pipeline.py .
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# ---------------------------------------------------------------------------
# 6. Crear directorios de salida (montados como volúmenes en compose)
# ---------------------------------------------------------------------------
RUN mkdir -p /app/data /app/models /app/outputs

# ---------------------------------------------------------------------------
# 7. Punto de entrada: detecta JAVA_HOME y ejecuta el pipeline completo
# ---------------------------------------------------------------------------
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "run_pipeline.py"]
