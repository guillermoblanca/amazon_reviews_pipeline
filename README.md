# Amazon Reviews — Sentiment Analysis Pipeline

Pipeline de análisis de sentimiento sobre reseñas de Amazon, diseñado como proyecto de portfolio para demostrar capacidad end-to-end en ingeniería de datos y machine learning.

**Stack:** PySpark · MLlib · Docker · TF-IDF · Logistic Regression

---

## Objetivo

Clasificar automáticamente reseñas de Amazon como **positivas** o **negativas** en función del texto, entrenando un modelo de NLP con PySpark MLlib y evaluándolo con métricas estándar de clasificación.

**Dataset:** ~21.000 reseñas de Amazon con campos de rating (1-5 estrellas), título, texto, país y fecha de publicación.

---

## Arquitectura del Pipeline

```
CSV (Amazon Reviews)
        │
        ▼
┌─────────────────┐
│  PHASE 1        │  Ingesta con PySpark
│  Ingest         │  · Validación de esquema
│                 │  · Estadísticas del dataset crudo
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PHASE 2        │  Transformación y Feature Engineering
│  Transform      │  · Parseo de rating ("Rated 4 out of 5 stars" → 4)
│                 │  · Etiquetado binario (≥4 → positivo, ≤2 → negativo)
│                 │  · Limpieza de texto (lower, sin HTML, sin puntuación)
│                 │  · Combinación título + cuerpo de review
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PHASE 3        │  Entrenamiento (MLlib Pipeline)
│  Train          │  · Tokenizer → StopWordsRemover → HashingTF → IDF
│                 │  · Logistic Regression con Cross-Validation (3 folds)
│                 │  · Búsqueda de hiperparámetros (regParam, elasticNet)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PHASE 4        │  Evaluación y Reporte
│  Evaluate       │  · Accuracy, Precision, Recall, F1-Score, AUC-ROC
│                 │  · Matriz de confusión, Curva ROC, Distribución P(positivo)
│                 │  · Exportación de métricas en JSON
└─────────────────┘
         │
         ▼
   /outputs/     → métricas + gráficos
   /models/      → modelo entrenado (PySpark format)
```

---

## Requisitos

- [Docker](https://www.docker.com/get-started) ≥ 24.0
- [Docker Compose](https://docs.docker.com/compose/) ≥ 2.0
- RAM disponible: mínimo 4 GB (configurado en `docker-compose.yml`)

---

## Ejecución

### Pipeline completo (un solo comando)

```bash
docker-compose up --build
```

Primera ejecución: ~5-10 min (descarga imagen base + instala dependencias).  
Ejecuciones posteriores: ~2-5 min (imagen cacheada).

### Ver resultados sin reconstruir

```bash
docker-compose up
```

### Limpiar contenedores

```bash
docker-compose down
```

---

## Estructura del Proyecto

```
etl_project/
├── data/
│   └── Amazon_Reviews.csv       # Dataset de entrada
├── src/
│   ├── __init__.py
│   ├── phase1_ingest.py         # Phase 1: Ingesta con PySpark
│   ├── phase2_transform.py      # Phase 2: Transformación y etiquetado
│   ├── phase3_train.py          # Phase 3: Entrenamiento del modelo
│   └── phase4_evaluate.py       # Phase 4: Evaluación y visualizaciones
├── models/
│   └── sentiment_model/         # Modelo PySpark exportado (generado)
├── outputs/
│   ├── metrics.json             # Métricas del modelo (generado)
│   ├── confusion_matrix.png     # Matriz de confusión (generado)
│   ├── roc_curve.png            # Curva ROC (generado)
│   ├── probability_distribution.png  # Calibración (generado)
│   └── pipeline.log             # Log completo de ejecución (generado)
├── code/
│   └── inspector.py             # Script de exploración inicial del CSV
├── Dockerfile                   # Imagen Docker del pipeline
├── docker-compose.yml           # Configuración de servicios y volúmenes
├── requirements.txt             # Dependencias Python
└── run_pipeline.py              # Orquestador principal (entrypoint)
```

---

## Configuración

Todos los parámetros se configuran como variables de entorno en `docker-compose.yml`, sin necesidad de modificar el código:

| Variable | Default | Descripción |
|---|---|---|
| `DATA_PATH` | `/app/data/Amazon_Reviews.csv` | Ruta al dataset dentro del contenedor |
| `SPARK_MASTER` | `local[*]` | Modo de ejecución de Spark |
| `POSITIVE_RATING_THRESHOLD` | `4` | Rating mínimo para etiqueta positiva |
| `NEGATIVE_RATING_THRESHOLD` | `2` | Rating máximo para etiqueta negativa |
| `TEST_SPLIT_RATIO` | `0.2` | Proporción del dataset para test |
| `RANDOM_SEED` | `42` | Semilla aleatoria (reproducibilidad) |
| `CV_FOLDS` | `3` | Número de folds en cross-validation |

---

## Decisiones de Diseño

### Por qué PySpark MLlib y no scikit-learn

PySpark permite procesar datasets que superan la RAM disponible distribuyendo el trabajo en particiones. Aunque este dataset (~21k filas) cabría en memoria, la arquitectura es idéntica a la que se usaría con millones de reviews en un cluster real.

### Por qué clasificación binaria (no multiclase 1-5)

Las reviews de 3 estrellas son semánticamente ambiguas (pueden contener tanto lenguaje positivo como negativo). Incluirlas como clase "neutral" degrada el rendimiento del modelo. La clasificación binaria positivo/negativo produce modelos más interpretables y con mejor F1-Score.

### Por qué TF-IDF + Logistic Regression

- **TF-IDF** captura la importancia relativa de cada término en el corpus, filtrando palabras muy comunes (bajo poder discriminativo) y resaltando términos específicos de cada clase.
- **Logistic Regression** es la baseline estándar para clasificación de texto: interpretable (los pesos revelan qué palabras son más predictivas), eficiente, y difícil de superar significativamente con modelos más complejos en tareas de sentimiento binario.
- **Cross-Validation** garantiza que la selección de hiperparámetros no sobreajusta al conjunto de validación.

### Por qué se excluye el rating 3 (neutral)

Un modelo entrenado con reviews ambiguas aprende representaciones ruidosas. Al excluirlos, el modelo aprende patrones limpios: lenguaje claramente positivo (4-5★) vs. claramente negativo (1-2★). Esto resulta en mejor generalización y métricas más honestas.

---

## Outputs del Pipeline

Tras ejecutar `docker-compose up --build`, los siguientes artefactos quedan disponibles en el directorio local `./outputs/`:

| Archivo | Descripción |
|---|---|
| `metrics.json` | Accuracy, Precision, Recall, F1, AUC-ROC en formato JSON |
| `confusion_matrix.png` | TP/FP/TN/FN visualizados como heatmap |
| `roc_curve.png` | Curva ROC con área bajo la curva |
| `probability_distribution.png` | Histograma de P(positivo) separado por clase real |
| `pipeline.log` | Log completo con tiempos de cada fase |

El modelo entrenado se guarda en `./models/sentiment_model/` en formato PySpark nativo, listo para ser cargado con:

```python
from pyspark.ml import PipelineModel
model = PipelineModel.load("./models/sentiment_model")
predictions = model.transform(new_data_df)
```

---

## Métricas Esperadas

Basado en la naturaleza del dataset (reviews con lenguaje claro de satisfacción/insatisfacción):

| Métrica | Esperado |
|---|---|
| Accuracy | > 90% |
| F1-Score | > 0.88 |
| AUC-ROC | > 0.95 |

---

## Tecnologías

| Tecnología | Versión | Rol |
|---|---|---|
| Python | 3.11 | Lenguaje del pipeline |
| PySpark | 3.5.1 | Procesamiento distribuido + MLlib |
| OpenJDK | 17 | Runtime de Java para Spark |
| Pandas | 2.2.2 | Análisis exploratorio inicial |
| scikit-learn | 1.5.0 | Métricas de evaluación |
| Matplotlib / Seaborn | 3.9 / 0.13 | Visualizaciones |
| Docker | ≥ 24 | Contenedorización |
| Docker Compose | ≥ 2 | Orquestación de servicios |
