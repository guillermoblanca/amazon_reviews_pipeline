# Reporte Técnico — Análisis de Sentimiento sobre Amazon Reviews

**Autor:** Portfolio Project  
**Tecnologías:** PySpark 3.5 · MLlib · Docker · TF-IDF · Logistic Regression  
**Dataset:** ~21.000 reseñas de Amazon (Kaggle)  
**Tarea:** Clasificación binaria de sentimiento (Positivo / Negativo)

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Dataset](#2-dataset)
3. [Arquitectura del Pipeline](#3-arquitectura-del-pipeline)
4. [Phase 1 — Ingesta](#4-phase-1--ingesta)
5. [Phase 2 — Transformación y Feature Engineering](#5-phase-2--transformación-y-feature-engineering)
6. [Phase 3 — Entrenamiento del Modelo](#6-phase-3--entrenamiento-del-modelo)
7. [Phase 4 — Evaluación y Resultados](#7-phase-4--evaluación-y-resultados)
8. [Decisiones Técnicas Justificadas](#8-decisiones-técnicas-justificadas)
9. [Cómo Reproducir el Proyecto](#9-cómo-reproducir-el-proyecto)

---

## 1. Resumen Ejecutivo

Este proyecto implementa un pipeline de análisis de sentimiento end-to-end sobre reseñas de Amazon. El objetivo es demostrar capacidad para:

- Ingestar y transformar datos no estructurados con **PySpark** a escala
- Construir y evaluar un modelo de **NLP** con **MLlib** usando buenas prácticas (cross-validation, separación train/test, métricas múltiples)
- Empaquetar toda la solución en **Docker** para reproducibilidad total con un único comando

El modelo final (TF-IDF + Logistic Regression) alcanza resultados sólidos para producción, demostrando que las decisiones de feature engineering superan en importancia a la complejidad del clasificador.

---

## 2. Dataset

| Campo | Detalle |
|---|---|
| Fuente | Kaggle — Amazon Reviews |
| Filas totales | ~21.056 reseñas |
| Columnas clave | `Rating`, `Review Title`, `Review Text`, `Country`, `Review Date` |
| Formato del rating | Texto: `"Rated 4 out of 5 stars"` |
| Distribución geográfica | Multi-país (US, GB, DE, FR, ...) |

### Campos del dataset

```
Reviewer Name  │ Profile Link │ Country │ Review Count │ Review Date
Rating         │ Review Title │ Review Text              │ Date of Experience
```

**Observación crítica:** el campo `Rating` no es numérico directamente sino una cadena de texto. Extraer el valor numérico es el primer paso de la transformación.

---

## 3. Arquitectura del Pipeline

El pipeline está dividido en 4 fases independientes y desacopladas. Cada fase recibe el artefacto de la anterior y produce uno nuevo:

```
┌──────────────────────────────────────────────────────────────┐
│                    docker-compose up --build                  │
└────────────────────────┬─────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │     PHASE 1         │
              │     Ingesta         │  CSV → DataFrame PySpark
              │  phase1_ingest.py   │  · Validación de esquema
              └──────────┬──────────┘  · Estadísticas del dataset crudo
                         │
              ┌──────────▼──────────┐
              │     PHASE 2         │
              │  Transformación     │  · Parseo de rating (regex)
              │ phase2_transform.py │  · Etiquetado binario de sentimiento
              └──────────┬──────────┘  · Limpieza y normalización de texto
                         │             · Combinación título + cuerpo
              ┌──────────▼──────────┐
              │     PHASE 3         │
              │  Entrenamiento      │  · Split 80/20 train/test
              │  phase3_train.py    │  · Pipeline: Tokenizer→TF-IDF→LogReg
              └──────────┬──────────┘  · Cross-validation 3-fold
                         │
              ┌──────────▼──────────┐
              │     PHASE 4         │
              │    Evaluación       │  · F1, AUC-ROC, Precision, Recall
              │ phase4_evaluate.py  │  · Matriz de confusión
              └──────────┬──────────┘  · Curva ROC
                         │             · Distribución de probabilidades
              ┌──────────▼──────────┐
              │     OUTPUTS         │
              │  ./outputs/         │  metrics.json + 3 gráficos
              │  ./models/          │  Modelo PySpark serializado
              └─────────────────────┘
```

---

## 4. Phase 1 — Ingesta

**Archivo:** `src/phase1_ingest.py`

### ¿Qué hace?

1. Crea una **SparkSession** en modo `local[*]` (usa todos los cores del host)
2. Lee el CSV con inferencia de esquema y tolerancia a líneas malformadas (`mode=PERMISSIVE`)
3. Valida que las columnas críticas existen antes de continuar (fail-fast)
4. Registra estadísticas del dataset crudo: dimensiones, nulos por columna, distribución de ratings

### Configuración de la SparkSession

```python
spark = (
    SparkSession.builder
    .appName("AmazonSentimentAnalysis")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
```

**Por qué `local[*]`:** ejecuta Spark en la misma JVM usando todos los cores disponibles. Es idéntico a la API de un cluster real: el mismo código escala a un clúster con cero cambios, solo cambiando el `master` en `docker-compose.yml`.

### Estadísticas registradas al iniciar

```
ESTADÍSTICAS DEL DATASET CRUDO
  Filas totales : 21,056
  Columnas      : 9
  Valores nulos / vacíos en columnas clave:
    Rating              : 0 (0.0%)
    Review Text         : 124 (0.6%)
    Review Title        : 89 (0.4%)
```

### Lectura tolerante a errores

```python
df = (
    spark.read
    .option("multiLine", "true")   # Reviews con saltos de línea internos
    .option("escape", '"')          # Comillas anidadas en el texto
    .option("mode", "PERMISSIVE")   # Filas corruptas → columna _corrupt_record
    .csv(data_path)
)
```

La opción `multiLine=true` es crítica porque los textos de review frecuentemente contienen saltos de línea que romperían un lector CSV estándar.

---

## 5. Phase 2 — Transformación y Feature Engineering

**Archivo:** `src/phase2_transform.py`

### Flujo de transformaciones

```
Rating crudo                    "Rated 4 out of 5 stars"
    │  _parse_rating()
    ▼
rating_numeric                  4
    │  _assign_sentiment_label()
    ▼
label                           1 (Positivo)  /  0 (Negativo)  /  null (excluido)
    │
Review Title + Review Text      "Great product  I bought this..."
    │  _clean_text()
    ▼
title_clean + text_clean        "great product  i bought this"
    │  _combine_text_fields()
    ▼
review_combined                 "great product i bought this"
    │  _drop_invalid_rows()
    ▼
Dataset limpio                  Solo filas con label válida y texto no vacío
```

### Regla de etiquetado binario

| Rating | Clase | Label |
|---|---|---|
| ⭐⭐⭐⭐⭐ (5) | Positivo | `1` |
| ⭐⭐⭐⭐ (4) | Positivo | `1` |
| ⭐⭐⭐ (3) | **Excluido** | `null` |
| ⭐⭐ (2) | Negativo | `0` |
| ⭐ (1) | Negativo | `0` |

**Justificación de excluir el rating 3:** las reseñas de 3 estrellas son semánticamente ambiguas. El mismo texto puede contener "buen producto pero la entrega fue horrible" — un modelo no puede aprender un patrón claro de ese caso. Incluirlas introduce ruido que reduce F1 y AUC. La elección binaria polar produce representaciones más limpias.

### Parseo del rating con regex

```python
df = df.withColumn(
    "rating_numeric",
    F.regexp_extract(F.col("Rating"), r"(\d)", 1).cast(IntegerType())
)
```

La expresión `(\d)` captura el primer dígito del string `"Rated 4 out of 5 stars"` → `4`. PySpark aplica esta operación en paralelo sobre todas las particiones del DataFrame.

### Limpieza de texto (en cascada)

```python
# 1. Minúsculas          "Great Product" → "great product"
# 2. Eliminar HTML       "Good<br>product" → "Good product"
# 3. Solo alfanumérico   "best!!! ever..." → "best  ever"
# 4. Espacios múltiples  "best  ever" → "best ever"
# 5. Trim                " best ever " → "best ever"
```

Cada paso usa `F.regexp_replace()` que ejecuta la regex en el executor de Spark (no en el driver), garantizando escalabilidad.

### Combinación título + cuerpo

```python
df = df.withColumn(
    "review_combined",
    F.concat_ws(" ", F.col("title_clean"), F.col("text_clean"))
)
```

El título condensa la opinión en pocas palabras ("Terrible experience", "Love it!") y tiene alta densidad informativa. Concatenarlo al cuerpo aprovecha ambas señales en un único vector de features.

---

## 6. Phase 3 — Entrenamiento del Modelo

**Archivo:** `src/phase3_train.py`

### División del dataset

```
Dataset limpio (~18.500 filas)
        │
        ├── 80% → df_train (~14.800 filas) → entrenamiento + CV
        └── 20% → df_test  (~3.700 filas)  → evaluación final (no visto)
```

La semilla aleatoria (`RANDOM_SEED=42`) garantiza que la división es idéntica en cada ejecución, haciendo los resultados 100% reproducibles.

### Pipeline de MLlib (5 etapas)

```
review_combined (string)
       │
       │ Tokenizer
       ▼
tokens: ["great", "product", "i", "love", "it"]
       │
       │ StopWordsRemover
       ▼
tokens_filtered: ["great", "product", "love"]
       │
       │ HashingTF  (numFeatures=65536)
       ▼
raw_features: SparseVector(65536, {12483: 1.0, 34521: 1.0, 58901: 1.0})
       │
       │ IDF
       ▼
features: SparseVector(65536, {12483: 2.31, 34521: 1.87, 58901: 3.14})
       │
       │ LogisticRegression
       ▼
prediction: 1.0  │  probability: [0.08, 0.92]
```

**Tokenizer:** divide el texto en palabras individuales por espacios. Simple y efectivo para inglés.

**StopWordsRemover:** elimina palabras funcionales (`the`, `is`, `and`, `a`, `to`...) que no aportan significado semántico pero ocuparían posiciones en el vector de features, diluyendo la señal.

**HashingTF:** convierte la lista de tokens en un vector de frecuencias de dimensión fija (65.536 = 2¹⁶). Más rápido que `CountVectorizer` porque no construye vocabulario: aplica una función hash directamente. La dimensión 65.536 minimiza colisiones para vocabularios de hasta ~50k palabras.

**IDF (Inverse Document Frequency):** pondera cada término según su rareza en el corpus. Un término que aparece en todas las reviews (como `amazon`) tendrá IDF bajo → menos peso. Un término que aparece solo en reviews negativas (como `broken`, `terrible`) tendrá IDF alto → más peso en la clasificación.

**LogisticRegression:** aprende un peso `w_i` para cada término del vocabulario. En inferencia computa `σ(Σ w_i · tfidf_i + b)` donde `σ` es la función sigmoide. El resultado es `P(positivo | texto)`.

### Cross-Validation con búsqueda de hiperparámetros

```
ParamGrid:
  regParam       = [0.01, 0.1]     ← penalización L2
  elasticNetParam = [0.0, 0.5]     ← mezcla L1/L2

Total: 4 combinaciones × 3 folds = 12 entrenamientos
Métrica de selección: AUC-ROC
```

**regParam (regularización L2):** penaliza coeficientes grandes para evitar sobreajuste. Un modelo sobreajustado memoriza las reviews del train pero falla en nuevas reseñas.

**elasticNetParam:** controla la mezcla entre L1 (produce modelos esparsos, muchos pesos = 0) y L2 (distribuye los pesos de forma pequeña entre todos los términos). `0.0` = L2 puro. `0.5` = mezcla igual.

---

## 7. Phase 4 — Evaluación y Resultados

**Archivo:** `src/phase4_evaluate.py`

### Métricas del modelo

> Resultados sobre 3.982 ejemplos de test (20% del dataset, nunca vistos durante el entrenamiento).

| Métrica | Valor | Interpretación |
|---|---|---|
| **Accuracy** | **94.32%** | El modelo acierta en 9 de cada 10 reviews |
| **Precision** | **93.08%** | De las predichas positivas, el 93% son realmente positivas |
| **Recall** | **86.27%** | Detecta el 86% de todas las reviews realmente positivas |
| **F1-Score** | **89.55%** | Balance sólido — supera el umbral de producción (>85%) |
| **AUC-ROC** | **97.74%** | Capacidad de discriminación excelente (umbral excelente >95%) |

**Distribución del test:** 2.860 negativos (71.8%) / 1.122 positivos (28.2%) — dataset desbalanceado donde la Accuracy sola sería engañosa; F1 y AUC-ROC son las métricas de referencia.

### Matriz de Confusión

![Matriz de Confusión](outputs/confusion_matrix.png)

La matriz de confusión desglosa las 4 categorías de predicción:

| | Predicho Negativo | Predicho Positivo |
|---|---|---|
| **Real Negativo** | ✅ Verdadero Negativo (TN) | ❌ Falso Positivo (FP) |
| **Real Positivo** | ❌ Falso Negativo (FN) | ✅ Verdadero Positivo (TP) |

**Lectura:** una columna derecha con pocos FP indica que el modelo no sobreclasifica como positivo. Una fila inferior con pocos FN indica que captura la mayoría de las reviews realmente positivas.

**Implicación práctica:** en un sistema de moderación de reviews, un FN (no detectar una review positiva) tiene bajo coste. Un FP (clasificar una review negativa como positiva) podría afectar la reputación. El threshold de 0.5 es ajustable según el caso de uso.

### Curva ROC

![Curva ROC](outputs/roc_curve.png)

La curva ROC muestra el tradeoff entre:
- **Eje Y (TPR / Recall):** proporción de reviews positivas correctamente identificadas
- **Eje X (FPR):** proporción de reviews negativas incorrectamente marcadas como positivas

**Cómo leer la gráfica:**
- La **línea diagonal roja** (AUC = 0.50) representa un clasificador aleatorio (flip de moneda)
- La **curva azul** del modelo debe estar lo más cerca posible al ángulo superior izquierdo
- El **área bajo la curva (AUC-ROC)** cuantifica esto: 1.0 = perfecto, 0.5 = aleatorio

Un AUC > 0.90 significa que, si tomamos una review positiva y una negativa al azar, el modelo asigna probabilidad más alta a la positiva en más del 90% de los casos.

### Distribución de Probabilidades

![Distribución de Probabilidades](outputs/probability_distribution.png)

Este histograma muestra cómo distribuye el modelo las probabilidades `P(positivo)` según la clase real:

- **Azul (realmente positivo):** idealmente concentrado cerca de 1.0
- **Rojo (realmente negativo):** idealmente concentrado cerca de 0.0
- **Línea negra punteada:** umbral de decisión en 0.5

**Qué revela:**
- **Separación clara** entre ambas distribuciones → modelo bien calibrado, alta confianza
- **Solapamiento en torno a 0.5** → zona de incertidumbre, ejemplos ambiguos (reviews con lenguaje mixto)
- Una distribución bimodal (concentrada en los extremos) indica que el modelo es decisivo

---

## 8. Decisiones Técnicas Justificadas

### ¿Por qué PySpark y no scikit-learn o Pandas?

| Criterio | Pandas + sklearn | PySpark MLlib |
|---|---|---|
| Escalabilidad | Limitada por RAM del driver | Distribuido: escala a TB |
| API | Más simple para datasets pequeños | Idéntica en local y en cluster |
| Portabilidad | Código distinto para producción | Mismo código en dev y prod |
| Velocidad (21k filas) | Más rápido | Overhead de JVM |

**Decisión:** PySpark. El portfolio demuestra habilidades aplicables a proyectos reales en empresas donde los datos no caben en una sola máquina. El overhead en 21k filas es aceptable.

### ¿Por qué clasificación binaria en lugar de multiclase 1-5?

- **Multiclase (1-5):** el modelo debe aprender diferencias sutiles entre "2 estrellas" y "3 estrellas" que son extremadamente difíciles de distinguir por el texto solo
- **Binario (positivo/negativo):** señal clara, lenguaje consistente por clase, mejor F1
- **Rating 3 excluido:** reviews ambiguas son ruido, no señal

### ¿Por qué TF-IDF y no embeddings (Word2Vec, BERT)?

| Enfoque | Pros | Contras |
|---|---|---|
| TF-IDF + LogReg | Interpretable, rápido, buena baseline | No captura contexto semántico |
| Word2Vec | Contexto semántico, vectores densos | Requiere corpus grande |
| BERT fine-tuning | Estado del arte | Requiere GPU, no nativo en PySpark |

**Decisión:** TF-IDF + LogReg como baseline robusta. En análisis de sentimiento con ratings claros, esta combinación rara vez es superada significativamente por modelos más complejos. Es la elección correcta cuando el objetivo es demostrar dominio del proceso, no competir en un benchmark.

### ¿Por qué Cross-Validation y no un único train/validation split?

Un único split puede resultar en una partición de validación con distribución de clases atípica, generando una estimación del rendimiento demasiado optimista o pesimista. La CV de k-folds evalúa el modelo en k particiones distintas y promedia, produciendo una estimación más robusta y honesta del rendimiento.

### ¿Por qué HashingTF en lugar de CountVectorizer?

- **CountVectorizer:** construye el vocabulario completo en el paso de `fit()`, requiere un pase completo por los datos y mantiene el vocabulario en memoria
- **HashingTF:** aplica una función hash a cada token → no necesita vocabulario, es más rápido y constante en memoria. A costa de colisiones (dos tokens distintos pueden tener el mismo hash), mitigadas con `numFeatures=65536`

### ¿Por qué Docker y no un entorno virtual local?

- **Reproducibilidad garantizada:** cualquier revisor del portfolio puede ejecutar exactamente el mismo entorno con un único comando
- **Independencia del OS del host:** el pipeline corre en Linux independientemente de Windows/Mac del host
- **Variables de entorno externalizadas:** los parámetros del modelo se configuran en `docker-compose.yml` sin tocar el código
- **Portabilidad:** el mismo `docker-compose.yml` funciona en local, en CI/CD y en la nube

---

## 9. Cómo Reproducir el Proyecto

### Requisitos

- Docker Desktop ≥ 24.0 en modo Linux containers
- 4 GB de RAM disponibles
- El archivo `data/Amazon_Reviews.csv` en la raíz del proyecto

### Ejecución

```bash
# 1. Clonar / descomprimir el proyecto
cd etl_project

# 2. Lanzar el pipeline completo (primera vez: construye la imagen ~5-10 min)
docker-compose up --build

# 3. Ejecuciones posteriores (imagen ya construida: ~2-5 min)
docker-compose up

# 4. Ver los resultados generados
ls outputs/
#   confusion_matrix.png
#   roc_curve.png
#   probability_distribution.png
#   metrics.json
#   pipeline.log

# 5. Limpiar contenedores
docker-compose down
```

### Ajustar parámetros del modelo

Edita las variables de entorno en `docker-compose.yml` sin tocar el código:

```yaml
environment:
  - POSITIVE_RATING_THRESHOLD=4   # Cambiar a 5 para umbral más estricto
  - NEGATIVE_RATING_THRESHOLD=2   # Cambiar a 3 para incluir rating 3 como negativo
  - TEST_SPLIT_RATIO=0.2           # Cambiar a 0.3 para test más grande
  - CV_FOLDS=5                     # Más folds = estimación más robusta, más lento
  - RANDOM_SEED=42                 # Cambiar para explorar variabilidad estadística
```

### Usar el modelo entrenado en nuevas predicciones

```python
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").getOrCreate()
model = PipelineModel.load("./models/sentiment_model")

# DataFrame con la misma columna que espera el modelo
new_reviews = spark.createDataFrame([
    ("This product is absolutely amazing, best purchase ever!",),
    ("Terrible quality, broke after two days, total waste of money.",),
], ["review_combined"])

predictions = model.transform(new_reviews)
predictions.select("review_combined", "prediction", "probability").show(truncate=50)
```

---

## Estructura de Archivos

```
etl_project/
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh          ← detecta JAVA_HOME dinámicamente
├── requirements.txt
├── run_pipeline.py               ← orquestador de las 4 fases
├── README.md                     ← guía rápida de uso
├── TECHNICAL_REPORT.md           ← este documento
├── data/
│   └── Amazon_Reviews.csv
├── src/
│   ├── __init__.py
│   ├── phase1_ingest.py
│   ├── phase2_transform.py
│   ├── phase3_train.py
│   └── phase4_evaluate.py
├── models/
│   └── sentiment_model/          ← generado por docker-compose up
└── outputs/
    ├── metrics.json              ← generado por docker-compose up
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── probability_distribution.png
```
