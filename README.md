# Amazon Reviews — Sentiment Analysis Pipeline

Pipeline de análisis de sentimiento end-to-end sobre ~21.000 reseñas de Amazon, construido con **PySpark MLlib** y empaquetado en **Docker** para ejecutarse con un único comando.

**Stack:** Python 3.11 · PySpark 3.5 · MLlib · Docker · TF-IDF · Logistic Regression

---

## Resultados del Modelo

| Métrica | Resultado | Interpretación |
|---|---|---|
| **Accuracy** | **94.32%** | Acierta en 9 de cada 10 reviews |
| **F1-Score** | **89.55%** | Supera el umbral de producción (> 85%) |
| **AUC-ROC** | **97.74%** | Discriminación excelente entre clases |
| **Precision** | **93.08%** | Baja tasa de falsos positivos |
| **Recall** | **86.27%** | Detecta el 86% de las reviews positivas reales |

> Evaluado sobre **3.982 ejemplos de test** nunca vistos durante el entrenamiento.

---

## Ejecución

```bash
docker-compose up --build
```

Primera ejecución: ~5–10 min (construye la imagen). Posteriores: ~2–5 min.

Los resultados quedan en el directorio local `./outputs/` y el modelo en `./models/`.

```bash
docker-compose down   # limpia el contenedor al terminar
```

---

## Índice

1. [Arquitectura del Pipeline](#1-arquitectura-del-pipeline)
2. [Phase 1 — Ingesta](#2-phase-1--ingesta)
3. [Phase 2 — Transformación y Feature Engineering](#3-phase-2--transformación-y-feature-engineering)
4. [Phase 3 — Entrenamiento del Modelo](#4-phase-3--entrenamiento-del-modelo)
5. [Phase 5 — Evaluación y Resultados](#5-phase-4--evaluación-y-resultados)
6. [Decisiones Técnicas](#6-decisiones-técnicas)
7. [Configuración](#7-configuración)
8. [Requisitos y Estructura](#8-requisitos-y-estructura)

---

## 1. Arquitectura del Pipeline

```
CSV (Amazon Reviews ~21k filas)
             │
             ▼
┌────────────────────────┐
│  PHASE 1 — Ingesta     │  · SparkSession local[*]
│  phase1_ingest.py      │  · Validación de esquema (fail-fast)
│                        │  · Estadísticas del dataset crudo
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  PHASE 2 — Transform   │  · Parseo de rating (regex)
│  phase2_transform.py   │  · Etiquetado binario (≥4=pos, ≤2=neg, 3=excluido)
│                        │  · Limpieza de texto + combinación título+cuerpo
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  PHASE 3 — Train       │  · Split 80/20 train/test
│  phase3_train.py       │  · Pipeline: Tokenizer→StopWords→HashingTF→IDF→LogReg
│                        │  · Cross-Validation 3-fold + búsqueda de hiperparámetros
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  PHASE 4 — Evaluate    │  · F1, AUC-ROC, Precision, Recall, Accuracy
│  phase4_evaluate.py    │  · Matriz de confusión · Curva ROC
│                        │  · Distribución de probabilidades · metrics.json
└───────────┬────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
 ./outputs/     ./models/
 (gráficos      (modelo PySpark
  + métricas)    serializado)
```

---

## 2. Phase 1 — Ingesta

**Archivo:** `src/phase1_ingest.py`

Lee el CSV con PySpark tolerando líneas malformadas, valida el esquema y registra estadísticas del dataset crudo antes de cualquier transformación.

### SparkSession en modo local

```python
spark = (
    SparkSession.builder
    .master("local[*]")          # Usa todos los cores del host
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
```

`local[*]` ejecuta Spark en una sola máquina usando múltiples threads. La API es idéntica a la de un cluster real: el mismo código escala cambiando solo el parámetro `SPARK_MASTER` en `docker-compose.yml`.

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

`multiLine=true` es crítico porque los textos de review frecuentemente contienen saltos de línea que romperían un lector CSV estándar. `PERMISSIVE` evita que una fila malformada detenga toda la ingesta.

### Validación fail-fast

Antes de continuar al pipeline, se verifica que `Rating`, `Review Text` y `Review Title` existen. Un error temprano con mensaje claro es preferible a un fallo críptico en la fase de entrenamiento.

---

## 3. Phase 2 — Transformación y Feature Engineering

**Archivo:** `src/phase2_transform.py`

### Flujo completo

```
"Rated 4 out of 5 stars"  →  rating_numeric: 4  →  label: 1 (Positivo)

"Great product!           →  title_clean: "great product"
 I really love it..."        text_clean:  "i really love it"
                          →  review_combined: "great product i really love it"
```

### Regla de etiquetado binario

| Rating | Clase | Label |
|---|---|---|
| ⭐⭐⭐⭐⭐ (5) y ⭐⭐⭐⭐ (4) | Positivo | `1` |
| ⭐⭐⭐ (3) | **Excluido** | `null` |
| ⭐⭐ (2) y ⭐ (1) | Negativo | `0` |

Las reviews de 3 estrellas se excluyen porque son semánticamente ambiguas — el mismo texto puede contener "buen producto pero entrega horrible". Incluirlas introduce ruido que reduce F1 sin aportar señal discriminativa limpia.

### Parseo del rating con expresión regular

El campo crudo tiene formato `"Rated 4 out of 5 stars"`. Se extrae el dígito con regex en paralelo sobre todas las particiones:

```python
F.regexp_extract(F.col("Rating"), r"(\d)", 1).cast(IntegerType())
```

### Limpieza de texto (5 pasos en cascada)

```python
# 1. Minúsculas          "Great Product!" → "great product!"
# 2. Eliminar HTML       "Good<br>value"  → "good value"
# 3. Solo alfanumérico   "best!!! ever"   → "best  ever"
# 4. Colapsar espacios   "best  ever"     → "best ever"
# 5. Trim                " best ever "    → "best ever"
```

Cada paso usa `F.regexp_replace()`, que ejecuta la regex en los executors de Spark (no en el driver), manteniendo la escalabilidad distribuida.

### Combinación título + cuerpo

```python
F.concat_ws(" ", F.col("title_clean"), F.col("text_clean"))
```

El título resume la opinión en pocas palabras (`"Terrible experience"`, `"Love it!"`) y tiene densidad informativa alta. Concatenarlo al cuerpo aprovecha ambas señales en un único vector de features para el modelo.

---

## 4. Phase 3 — Entrenamiento del Modelo

**Archivo:** `src/phase3_train.py`

### División del dataset

```
~18.500 filas limpias
    ├── 80% → df_train (~14.800) → entrenamiento + cross-validation
    └── 20% → df_test  (~3.700)  → evaluación final (no visto)
```

La semilla `RANDOM_SEED=42` garantiza reproducibilidad exacta entre ejecuciones.

### Pipeline de MLlib (5 etapas encadenadas)

Un `Pipeline` de PySpark agrupa transformadores y estimadores en secuencia. Garantiza que las mismas transformaciones se aplican de forma idéntica a train y test, previniendo data leakage.

```
review_combined (texto limpio)
       │
       │  Tokenizer
       ▼
["great", "product", "i", "love", "it"]
       │
       │  StopWordsRemover
       ▼
["great", "product", "love"]          ← elimina "i" (stopword)
       │
       │  HashingTF  (numFeatures=65.536)
       ▼
SparseVector(65536, {12483: 1.0, 34521: 1.0, 58901: 1.0})
       │
       │  IDF
       ▼
SparseVector(65536, {12483: 2.31, 34521: 1.87, 58901: 3.14})
       │
       │  LogisticRegression
       ▼
prediction: 1.0  │  probability: [0.08, 0.92]
```

**HashingTF** convierte tokens en un vector de frecuencias de dimensión fija sin necesidad de construir un vocabulario completo. `numFeatures=65.536` (2¹⁶) minimiza colisiones de hash para vocabularios de hasta ~50k palabras.

**IDF** pondera cada término por su rareza en el corpus. Un término como `amazon` aparece en casi todas las reviews → IDF bajo → poco peso. Un término como `broken` aparece solo en reviews negativas → IDF alto → peso discriminativo alto.

### Cross-Validation con búsqueda de hiperparámetros

```
ParamGrid explorado:
  regParam        = [0.01, 0.1]    ← penalización L2
  elasticNetParam = [0.0,  0.5]    ← mezcla L1/L2

Total: 4 combinaciones × 3 folds = 12 entrenamientos
Métrica de selección: AUC-ROC
```

La cross-validation evalúa cada combinación en 3 particiones distintas del training set y promedia los resultados, produciendo una estimación robusta que no sobreajusta al conjunto de validación.

---

## 5. Phase 4 — Evaluación y Resultados

**Archivo:** `src/phase4_evaluate.py`

### Matriz de Confusión

![Matriz de Confusión](outputs/confusion_matrix.png)

Desglosa las 4 categorías de predicción sobre el conjunto de test:

| | Predicho Negativo | Predicho Positivo |
|---|---|---|
| **Real Negativo** | ✅ Verdadero Negativo (TN) | ❌ Falso Positivo (FP) |
| **Real Positivo** | ❌ Falso Negativo (FN) | ✅ Verdadero Positivo (TP) |

En un sistema de análisis de reviews, un **FP** (review negativa clasificada como positiva) tiene mayor coste reputacional que un **FN**. El umbral de decisión de 0.5 es ajustable en función del caso de uso sin reentrenar el modelo.

### Curva ROC

![Curva ROC](outputs/roc_curve.png)

Muestra el tradeoff entre tasa de verdaderos positivos (TPR) y tasa de falsos positivos (FPR) para todos los umbrales de clasificación posibles.

- La **línea diagonal** representa un clasificador aleatorio (AUC = 0.50)
- La **curva azul** del modelo (AUC = **0.9774**) se acerca al ángulo superior izquierdo — discriminación excelente
- Un AUC > 0.97 significa que, tomando una review positiva y una negativa al azar, el modelo asigna mayor probabilidad a la positiva en el 97% de los casos

### Distribución de Probabilidades

![Distribución de Probabilidades](outputs/probability_distribution.png)

Histograma de `P(positivo)` separado por clase real. Revela la calibración del modelo:

- **Azul (realmente positivo):** concentrado cerca de 1.0 → el modelo es seguro al clasificar positivos
- **Rojo (realmente negativo):** concentrado cerca de 0.0 → el modelo es seguro al clasificar negativos
- **Solapamiento en torno a 0.5:** zona de incertidumbre — reviews con lenguaje mixto independientemente de su rating

La separación clara entre ambas distribuciones confirma que el modelo está bien calibrado y es decisivo.

---

## 6. Decisiones Técnicas

### ¿Por qué PySpark y no Pandas + scikit-learn?

| Criterio | Pandas + sklearn | PySpark MLlib |
|---|---|---|
| Escalabilidad | Limitada por RAM del driver | Distribuido: escala a TB |
| API | Más simple para datasets pequeños | Idéntica en local y en cluster |
| Portabilidad | Código diferente para producción | Mismo código en dev y prod |

El dataset de 21k filas cabría en memoria, pero la arquitectura es idéntica a la de millones de reviews en un cluster real. El pipeline escala sin cambiar una sola línea de código.

### ¿Por qué TF-IDF y no embeddings (Word2Vec, BERT)?

| Enfoque | Pros | Contras |
|---|---|---|
| **TF-IDF + LogReg** | Interpretable, rápido, baseline sólida | No captura contexto semántico |
| Word2Vec | Contexto semántico, vectores densos | Requiere corpus grande |
| BERT fine-tuning | Estado del arte | Requiere GPU, no nativo en PySpark |

En clasificación de sentimiento con ratings como etiqueta (señal fuerte y limpia), TF-IDF + Logistic Regression rara vez es superado de forma significativa por modelos más complejos. Es la elección correcta para demostrar dominio del proceso sin ocultar la lógica en una caja negra.

### ¿Por qué Cross-Validation y no un único split de validación?

Un único split puede producir una partición de validación con distribución de clases atípica, generando una estimación del rendimiento demasiado optimista o pesimista. La CV de k-folds evalúa el modelo en k particiones distintas y promedia, produciendo una estimación robusta y honesta.

### ¿Por qué HashingTF y no CountVectorizer?

`CountVectorizer` construye el vocabulario completo en el paso `fit()`, requiere un pase completo por los datos y mantiene el vocabulario en memoria. `HashingTF` aplica una función hash directamente sobre cada token, sin vocabulario, con coste de memoria constante. Las colisiones se mitigan con `numFeatures=65.536`.

---

## 7. Configuración

Todos los parámetros del modelo se externalizan en `docker-compose.yml` sin tocar el código:

| Variable | Default | Descripción |
|---|---|---|
| `DATA_PATH` | `/app/data/Amazon_Reviews.csv` | Ruta al dataset dentro del contenedor |
| `SPARK_MASTER` | `local[*]` | Modo de ejecución de Spark |
| `POSITIVE_RATING_THRESHOLD` | `4` | Rating mínimo para etiqueta positiva |
| `NEGATIVE_RATING_THRESHOLD` | `2` | Rating máximo para etiqueta negativa |
| `TEST_SPLIT_RATIO` | `0.2` | Fracción del dataset reservada para test |
| `RANDOM_SEED` | `42` | Semilla aleatoria para reproducibilidad |
| `CV_FOLDS` | `3` | Número de folds en cross-validation |

### Usar el modelo en nuevas predicciones

```python
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").getOrCreate()
model = PipelineModel.load("./models/sentiment_model")

new_reviews = spark.createDataFrame([
    ("This product is absolutely amazing, best purchase ever!",),
    ("Terrible quality, broke after two days, total waste of money.",),
], ["review_combined"])

predictions = model.transform(new_reviews)
predictions.select("review_combined", "prediction", "probability").show(truncate=60)
```

---

## 8. Requisitos y Estructura

### Requisitos

- [Docker Desktop](https://www.docker.com/get-started) ≥ 24.0 en modo **Linux containers**
- Docker Compose ≥ 2.0
- RAM disponible: mínimo 4 GB

### Estructura del Proyecto

```
etl_project/
├── .gitignore
├── Dockerfile                        # Python 3.11 + OpenJDK 17
├── docker-compose.yml                # Configuración de servicios y volúmenes
├── docker-entrypoint.sh              # Detecta JAVA_HOME dinámicamente
├── requirements.txt                  # PySpark, sklearn, matplotlib, seaborn
├── run_pipeline.py                   # Orquestador: ejecuta las 4 fases
├── data/
│   └── Amazon_Reviews.csv
├── src/
│   ├── phase1_ingest.py              # Ingesta con PySpark
│   ├── phase2_transform.py           # Transformación y etiquetado
│   ├── phase3_train.py               # Entrenamiento con MLlib
│   └── phase4_evaluate.py            # Evaluación y visualizaciones
├── models/
│   └── sentiment_model/              # Modelo PySpark serializado (generado)
└── outputs/
    ├── metrics.json                  # F1, AUC, Precision, Recall, Accuracy
    ├── confusion_matrix.png          # Matriz de confusión
    ├── roc_curve.png                 # Curva ROC
    ├── probability_distribution.png  # Calibración del modelo
    └── pipeline.log                  # Log completo de ejecución
```

### Tecnologías

| Tecnología | Versión | Rol |
|---|---|---|
| Python | 3.11 | Lenguaje del pipeline |
| PySpark | 3.5.1 | Procesamiento distribuido + MLlib |
| OpenJDK | 17 | Runtime de Java para Spark |
| scikit-learn | 1.5.0 | Métricas de evaluación |
| Matplotlib / Seaborn | 3.9 / 0.13 | Visualizaciones |
| Docker | ≥ 24 | Contenedorización y reproducibilidad |
