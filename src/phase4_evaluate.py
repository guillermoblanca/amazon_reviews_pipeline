"""
=============================================================================
PHASE 4 — Evaluación del Modelo y Generación de Reporte
=============================================================================
Responsabilidad:
  - Aplicar el modelo entrenado al conjunto de test (datos no vistos)
  - Calcular métricas de clasificación completas:
      · Accuracy   : proporción de predicciones correctas
      · Precision  : de los predichos positivos, ¿cuántos son realmente positivos?
      · Recall     : de los realmente positivos, ¿cuántos fueron detectados?
      · F1-Score   : media armónica de Precision y Recall
      · AUC-ROC    : área bajo la curva ROC (mide separabilidad de clases)
  - Generar visualizaciones:
      · Matriz de confusión  : muestra errores tipo I (FP) y tipo II (FN)
      · Curva ROC            : relación entre tasa de verdaderos/falsos positivos
      · Distribución de probabilidades: cómo de seguro está el modelo
  - Guardar el reporte en /app/outputs para acceso externo al contenedor

Interpretación de métricas:
  · Un F1-Score > 0.85 indica un modelo sólido para producción.
  · AUC-ROC > 0.90 indica excelente capacidad de discriminación.
  · La matriz de confusión revela si el modelo falla más en falsos positivos
    o falsos negativos, lo que orienta mejoras futuras.
=============================================================================
"""

import json
import os
from pathlib import Path
from loguru import logger

import matplotlib
matplotlib.use("Agg")  # Backend sin display (necesario en Docker sin GUI)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)


OUTPUTS_DIR = Path("/app/outputs")


def evaluate_model(model: PipelineModel, df_test: DataFrame) -> dict:
    """
    Orquesta la evaluación completa: predicciones → métricas → visualizaciones → reporte.

    Args:
        model:   PipelineModel entrenado (salida de Phase 3).
        df_test: DataFrame de test (20% del dataset, no visto durante entrenamiento).

    Returns:
        Diccionario con todas las métricas calculadas (también guardado en JSON).
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions = _generate_predictions(model, df_test)
    y_true, y_pred, y_prob = _extract_arrays(predictions)
    metrics = _compute_metrics(y_true, y_pred, y_prob)
    _save_metrics_json(metrics)
    _plot_confusion_matrix(y_true, y_pred)
    _plot_roc_curve(y_true, y_prob)
    _plot_probability_distribution(predictions)
    _print_final_report(metrics)

    return metrics


def _generate_predictions(model: PipelineModel, df_test: DataFrame) -> DataFrame:
    """
    Aplica el modelo al conjunto de test y genera las columnas de predicción.

    PySpark MLlib añade automáticamente tres columnas al DataFrame:
      · rawPrediction: puntuaciones antes de aplicar la función sigmoide
      · probability:   vector [P(negativo), P(positivo)] para cada fila
      · prediction:    clase predicha (0.0 o 1.0) basada en umbral 0.5

    Args:
        model:   PipelineModel entrenado.
        df_test: Datos de test con columnas 'review_combined' y 'label'.

    Returns:
        DataFrame con columnas adicionales: rawPrediction, probability, prediction.
    """
    logger.info("Generando predicciones sobre el conjunto de test...")
    predictions = model.transform(df_test)

    # Extrae la probabilidad de la clase positiva (índice 1 del vector)
    extract_prob = F.udf(lambda v: float(v[1]))
    predictions = predictions.withColumn("prob_positive", extract_prob(F.col("probability")))

    total = predictions.count()
    logger.info(f"Predicciones generadas para {total:,} ejemplos de test.")
    return predictions


def _extract_arrays(predictions: DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convierte las columnas de predicción de PySpark a arrays NumPy.

    PySpark opera de forma distribuida en el cluster; para calcular métricas
    con scikit-learn es necesario traer los datos al driver (collect()).
    Esto es seguro porque el conjunto de test es una fracción pequeña del dataset.

    Args:
        predictions: DataFrame con columnas 'label', 'prediction', 'prob_positive'.

    Returns:
        Tupla de arrays NumPy: (y_true, y_pred, y_prob).
    """
    logger.info("Recolectando predicciones en el driver para calcular métricas...")

    rows = predictions.select("label", "prediction", "prob_positive").collect()

    y_true = np.array([int(r["label"]) for r in rows])
    y_pred = np.array([int(r["prediction"]) for r in rows])
    y_prob = np.array([float(r["prob_positive"]) for r in rows])

    return y_true, y_pred, y_prob


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> dict:
    """
    Calcula el conjunto completo de métricas de clasificación.

    Explicación de cada métrica:
      · Accuracy:  (TP + TN) / Total — fácil de entender pero engañosa en datasets
                   desbalanceados (un modelo que siempre predice "positivo" tendría
                   accuracy alta si el 90% son positivos).
      · Precision: TP / (TP + FP) — de las reviews marcadas como positivas,
                   ¿cuántas realmente lo son? Crucial si el coste de FP es alto.
      · Recall:    TP / (TP + FN) — de las reviews realmente positivas,
                   ¿cuántas detectamos? Crucial si el coste de FN es alto.
      · F1-Score:  2 * (P * R) / (P + R) — balance entre Precision y Recall.
                   Métrica principal para comparar modelos en texto desbalanceado.
      · AUC-ROC:   Probabilidad de que el modelo clasifique un positivo aleatorio
                   más alto que un negativo aleatorio. No depende del umbral.

    Args:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas (umbral 0.5).
        y_prob: Probabilidades de clase positiva.

    Returns:
        Diccionario con todas las métricas como floats.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    metrics = {
        "accuracy":        round(float(accuracy_score(y_true, y_pred)), 4),
        "precision":       round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":          round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc_roc":         round(float(roc_auc), 4),
        "test_samples":    int(len(y_true)),
        "positive_samples": int(np.sum(y_true == 1)),
        "negative_samples": int(np.sum(y_true == 0)),
    }

    return metrics


def _save_metrics_json(metrics: dict) -> None:
    """
    Persiste las métricas en formato JSON estructurado.

    El archivo JSON permite:
      · Comparar resultados entre distintas ejecuciones del pipeline
      · Integrarse con dashboards o herramientas de MLOps
      · Ser leído programáticamente para tests de regresión del modelo

    Args:
        metrics: Diccionario de métricas calculadas.
    """
    output_path = OUTPUTS_DIR / "metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Métricas guardadas en: {output_path}")


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Genera y guarda la matriz de confusión como imagen PNG.

    La matriz de confusión muestra en detalle cómo falla el modelo:
      · Verdaderos Positivos (TP): reviews positivas correctamente identificadas
      · Verdaderos Negativos (TN): reviews negativas correctamente identificadas
      · Falsos Positivos (FP):     reviews negativas clasificadas como positivas
      · Falsos Negativos (FN):     reviews positivas clasificadas como negativas

    Un exceso de FP (columna derecha, fila superior) indica que el modelo es
    demasiado optimista. Un exceso de FN (columna izquierda, fila inferior)
    indica que pierde reviews positivas reales.

    Args:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Negativo (0)", "Positivo (1)"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5
    )
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Valor Real", fontsize=12)
    ax.set_title("Matriz de Confusión — Análisis de Sentimiento Amazon Reviews", fontsize=13)

    path = OUTPUTS_DIR / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Matriz de confusión guardada en: {path}")


def _plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """
    Genera y guarda la curva ROC (Receiver Operating Characteristic).

    La curva ROC muestra el tradeoff entre:
      · Tasa de Verdaderos Positivos (TPR = Recall) en el eje Y
      · Tasa de Falsos Positivos (FPR) en el eje X
    para todos los posibles umbrales de clasificación.

    La diagonal punteada representa un clasificador aleatorio (AUC = 0.5).
    Cuanto más se acerca la curva al ángulo superior izquierdo, mejor el modelo.

    Args:
        y_true: Etiquetas reales.
        y_prob: Probabilidades de clase positiva.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2E86AB", lw=2, label=f"Modelo (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#E84855", lw=1.5, linestyle="--", label="Clasificador aleatorio")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2E86AB")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
    ax.set_title("Curva ROC — Análisis de Sentimiento Amazon Reviews", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    path = OUTPUTS_DIR / "roc_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Curva ROC guardada en: {path}")


def _plot_probability_distribution(predictions: DataFrame) -> None:
    """
    Genera un histograma de las probabilidades predichas separado por clase real.

    Esta visualización revela la calibración del modelo:
      · Un modelo bien calibrado separa claramente las distribuciones:
        la clase positiva acumula probabilidades > 0.7 y la negativa < 0.3.
      · Solapamiento alto en torno a 0.5 indica ejemplos ambiguos o ruido
        en los datos (reviews con lenguaje mixto independientemente del rating).

    Args:
        predictions: DataFrame PySpark con columnas 'label' y 'prob_positive'.
    """
    # Recolectar en driver solo las dos columnas necesarias
    rows = predictions.select("label", "prob_positive").collect()
    labels = np.array([int(r["label"]) for r in rows])
    probs = np.array([float(r["prob_positive"]) for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(probs[labels == 1], bins=50, alpha=0.6, color="#2E86AB", label="Positivo (real)")
    ax.hist(probs[labels == 0], bins=50, alpha=0.6, color="#E84855", label="Negativo (real)")
    ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Umbral de decisión (0.5)")
    ax.set_xlabel("Probabilidad predicha (P = Positivo)", fontsize=12)
    ax.set_ylabel("Número de reviews", fontsize=12)
    ax.set_title("Distribución de Probabilidades por Clase Real", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    path = OUTPUTS_DIR / "probability_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Distribución de probabilidades guardada en: {path}")


def _print_final_report(metrics: dict) -> None:
    """
    Imprime un resumen legible del rendimiento del modelo al finalizar el pipeline.

    Args:
        metrics: Diccionario con todas las métricas calculadas.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("  REPORTE FINAL — MODELO DE SENTIMIENTO")
    logger.info("=" * 60)
    logger.info(f"  Samples de test    : {metrics['test_samples']:,}")
    logger.info(f"  Positivos reales   : {metrics['positive_samples']:,}")
    logger.info(f"  Negativos reales   : {metrics['negative_samples']:,}")
    logger.info("  ─" * 30)
    logger.info(f"  Accuracy           : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    logger.info(f"  Precision          : {metrics['precision']:.4f}")
    logger.info(f"  Recall             : {metrics['recall']:.4f}")
    logger.info(f"  F1-Score           : {metrics['f1_score']:.4f}")
    logger.info(f"  AUC-ROC            : {metrics['auc_roc']:.4f}")
    logger.info("  ─" * 30)

    # Interpretación automática
    f1 = metrics["f1_score"]
    auc_val = metrics["auc_roc"]
    if f1 >= 0.90 and auc_val >= 0.95:
        verdict = "EXCELENTE — Listo para producción"
    elif f1 >= 0.80 and auc_val >= 0.85:
        verdict = "BUENO — Sólido para uso en producción"
    elif f1 >= 0.70:
        verdict = "ACEPTABLE — Requiere mejoras antes de producción"
    else:
        verdict = "DEFICIENTE — Revisar pipeline y datos"

    logger.info(f"  Veredicto          : {verdict}")
    logger.info("=" * 60)
    logger.info(f"  Artefactos generados en /app/outputs/:")
    logger.info(f"    · metrics.json")
    logger.info(f"    · confusion_matrix.png")
    logger.info(f"    · roc_curve.png")
    logger.info(f"    · probability_distribution.png")
    logger.info("=" * 60)
