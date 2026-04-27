"""
gradio_app/app.py — Demo interactiva del modelo de análisis de sentimiento
"""

import gradio as gr
import pandas as pd

from inference import load_model, predict_batch, predict_single

load_model()

# ── Ejemplos pre-cargados ────────────────────────────────────────────────────
EXAMPLES = [
    [
        "Best laptop I have ever owned",
        "Fast, reliable, and the battery lasts all day. Completely worth the price. "
        "Setup was easy and performance is outstanding even for demanding tasks.",
    ],
    [
        "Complete waste of money",
        "This product broke after just two days of normal use. The build quality is "
        "terrible. Customer service ignored my refund request for three weeks. Avoid.",
    ],
    [
        "Great value for the price",
        "Works exactly as described. Delivery was fast and packaging was solid. "
        "I was a bit hesitant but I am very happy with this purchase. Recommended.",
    ],
    [
        "Extremely disappointed",
        "Nothing works as advertised. The instructions are incomprehensible and "
        "customer support was useless. This is my worst online purchase ever.",
    ],
    [
        "Outstanding quality",
        "The craftsmanship is remarkable. You can tell real care went into making it. "
        "Arrived in perfect condition. Very satisfied customer, will buy again.",
    ],
    [
        "",
        "Terrible. Just terrible. Broken on arrival and the seller won't answer.",
    ],
]


# ── Callbacks ────────────────────────────────────────────────────────────────

def analyze_single(title: str, text: str):
    """
    Retorna valores directos (sin gr.update) para compatibilidad con Gradio 4.x.
    gr.Label acepta un dict {label: confidence} para mostrar barras de probabilidad.
    """
    if not text or not text.strip():
        # Retorna None para el Label (lo vacía) y un mensaje de aviso para el Markdown
        return None, "⚠️ Please enter a review text."

    result = predict_single(title or "", text)

    emoji = "✅ POSITIVE" if result["label"] == 1 else "❌ NEGATIVE"
    confidence_pct = f"{result['confidence'] * 100:.1f}%"
    verdict = f"**{emoji}** — Confidence: {confidence_pct}"

    # gr.Label espera {str: float} — las claves deben ser strings simples
    proba_dict = {
        "Positive": float(result["probabilities"]["Positive ⭐"]),
        "Negative": float(result["probabilities"]["Negative ❌"]),
    }

    return proba_dict, verdict


def analyze_batch(raw_text: str):
    """Clasifica múltiples reviews (una por línea) y devuelve tabla + resumen."""
    lines = [line.strip() for line in (raw_text or "").splitlines() if line.strip()]

    if not lines:
        return pd.DataFrame(columns=["Review", "Sentiment", "Confidence"]), ""

    reviews = [{"title": "", "text": line} for line in lines]
    results = predict_batch(reviews)

    rows = []
    for line, res in zip(lines, results):
        emoji = "✅ Positive" if res["label"] == 1 else "❌ Negative"
        rows.append({
            "Review": (line[:120] + "…") if len(line) > 120 else line,
            "Sentiment": emoji,
            "Confidence": f"{res['confidence'] * 100:.1f}%",
        })

    pos = sum(1 for r in results if r["label"] == 1)
    neg = len(results) - pos
    summary = (
        f"**{len(results)} reviews analyzed** — "
        f"✅ {pos} positive ({pos / len(results) * 100:.0f}%)  "
        f"❌ {neg} negative ({neg / len(results) * 100:.0f}%)"
    )

    return pd.DataFrame(rows), summary


# ── Textos ───────────────────────────────────────────────────────────────────

HEADER = """
# 🛍️ Amazon Reviews — Sentiment Analysis

Model trained on ~21,000 Amazon reviews · **TF-IDF + Logistic Regression** · PySpark MLlib

| Accuracy | F1-Score | AUC-ROC | Precision | Recall |
|:---:|:---:|:---:|:---:|:---:|
| 94.32% | 89.55% | 97.74% | 93.08% | 86.27% |

*Evaluated on 3,982 held-out examples never seen during training.*
"""

ABOUT = """
### How it works

**Training pipeline (PySpark MLlib):**
1. **Phase 1 — Ingestion:** Load CSV, validate schema, log dataset stats
2. **Phase 2 — Transform:** Parse rating (`"Rated 4 out of 5 stars"` → `4`), binary labeling
   (≥ 4★ = positive · ≤ 2★ = negative · 3★ excluded as ambiguous), text cleaning
3. **Phase 3 — Train:** TF-IDF + Logistic Regression with 3-fold Cross-Validation
4. **Phase 4 — Evaluate:** Accuracy, F1, AUC-ROC, confusion matrix, ROC curve

**Export for serving (this demo):**
The PySpark model (requires JVM) is re-exported as a scikit-learn pipeline
(`TfidfVectorizer` + `LogisticRegression`) serialized with joblib.
This is the standard enterprise pattern: *train at scale → serve lightweight*.

**sklearn equivalences:**
- `TfidfVectorizer(sublinear_tf=True, min_df=2)` ≈ `HashingTF + IDF`
- `stop_words='english'` ≈ `StopWordsRemover`
- `LogisticRegression(C=10)` ≈ `regParam=0.1` from cross-validation

**Tech stack:** Python 3.11 · PySpark 3.5 · scikit-learn · FastAPI · Docker · Gradio
"""


# ── Interfaz ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Amazon Reviews Sentiment", theme=gr.themes.Soft()) as demo:

    gr.Markdown(HEADER)

    with gr.Tabs():

        # ── Single review ────────────────────────────────────────────────────
        with gr.Tab("🔍 Single Review"):
            with gr.Row():
                with gr.Column(scale=3):
                    title_input = gr.Textbox(
                        label="Review Title (optional)",
                        placeholder="e.g. Best product ever",
                        max_lines=1,
                    )
                    text_input = gr.Textbox(
                        label="Review Text",
                        placeholder="Paste the review here…",
                        lines=6,
                    )
                    analyze_btn = gr.Button("Analyze Sentiment ▶", variant="primary")

                with gr.Column(scale=2):
                    proba_output = gr.Label(
                        label="Class Probabilities",
                        num_top_classes=2,
                    )
                    verdict_output = gr.Markdown()

            gr.Examples(
                examples=EXAMPLES,
                inputs=[title_input, text_input],
                label="📋 Real Amazon review examples — click to load",
                examples_per_page=6,
            )

            analyze_btn.click(
                fn=analyze_single,
                inputs=[title_input, text_input],
                outputs=[proba_output, verdict_output],
            )
            text_input.submit(
                fn=analyze_single,
                inputs=[title_input, text_input],
                outputs=[proba_output, verdict_output],
            )

        # ── Batch ────────────────────────────────────────────────────────────
        with gr.Tab("📊 Batch Analysis"):
            gr.Markdown("Paste multiple reviews — **one review per line**.")

            batch_input = gr.Textbox(
                label="Reviews (one per line)",
                placeholder=(
                    "This product is amazing, totally worth it!\n"
                    "Broken on arrival, terrible quality.\n"
                    "Fast delivery, works as expected."
                ),
                lines=10,
            )
            batch_btn = gr.Button("Analyze All ▶", variant="primary")
            batch_summary = gr.Markdown()
            batch_output = gr.Dataframe(
                headers=["Review", "Sentiment", "Confidence"],
                datatype=["str", "str", "str"],
                wrap=True,
            )

            batch_btn.click(
                fn=analyze_batch,
                inputs=[batch_input],
                outputs=[batch_output, batch_summary],
            )

        # ── About ────────────────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown(ABOUT)

    gr.Markdown(
        "---\nBuilt as a portfolio project · "
        "Model: TF-IDF + Logistic Regression trained on Amazon Reviews (Kaggle)"
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)
