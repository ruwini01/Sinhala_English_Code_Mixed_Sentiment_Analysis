"""
backend_example.py  -  Reference backend that serves YOUR trained model.
=========================================================================
This is the server the frontend talks to when NEXT_PUBLIC_USE_MOCK=false.
You run this AFTER your model is trained and saved. It is intentionally
small and commented so you can adapt it.

It does three things the frontend needs:
  1. /predict      -> returns label + probabilities + token SHAP weights
  2. CORS enabled  -> so the browser can call it
  3. same JSON shape the mock used -> frontend needs zero changes

Run it:
    pip install fastapi uvicorn torch transformers peft shap
    uvicorn backend_example:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# 1. LOAD YOUR MODEL ONCE AT STARTUP
# ---------------------------------------------------------------------------
# Point this at the folder where you saved your trained LoRA model, e.g. the
# output of model.save_pretrained("singlish-lora") merged with the base, OR a
# Hugging Face repo id you pushed to (e.g. "ruwini/singlish-sentiment-lora").
MODEL_PATH = "ruwini/singlish-sentiment-lora"   # <-- CHANGE THIS

LABELS = ["negative", "neutral", "positive"]    # <-- match your training order!

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# ---------------------------------------------------------------------------
# 2. (OPTIONAL) SHAP EXPLAINER
# ---------------------------------------------------------------------------
# SHAP can be slow per request. For a smooth demo you can either:
#   (a) compute SHAP live (shown below, simplest), or
#   (b) precompute and cache, or use attention/gradient attributions for speed.
import shap
import numpy as np

def predict_proba(texts):
    enc = tokenizer(list(texts), return_tensors="pt", padding=True,
                    truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=-1).numpy()

explainer = shap.Explainer(predict_proba, tokenizer)

# ---------------------------------------------------------------------------
# 3. THE API
# ---------------------------------------------------------------------------
app = FastAPI(title="Singlish Sentiment API")

# allow the Next.js dev server (and your deployed frontend) to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

class Req(BaseModel):
    text: str

def lang_tag(token: str) -> str:
    # same heuristic idea as your tag_language.py - reuse that module instead
    return "SI" if any("\u0d80" <= c <= "\u0dff" for c in token) else "EN"

@app.post("/predict")
def predict(req: Req):
    text = req.text.strip()

    # ----- prediction -----
    probs = predict_proba([text])[0]
    pred_idx = int(np.argmax(probs))
    label = LABELS[pred_idx]
    probabilities = {LABELS[i]: float(round(probs[i], 3)) for i in range(len(LABELS))}

    # ----- SHAP token attributions toward the predicted class -----
    shap_values = explainer([text])
    # shap_values[0].data are the tokens; .values[:, pred_idx] are contributions
    toks = shap_values[0].data
    vals = shap_values[0].values[:, pred_idx]
    tokens = []
    for tok, val in zip(toks, vals):
        tok = tok.strip()
        if not tok:
            continue
        tokens.append({
            "text": tok,
            "weight": float(val),         # signed contribution toward `label`
            "lang": lang_tag(tok),
        })

    SI = sum(1 for t in tokens if t["lang"] == "SI")
    EN = len(tokens) - SI

    # ----- human-readable explanation -----
    top = sorted(tokens, key=lambda t: abs(t["weight"]), reverse=True)[:2]
    top_str = " and ".join(f'"{t["text"]}"' for t in top)
    explanation = (
        f'The words {top_str} most strongly pushed this toward {label}.'
        if label != "neutral" else
        "No strong sentiment words dominated, so the comment reads as neutral."
    )

    return {
        "label": label,
        "confidence": probabilities[label],
        "probabilities": probabilities,
        "tokens": tokens,
        "language_mix": {"SI": SI, "EN": EN},
        "explanation": explanation,
    }

@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_PATH}
