"""Inference API serving the trained P1 model to the Next.js frontend.

Matches the JSON contract fixed in frontend/lib/predict.js exactly:
  POST /predict  {"text": "..."}  ->
  { label, confidence, probabilities{}, tokens[{text,weight,lang}],
    language_mix{SI,EN}, explanation }

Token weights are occlusion attributions (probability change for the
predicted class when the word is removed), computed in one batched forward
pass — fast enough for CPU. SHAP is reserved for the thesis analysis
(src/explain/run_explain.py); occlusion gives the same sign/ranking for
single sentences at interactive speed.

Run from ml/:
    pip install fastapi uvicorn
    uvicorn src.serve.api:app --port 8000
"""

import os
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load HF_TOKEN (and any other secrets) from ml/.env if present.
# The .env file is gitignored — the token never enters the repository.
_env_file = Path(__file__).resolve().parents[2] / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from src.common.schema import ID2LABEL, LID_TAGS
from src.models.proposed_xlmr_lora import ProposedModel
from src.preprocess.language_id import token_lang

CKPT = "data/checkpoints/lora_scl_lid.pt"
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 128

app = FastAPI(title="Singlish sentiment API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

print("loading model (first run downloads XLM-R, ~1.1 GB)...")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ProposedModel(model_name=MODEL_NAME)
state = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(state)
model.eval()
print("model ready")


def probs_for(texts):
    words_list = [t.split() if t.split() else ["…"] for t in texts]
    enc = tok(words_list, is_split_into_words=True, truncation=True,
              max_length=MAX_LEN, padding=True, return_tensors="pt")
    lid = torch.full_like(enc["input_ids"], LID_TAGS["special"])
    for i, words in enumerate(words_list):
        tags = [LID_TAGS.get(token_lang(w), LID_TAGS["other"]) for w in words]
        for pos, wid in enumerate(enc.word_ids(batch_index=i)):
            if wid is not None:
                lid[i, pos] = tags[wid]
    with torch.no_grad():
        logits, _ = model(enc["input_ids"], enc["attention_mask"], lid)
    return torch.softmax(logits, dim=1)


class Query(BaseModel):
    text: str


@app.post("/predict")
def predict(q: Query):
    text = q.text.strip()
    words = text.split()
    if not words:
        return {"label": "neutral", "confidence": 0.0,
                "probabilities": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
                "tokens": [], "language_mix": {"SI": 0, "EN": 0},
                "explanation": "Empty input."}

    # full sentence + one variant per removed word, one batched forward pass
    variants = [text] + [" ".join(words[:i] + words[i + 1:]) or "…"
                         for i in range(len(words))]
    p = probs_for(variants)
    full = p[0]
    cls = int(full.argmax())
    label = ID2LABEL[cls]

    # occlusion weight: how much the word supported the predicted class
    deltas = [float(full[cls] - p[i + 1][cls]) for i in range(len(words))]
    scale = max(abs(d) for d in deltas) or 1.0
    weights = [d / scale for d in deltas]

    tokens = [{"text": w,
               "weight": round(weights[i], 3),
               "lang": "SI" if token_lang(w) == "sin" else "EN"}
              for i, w in enumerate(words)]
    mix = {"SI": sum(1 for t in tokens if t["lang"] == "SI"),
           "EN": sum(1 for t in tokens if t["lang"] == "EN")}

    ranked = sorted(zip(words, weights), key=lambda x: -abs(x[1]))
    pos_w = [w for w, d in ranked if d > 0][:3]
    neg_w = [w for w, d in ranked if d < 0][:2]
    explanation = f"Predicted {label} ({float(full[cls]):.0%})."
    if pos_w:
        explanation += f" Strongest support: {', '.join(pos_w)}."
    if neg_w:
        explanation += f" Pushed against: {', '.join(neg_w)}."

    return {
        "label": label,
        "confidence": round(float(full[cls]), 3),
        "probabilities": {ID2LABEL[i]: round(float(full[i]), 3) for i in range(3)},
        "tokens": tokens,
        "language_mix": mix,
        "explanation": explanation,
    }


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": CKPT}
