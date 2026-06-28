# Singlish Sentiment Analyzer — Frontend

An orange-themed Next.js web app for **explainable sentiment analysis** of
Sinhala-English code-mixed ("Singlish") text. Built for the final-year research
demo / viva. Uses Poppins for English and Noto Sans Sinhala for Sinhala script.

It runs **right now with a built-in mock predictor**, so you can develop and
demo the interface before the model exists. When your model is trained, you
connect it by changing **one environment variable** — no UI code changes.

---

## What it shows

- A text box that accepts Singlish, Sinhala script (සිංහල), English, or any mix
- The predicted **sentiment**: Positive / Negative / Neutral, with confidence
- **Probability bars** for all three classes
- **Explainable AI**: a token-level heatmap (green = pushed toward the label,
  red = pushed against) — the SHAP view, plus a plain-language "why" sentence
- **Language mix** detection ([SI]/[EN]) — the same signal your LID embeddings use
- Clickable example sentences (including Sinhala-script ones)

---

## 1. Run it (mock mode — works today)

You need **Node.js 18+** installed (nodejs.org).

```bash
npm install
npm run dev
```

Open <http://localhost:3000>. Type a comment or click an example. Done.

> The mock predictor lives in `lib/predict.js`. It uses a small Sinhala lexicon
> (intensifiers, negations, sentiment words) so its answers look realistic —
> negations flip the result, intensifiers strengthen it. It is **not** your
> model; it is a stand-in so the UI is fully working before the model is ready.

---

## 2. Connect your REAL model (after it's trained & checked)

The frontend always calls one function — `predictSentiment(text)` in
`lib/predict.js` — and expects this exact JSON back:

```json
{
  "label": "positive",
  "confidence": 0.79,
  "probabilities": { "positive": 0.79, "negative": 0.06, "neutral": 0.15 },
  "tokens": [
    { "text": "mama",   "weight": 0.02,  "lang": "SI" },
    { "text": "harima",  "weight": 0.31,  "lang": "SI" },
    { "text": "hondai",  "weight": 0.55,  "lang": "SI" }
  ],
  "language_mix": { "SI": 6, "EN": 2 },
  "explanation": "The words \"harima\" and \"hondai\" most strongly pushed this toward positive."
}
```

`weight` = each token's SHAP contribution **toward the predicted label**
(positive number = supports the label → green; negative = against → red).

### Step A — Serve your model as an API

Your model is Python (XLM-R + LoRA). The browser can't run it directly, so you
wrap it in a tiny web server that returns the JSON above. A complete, commented
template is included: **`backend_example.py`** (FastAPI). To use it:

```bash
pip install fastapi uvicorn torch transformers peft shap
# edit MODEL_PATH and LABELS at the top of backend_example.py
uvicorn backend_example:app --reload --port 8000
```

This gives you `http://localhost:8000/predict`. Test it:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"mama hithanne ne meka hondai kiyala"}'
```

> Two things to get right in `backend_example.py`:
> 1. **`MODEL_PATH`** — the folder you saved your trained model to, or your
>    Hugging Face repo id (e.g. `ruwini/singlish-sentiment-lora`).
> 2. **`LABELS`** — must be in the SAME order your model was trained with,
>    or the labels will be swapped. (Check `model.config.id2label`.)

### Step B — Point the frontend at it

Copy `.env.local.example` to `.env.local` and set:

```
NEXT_PUBLIC_USE_MOCK=false
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Restart `npm run dev`. That's it — the UI now shows **real predictions and real
SHAP explanations**. No component code changes, because the JSON shape is identical.

### How the pieces talk

```
 Browser (this Next.js app)            Your Python model
 ───────────────────────────          ─────────────────────────
 user types a comment
        │
        ▼
 predictSentiment(text)   ──POST /predict──►  FastAPI (backend_example.py)
        │                                         │ tokenize + XLM-R+LoRA
        │                                         │ softmax -> probabilities
        │                                         │ SHAP -> token weights
        ◄────────────── JSON ─────────────────────┘
        │
        ▼
 ResultPanel + TokenHeatmap render it
```

---

## 3. Deploy for the viva (optional)

- **Frontend:** `npm run build` then host on Vercel (free) or run `npm start` on
  your laptop. Set the two env vars in the host's dashboard.
- **Backend:** the model server is heavier. Easiest for a viva is to run
  `backend_example.py` on your own laptop and demo locally. Alternatively host
  the model on **Hugging Face Spaces** (free GPU-less tier works for inference).
- **Backup:** record a short screen video of the working demo in case the venue
  Wi-Fi fails — examiners accept that gracefully.

---

## Project structure

```
app/
  layout.js        fonts (Poppins + Noto Sans Sinhala) + metadata
  page.js          the whole page: hero, input, examples, result
  globals.css      orange theme tokens + base styles
components/
  ResultPanel.js   prediction headline, probability bars, language mix
  TokenHeatmap.js  the SHAP token strip (green/red word tinting)
lib/
  predict.js       *** the only file to edit to connect the real model ***
backend_example.py reference FastAPI server for your trained model
.env.local.example copy to .env.local to switch mock -> real
```

## Notes

- **Sinhala rendering:** any element that may contain Sinhala has the `.si`
  class, which applies Noto Sans Sinhala. The textarea uses it too, so users can
  type Sinhala script directly.
- **Accessibility:** keyboard focus is visible, reduced-motion is respected,
  the layout is responsive to mobile.
- The mock and the real backend return the **same shape on purpose** — that's
  what lets you build the UI first and swap the brain in later.
```
