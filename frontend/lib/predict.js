/**
 * lib/predict.js
 * =====================================================================
 * THIS IS THE ONLY FILE YOU CHANGE TO CONNECT YOUR REAL MODEL.
 * =====================================================================
 *
 * Right now it returns a realistic MOCK prediction so you can develop and
 * demo the UI before the model exists. When your model is trained and served
 * (see README "Connecting the real model"), flip USE_MOCK to false and set
 * NEXT_PUBLIC_API_URL to your backend's address.
 *
 * The shape of the returned object is fixed - your real backend must return
 * the SAME shape, so the UI never has to change:
 *
 * {
 *   label: "positive" | "negative" | "neutral",
 *   confidence: 0.0..1.0,
 *   probabilities: { positive: n, negative: n, neutral: n },   // sum ~= 1
 *   tokens: [ { text: "mama", weight: -1..1, lang: "SI"|"EN" }, ... ],
 *   language_mix: { SI: 6, EN: 2 },
 *   explanation: "human-readable why-string"
 * }
 *
 * `weight` is the SHAP-style contribution toward the predicted label:
 *   positive weight  -> pushed the prediction toward its label (green)
 *   negative weight  -> pushed against it (red)
 */

const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK !== "false";
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ---- a small Sinhala lexicon used only to make the MOCK look realistic ----
const SI_WORDS = new Set([
  "mama", "api", "oya", "eka", "ekata", "ne", "nehe", "naa", "na", "epa",
  "harima", "godak", "hari", "hondai", "honda", "lassanai", "patta", "niyamai",
  "supiri", "maru", "kiyala", "wage", "machan", "aiyo", "ela", "narakai",
  "kalakanni", "boru", "හොඳයි", "ලස්සනයි", "නරකයි", "හරිම", "ගොඩක්", "නෑ",
]);
const POS_WORDS = new Set([
  "hondai", "honda", "lassanai", "patta", "niyamai", "supiri", "maru", "ela",
  "good", "great", "nice", "love", "best", "superb", "හොඳයි", "ලස්සනයි", "supiriyi",
]);
const NEG_WORDS = new Set([
  "narakai", "kalakanni", "boru", "bad", "worst", "hate", "terrible", "waste",
  "නරකයි", "epa", "aparade",
]);
const NEG_FLIP = new Set(["ne", "nehe", "naa", "na", "නෑ", "නැහැ", "not", "epa"]);
const INTENSIFY = new Set(["harima", "godak", "hari", "හරිම", "ගොඩක්", "very", "really"]);

function isSinhalaScript(w) {
  return /[\u0D80-\u0DFF]/.test(w);
}

function mockPredict(text) {
  const raw = text.trim().split(/\s+/).filter(Boolean);
  let score = 0;
  let intensifier = 1;
  let flip = 1;

  const tokens = raw.map((w) => {
    const lw = w.toLowerCase().replace(/[.,!?]/g, "");
    let weight = 0;
    if (POS_WORDS.has(lw)) weight = 0.55;
    else if (NEG_WORDS.has(lw)) weight = -0.55;
    else if (NEG_FLIP.has(lw)) {
      weight = -0.45;
      flip *= -1;
    } else if (INTENSIFY.has(lw)) {
      weight = 0.3;
      intensifier = 1.5;
    } else weight = (Math.random() - 0.5) * 0.08;

    const lang =
      isSinhalaScript(w) || SI_WORDS.has(lw) ? "SI" : "EN";
    return { text: w, weight, lang };
  });

  // combine: intensifier scales sentiment words, negation flips overall
  tokens.forEach((t) => {
    if (POS_WORDS.has(t.text.toLowerCase()) || NEG_WORDS.has(t.text.toLowerCase())) {
      t.weight *= intensifier;
    }
    score += t.weight;
  });
  score *= flip;
  if (flip === -1) {
    // reflect the flip visually on the negation token
    const negTok = tokens.find((t) => NEG_FLIP.has(t.text.toLowerCase()));
    if (negTok) negTok.weight = -0.6;
  }

  let label = "neutral";
  if (score > 0.25) label = "positive";
  else if (score < -0.25) label = "negative";

  // build pseudo-probabilities
  const base = { positive: 0.2, negative: 0.2, neutral: 0.2 };
  const bump = Math.min(0.65, Math.abs(score) + 0.25);
  base[label] += bump;
  const sum = base.positive + base.negative + base.neutral;
  const probabilities = {
    positive: +(base.positive / sum).toFixed(3),
    negative: +(base.negative / sum).toFixed(3),
    neutral: +(base.neutral / sum).toFixed(3),
  };

  const SI = tokens.filter((t) => t.lang === "SI").length;
  const EN = tokens.length - SI;

  // human-readable explanation from the strongest tokens
  const top = [...tokens]
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
    .slice(0, 2)
    .map((t) => `"${t.text}"`)
    .join(" and ");
  const explanation =
    label === "neutral"
      ? `No strong sentiment words detected, so the comment reads as neutral.`
      : `The words ${top} most strongly pushed this toward ${label}.`;

  return {
    label,
    confidence: probabilities[label],
    probabilities,
    tokens,
    language_mix: { SI, EN },
    explanation,
  };
}

/**
 * The function the UI calls. Returns a Promise of the prediction object.
 */
export async function predictSentiment(text) {
  if (USE_MOCK) {
    // simulate a little network latency so loading states are visible
    await new Promise((r) => setTimeout(r, 450));
    return mockPredict(text);
  }

  // ---- REAL MODEL PATH ----
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    throw new Error(`Prediction failed (${res.status})`);
  }
  return res.json();
}

export const EXAMPLES = [
  "mama hithanne ne meka hondai kiyala",
  "ela machan, harima supiri video ekak",
  "මේ චිත්‍රපටය හරිම ලස්සනයි",
  "this update is a complete waste, godak narakai",
  "aiyo traffic eka නම් අමාරුයි",
  "song eka patta, niyamai වැඩක්",
];
