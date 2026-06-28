"use client";

import TokenHeatmap from "./TokenHeatmap";

const LABELS = {
  positive: { name: "Positive", color: "var(--pos)", bg: "var(--pos-bg)" },
  negative: { name: "Negative", color: "var(--neg)", bg: "var(--neg-bg)" },
  neutral: { name: "Neutral", color: "var(--neu)", bg: "var(--neu-bg)" },
};

export default function ResultPanel({ result }) {
  const L = LABELS[result.label];

  return (
    <div className="result">
      {/* prediction headline */}
      <div className="verdict" style={{ background: L.bg, borderColor: L.color }}>
        <div className="verdict-label">
          <span className="dot" style={{ background: L.color }} />
          <span style={{ color: L.color }}>{L.name}</span>
        </div>
        <div className="verdict-conf">
          {Math.round(result.confidence * 100)}% confident
        </div>
      </div>

      {/* probability bars */}
      <section className="block">
        <h3>How sure is the model?</h3>
        {["positive", "negative", "neutral"].map((k) => (
          <div className="bar-row" key={k}>
            <span className="bar-name" style={{ color: LABELS[k].color }}>
              {LABELS[k].name}
            </span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{
                  width: `${result.probabilities[k] * 100}%`,
                  background: LABELS[k].color,
                }}
              />
            </div>
            <span className="bar-pct">
              {Math.round(result.probabilities[k] * 100)}%
            </span>
          </div>
        ))}
      </section>

      {/* explainable AI: token heatmap */}
      <section className="block">
        <h3>
          Why <span style={{ color: L.color }}>{L.name.toLowerCase()}</span>?
          <span className="tag">Explainable AI</span>
        </h3>
        <p className="why si">{result.explanation}</p>
        <TokenHeatmap tokens={result.tokens} />
      </section>

      {/* language mix */}
      <section className="block lang-block">
        <h3>Detected language mix</h3>
        <div className="lang-bar">
          <div
            className="lang-seg si-seg"
            style={{
              flex: result.language_mix.SI || 0.001,
            }}
          >
            Sinhala · {result.language_mix.SI}
          </div>
          <div
            className="lang-seg en-seg"
            style={{
              flex: result.language_mix.EN || 0.001,
            }}
          >
            English · {result.language_mix.EN}
          </div>
        </div>
        <p className="lang-note">
          Token-level language identification ([SI]/[EN]) — the same signal fed
          to the model as language-ID embeddings.
        </p>
      </section>

      <style jsx>{`
        .result {
          display: flex;
          flex-direction: column;
          gap: 26px;
          animation: rise 0.4s ease both;
        }
        @keyframes rise {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
        }
        .verdict {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 22px 26px;
          border-radius: var(--radius);
          border: 1.5px solid;
        }
        .verdict-label {
          display: flex;
          align-items: center;
          gap: 12px;
          font-size: 1.7rem;
          font-weight: 700;
        }
        .dot {
          width: 16px;
          height: 16px;
          border-radius: 50%;
        }
        .verdict-conf {
          font-size: 0.95rem;
          color: var(--ink-600);
          font-weight: 500;
        }
        .block h3 {
          font-size: 1.02rem;
          font-weight: 600;
          margin-bottom: 14px;
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .tag {
          font-size: 0.62rem;
          font-weight: 600;
          letter-spacing: 0.05em;
          text-transform: uppercase;
          color: var(--orange-600);
          background: var(--orange-100);
          padding: 3px 9px;
          border-radius: 999px;
        }
        .why {
          background: var(--orange-50);
          border-left: 3px solid var(--orange-400);
          padding: 12px 16px;
          border-radius: 8px;
          font-size: 0.95rem;
          color: var(--ink-900);
          margin-bottom: 16px;
        }
        .bar-row {
          display: flex;
          align-items: center;
          gap: 14px;
          margin-bottom: 11px;
        }
        .bar-name {
          width: 74px;
          font-size: 0.86rem;
          font-weight: 500;
        }
        .bar-track {
          flex: 1;
          height: 12px;
          background: var(--orange-50);
          border-radius: 999px;
          overflow: hidden;
        }
        .bar-fill {
          height: 100%;
          border-radius: 999px;
          transition: width 0.6s cubic-bezier(0.2, 0.8, 0.2, 1);
        }
        .bar-pct {
          width: 44px;
          text-align: right;
          font-size: 0.84rem;
          font-variant-numeric: tabular-nums;
          color: var(--ink-600);
        }
        .lang-bar {
          display: flex;
          gap: 6px;
          height: 44px;
        }
        .lang-seg {
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 10px;
          font-size: 0.82rem;
          font-weight: 600;
          color: #fff;
          min-width: 0;
          overflow: hidden;
          white-space: nowrap;
        }
        .si-seg {
          background: var(--orange-500);
        }
        .en-seg {
          background: var(--ink-400);
        }
        .lang-note {
          margin-top: 10px;
          font-size: 0.78rem;
          color: var(--ink-400);
        }
      `}</style>
    </div>
  );
}
