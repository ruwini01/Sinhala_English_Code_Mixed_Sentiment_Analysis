"use client";

/**
 * TokenHeatmap - renders each token tinted by its SHAP-style weight.
 * Green = pushed toward the predicted label, red = pushed against it.
 */
export default function TokenHeatmap({ tokens }) {
  const color = (w) => {
    const a = Math.min(0.85, Math.abs(w) * 1.3 + 0.08);
    if (w > 0.05) return `rgba(47, 158, 68, ${a})`; // green
    if (w < -0.05) return `rgba(214, 51, 108, ${a})`; // red
    return "rgba(154, 133, 118, 0.12)"; // neutral grey
  };
  const textColor = (w) => (Math.abs(w) > 0.35 ? "#fff" : "var(--ink-900)");

  return (
    <div className="heatmap">
      <div className="heatmap-tokens">
        {tokens.map((t, i) => (
          <span
            key={i}
            className="token si"
            style={{ background: color(t.weight), color: textColor(t.weight) }}
            title={`${t.lang} · contribution ${t.weight.toFixed(2)}`}
          >
            {t.text}
            <em className="token-lang">{t.lang}</em>
          </span>
        ))}
      </div>

      <div className="heatmap-legend">
        <span>
          <i style={{ background: "rgba(47,158,68,0.8)" }} /> pushes toward label
        </span>
        <span>
          <i style={{ background: "rgba(214,51,108,0.8)" }} /> pushes against
        </span>
        <span>
          <i style={{ background: "rgba(154,133,118,0.25)" }} /> little effect
        </span>
      </div>

      <style jsx>{`
        .heatmap {
          margin-top: 6px;
        }
        .heatmap-tokens {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }
        .token {
          position: relative;
          padding: 10px 14px 14px;
          border-radius: 12px;
          font-weight: 500;
          font-size: 1.02rem;
          line-height: 1;
          transition: transform 0.12s ease;
        }
        .token:hover {
          transform: translateY(-2px);
        }
        .token-lang {
          position: absolute;
          bottom: 3px;
          right: 6px;
          font-size: 0.52rem;
          font-style: normal;
          letter-spacing: 0.04em;
          opacity: 0.65;
          font-family: var(--font-ui);
        }
        .heatmap-legend {
          display: flex;
          flex-wrap: wrap;
          gap: 18px;
          margin-top: 16px;
          font-size: 0.78rem;
          color: var(--ink-600);
        }
        .heatmap-legend span {
          display: inline-flex;
          align-items: center;
          gap: 7px;
        }
        .heatmap-legend i {
          width: 13px;
          height: 13px;
          border-radius: 4px;
          display: inline-block;
        }
      `}</style>
    </div>
  );
}
