"use client";

import { useState } from "react";
import { predictSentiment, EXAMPLES } from "../lib/predict";
import ResultPanel from "../components/ResultPanel";

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function analyze(input) {
    const value = (input ?? text).trim();
    if (!value) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const r = await predictSentiment(value);
      setResult(r);
    } catch (e) {
      setError(
        "Could not reach the model. Check that the backend is running, then try again."
      );
    } finally {
      setLoading(false);
    }
  }

  function useExample(ex) {
    setText(ex);
    analyze(ex);
  }

  return (
    <main>
      {/* ambient orange glow */}
      <div className="glow" aria-hidden />

      <header className="top">
        <div className="brand">
          <span className="mark">අ</span>
          <div>
            <div className="brand-name">Singlish Sentiment</div>
            <div className="brand-sub">Explainable analysis for Sinhala-English text</div>
          </div>
        </div>
        <a
          className="ghost-link"
          href="https://github.com"
          target="_blank"
          rel="noreferrer"
        >
          Research demo
        </a>
      </header>

      <section className="hero">
        <h1>
          Read the feeling behind
          <br />
          <span className="accent si">Singlish</span> &amp; code-mixed text.
        </h1>
        <p className="lede">
          Type a comment in Singlish, Sinhala, English, or any mix. The model
          predicts the sentiment and shows you <strong>which words</strong>{" "}
          drove its decision.
        </p>
      </section>

      <section className="panel">
        <div className="input-card">
          <label htmlFor="comment" className="input-label">
            Your comment
          </label>
          <textarea
            id="comment"
            className="si"
            placeholder="e.g.  mama hithanne ne meka hondai kiyala"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze();
            }}
            rows={3}
          />
          <div className="input-row">
            <span className="hint">Press ⌘/Ctrl + Enter to analyze</span>
            <button
              className="cta"
              onClick={() => analyze()}
              disabled={loading || !text.trim()}
            >
              {loading ? "Analyzing…" : "Analyze sentiment"}
            </button>
          </div>
        </div>

        <div className="examples">
          <span className="examples-label">Try an example</span>
          <div className="chips">
            {EXAMPLES.map((ex, i) => (
              <button key={i} className="chip si" onClick={() => useExample(ex)}>
                {ex}
              </button>
            ))}
          </div>
        </div>

        {error && <div className="error">{error}</div>}

        {loading && (
          <div className="skeleton">
            <div className="sk-line" style={{ width: "40%", height: 60 }} />
            <div className="sk-line" style={{ width: "100%" }} />
            <div className="sk-line" style={{ width: "85%" }} />
          </div>
        )}

        {result && !loading && (
          <div className="result-wrap">
            <ResultPanel result={result} />
          </div>
        )}
      </section>

      <footer className="foot">
        <p>
          Sentiment Analysis for Sinhala-English Code-Mixed Text · XLM-R + LoRA +
          SCL + LID · SHAP explainability
        </p>
      </footer>

      <style jsx>{`
        main {
          position: relative;
          max-width: 880px;
          margin: 0 auto;
          padding: 0 22px 80px;
        }
        .glow {
          position: fixed;
          top: -260px;
          left: 50%;
          transform: translateX(-50%);
          width: 760px;
          height: 520px;
          background: radial-gradient(
            ellipse at center,
            rgba(244, 115, 26, 0.16),
            transparent 70%
          );
          pointer-events: none;
          z-index: 0;
        }
        .top {
          position: relative;
          z-index: 1;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 26px 0 10px;
        }
        .brand {
          display: flex;
          align-items: center;
          gap: 13px;
        }
        .mark {
          width: 44px;
          height: 44px;
          border-radius: 13px;
          background: linear-gradient(
            135deg,
            var(--orange-500),
            var(--orange-600)
          );
          color: #fff;
          display: grid;
          place-items: center;
          font-size: 1.4rem;
          font-weight: 700;
          font-family: var(--font-si);
          box-shadow: 0 6px 18px rgba(232, 89, 12, 0.3);
        }
        .brand-name {
          font-weight: 700;
          font-size: 1.02rem;
        }
        .brand-sub {
          font-size: 0.76rem;
          color: var(--ink-400);
        }
        .ghost-link {
          font-size: 0.84rem;
          color: var(--orange-600);
          text-decoration: none;
          font-weight: 500;
          border: 1.5px solid var(--orange-100);
          padding: 8px 15px;
          border-radius: 999px;
          transition: background 0.15s;
        }
        .ghost-link:hover {
          background: var(--orange-50);
        }
        .hero {
          position: relative;
          z-index: 1;
          padding: 46px 0 30px;
        }
        h1 {
          font-size: clamp(2.1rem, 5vw, 3.1rem);
          font-weight: 700;
          line-height: 1.12;
          letter-spacing: -0.02em;
        }
        .accent {
          color: var(--orange-600);
        }
        .lede {
          margin-top: 18px;
          font-size: 1.04rem;
          color: var(--ink-600);
          max-width: 560px;
        }
        .panel {
          position: relative;
          z-index: 1;
        }
        .input-card {
          background: var(--card);
          border: 1px solid var(--line);
          border-radius: var(--radius);
          padding: 20px;
          box-shadow: var(--shadow);
        }
        .input-label {
          display: block;
          font-size: 0.78rem;
          font-weight: 600;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          color: var(--ink-400);
          margin-bottom: 10px;
        }
        textarea {
          width: 100%;
          resize: vertical;
          border: 1.5px solid var(--line);
          border-radius: 13px;
          padding: 14px 16px;
          font-size: 1.08rem;
          color: var(--ink-900);
          background: var(--paper);
          transition: border 0.15s, box-shadow 0.15s;
        }
        textarea:focus {
          outline: none;
          border-color: var(--orange-400);
          box-shadow: 0 0 0 4px var(--orange-50);
        }
        .input-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-top: 14px;
          gap: 12px;
        }
        .hint {
          font-size: 0.76rem;
          color: var(--ink-400);
        }
        .cta {
          background: linear-gradient(
            135deg,
            var(--orange-500),
            var(--orange-600)
          );
          color: #fff;
          font-weight: 600;
          font-size: 0.96rem;
          padding: 12px 24px;
          border-radius: 12px;
          box-shadow: 0 6px 16px rgba(232, 89, 12, 0.26);
          transition: transform 0.12s, box-shadow 0.12s, opacity 0.12s;
        }
        .cta:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 9px 22px rgba(232, 89, 12, 0.32);
        }
        .cta:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        .examples {
          margin-top: 22px;
        }
        .examples-label {
          font-size: 0.78rem;
          font-weight: 600;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          color: var(--ink-400);
        }
        .chips {
          display: flex;
          flex-wrap: wrap;
          gap: 9px;
          margin-top: 11px;
        }
        .chip {
          background: var(--card);
          border: 1px solid var(--line);
          border-radius: 999px;
          padding: 8px 15px;
          font-size: 0.88rem;
          color: var(--ink-600);
          transition: border 0.14s, color 0.14s, background 0.14s;
        }
        .chip:hover {
          border-color: var(--orange-400);
          color: var(--orange-600);
          background: var(--orange-50);
        }
        .error {
          margin-top: 22px;
          background: var(--neg-bg);
          color: var(--neg);
          border: 1px solid #ffd0e0;
          padding: 14px 18px;
          border-radius: 12px;
          font-size: 0.9rem;
        }
        .result-wrap,
        .skeleton {
          margin-top: 30px;
          background: var(--card);
          border: 1px solid var(--line);
          border-radius: var(--radius);
          padding: 26px;
          box-shadow: var(--shadow);
        }
        .sk-line {
          height: 16px;
          background: linear-gradient(
            90deg,
            var(--orange-50),
            var(--orange-100),
            var(--orange-50)
          );
          background-size: 200% 100%;
          border-radius: 8px;
          margin-bottom: 14px;
          animation: shimmer 1.3s infinite;
        }
        @keyframes shimmer {
          to {
            background-position: -200% 0;
          }
        }
        .foot {
          position: relative;
          z-index: 1;
          margin-top: 60px;
          padding-top: 22px;
          border-top: 1px solid var(--line);
          text-align: center;
          font-size: 0.8rem;
          color: var(--ink-400);
        }
      `}</style>
    </main>
  );
}
