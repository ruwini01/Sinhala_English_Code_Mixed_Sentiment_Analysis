# B1: TF-IDF (char+word n-grams) -> Logistic Regression
"""
Baseline B1 — the simplest model in the difficulty ladder.

No neural network and no notion of meaning: it counts which character and
word n-grams co-occur with each sentiment, weights them by TF-IDF, and fits
a linear (Logistic Regression) decision boundary. This is the floor every
other model must beat — if the proposed XLM-R+LoRA model cannot clear B1,
something is wrong with the pipeline, not the architecture.

Why char n-grams matter here: romanized Singlish is spelled inconsistently
(hondai / hondhai / hodai). Word features see three unrelated tokens; char
n-grams share the substring "hond", so the signal survives. We combine
char_wb (3-5) with word (1-2) n-grams via a FeatureUnion.

Usage:
    python -m src.models.baseline_tfidf
    python -m src.models.baseline_tfidf --text-col tagged_text --outdir results/b1
"""

import argparse
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.models.common import (
    evaluate,
    format_report,
    get_xy,
    load_split,
    save_metrics,
)
from src.common.logger import get_logger
from src.common.seed import set_seed

log = get_logger("baseline_tfidf")


def build_model(C: float = 1.0) -> Pipeline:
    """TF-IDF (word + char features) -> Logistic Regression."""
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,            # ignore tokens appearing in < 2 docs (noise)
        sublinear_tf=True,   # dampen very frequent terms: 1 + log(tf)
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",  # char n-grams, but kept inside word boundaries
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
    )
    features = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])

    clf = LogisticRegression(
        max_iter=1000,
        C=C,
        class_weight="balanced",  # counter the negative-class minority
    )
    return Pipeline([("features", features), ("clf", clf)])


def run(text_col: str = "cleaned_text", outdir: str = "results/b1_tfidf",
        processed_dir: str = "data/processed", C: float = 1.0) -> dict:
    set_seed(42)

    log.info("Loading splits ...")
    train_df = load_split("train", processed_dir)
    val_df = load_split("val", processed_dir)
    test_df = load_split("test", processed_dir)
    X_train, y_train = get_xy(train_df, text_col)
    X_val, y_val = get_xy(val_df, text_col)
    X_test, y_test = get_xy(test_df, text_col)
    log.info(f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}  | input column='{text_col}'")

    log.info("Fitting TF-IDF + Logistic Regression ...")
    model = build_model(C=C)
    model.fit(X_train, y_train)
    n_features = len(model.named_steps["features"].get_feature_names_out())
    log.info(f"Vocabulary size (word+char features): {n_features:,}")

    val_metrics = evaluate(y_val, model.predict(X_val))
    test_metrics = evaluate(y_test, model.predict(X_test))
    log.info(format_report(val_metrics, "VALIDATION"))
    log.info(format_report(test_metrics, "TEST"))

    os.makedirs(outdir, exist_ok=True)
    joblib.dump(model, os.path.join(outdir, "model.joblib"))
    save_metrics({"model": "B1_tfidf_logreg", "text_col": text_col,
                  "val": val_metrics, "test": test_metrics},
                 os.path.join(outdir, "metrics.json"))
    log.info(f"Saved model + metrics to {outdir}/")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="B1: TF-IDF + Logistic Regression baseline")
    parser.add_argument("--text-col", default="cleaned_text",
                        help="input column: cleaned_text (plain) or tagged_text (with LID tags)")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--outdir", default="results/b1_tfidf")
    parser.add_argument("--C", type=float, default=1.0, help="LogReg inverse regularisation strength")
    args = parser.parse_args()
    run(text_col=args.text_col, outdir=args.outdir,
        processed_dir=args.processed_dir, C=args.C)
