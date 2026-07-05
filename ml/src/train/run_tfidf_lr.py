"""B1 / tfidf_lr: TF-IDF + Logistic Regression baseline.

Word (1-2)-grams + character (2-5)-grams — char n-grams matter for romanized
Sinhala spelling variation. Class-weighted LR, seed=42.

Run from ml/ (CPU, ~1 min):
    python -m src.train.run_tfidf_lr
"""

import time

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.common.schema import SEED, TEXT_COL
from src.train.data_io import load_split_frames
from src.train.results_io import evaluate, save_results

HP = {
    "word_ngrams": (1, 2),
    "word_max_features": 50000,
    "char_ngrams": (2, 5),
    "char_max_features": 100000,
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 2000,
}


def main():
    t0 = time.time()
    splits = load_split_frames()
    texts = {k: v[TEXT_COL].fillna("") for k, v in splits.items()}
    y = {k: v["label_id"].tolist() for k, v in splits.items()}

    word_vec = TfidfVectorizer(analyzer="word", ngram_range=HP["word_ngrams"],
                               max_features=HP["word_max_features"], sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=HP["char_ngrams"],
                               max_features=HP["char_max_features"], sublinear_tf=True)

    X_train = hstack([word_vec.fit_transform(texts["train"]),
                      char_vec.fit_transform(texts["train"])])
    X_val = hstack([word_vec.transform(texts["val"]), char_vec.transform(texts["val"])])
    X_test = hstack([word_vec.transform(texts["test"]), char_vec.transform(texts["test"])])

    clf = LogisticRegression(C=HP["C"], class_weight=HP["class_weight"],
                             max_iter=HP["max_iter"], random_state=SEED)
    clf.fit(X_train, y["train"])

    val_metrics = evaluate(y["val"], clf.predict(X_val))
    test_metrics = evaluate(y["test"], clf.predict(X_test))

    save_results(
        experiment_id="tfidf_lr",
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        trainable_params=clf.coef_.size + clf.intercept_.size,
        wall_clock_sec=time.time() - t0,
        hyperparams={**HP, "word_ngrams": list(HP["word_ngrams"]),
                     "char_ngrams": list(HP["char_ngrams"])},
        notes="B1 classical floor; word+char TF-IDF, class-weighted LR",
    )


if __name__ == "__main__":
    main()