import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import f1_score, accuracy_score

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.text_normalize import normalize_text

TRAIN_PATH = "../data/train.csv"
TEST_PATH  = "../data/test.csv"

LABEL_COL = "label"
TEXT_COL = "text"

def make_features_word_char():
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)$$[A-Z_]+$$|\b\w\w+\b",
        lowercase=True,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        lowercase=True,
    )
    return FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])

def eval_model(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "weighted_f1": f1_score(y_test, pred, average="weighted"),
    }

def run():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train, y_train = train_df[TEXT_COL].astype(str), train_df[LABEL_COL].astype(str)
    X_test,  y_test  = test_df[TEXT_COL].astype(str),  test_df[LABEL_COL].astype(str)

    features = make_features_word_char()

    models = {
        "linear_svm": LinearSVC(class_weight="balanced"),
        "logreg": LogisticRegression(max_iter=3000, class_weight="balanced"),
    }

    rows = []

    # baseline (без нормализации)
    for clf_name, clf in models.items():
        pipe = Pipeline([
            ("tfidf", features),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        rows.append({"variant": "no_norm", "clf": clf_name, **eval_model(pipe, X_test, y_test)})

    # с нормализацией
    normalizer = FunctionTransformer(lambda xs: [normalize_text(x) for x in xs], validate=False)

    for clf_name, clf in models.items():
        pipe = Pipeline([
            ("norm", normalizer),
            ("tfidf", features),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        rows.append({"variant": "norm", "clf": clf_name, **eval_model(pipe, X_test, y_test)})

    res = pd.DataFrame(rows).sort_values(["weighted_f1", "macro_f1"], ascending=False)

    print("\n=== RESULTS (sorted by weighted_f1) ===")
    print(res.to_string(index=False))

if __name__ == "__main__":
    run()


# === RESULTS (sorted by weighted_f1) ===
# variant        clf  accuracy  macro_f1  weighted_f1
# no_norm linear_svm  0.639437  0.440005     0.617318
#    norm linear_svm  0.639437  0.440005     0.617318
# no_norm     logreg  0.588732  0.425016     0.569212
#    norm     logreg  0.588732  0.425739     0.569165

# Нормализация не улучшает метрики, поэтому её не используем