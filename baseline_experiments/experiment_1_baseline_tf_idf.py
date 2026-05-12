import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score

TRAIN_PATH = "../data/train.csv"
TEST_PATH  = "../data/test.csv"

LABEL_COL = "label"
TEXT_COL = "text"

# kind: "word", "char", "word+char"
def make_features(kind):
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        # $$[A-Z_]+$$ — ловит [DOCUMENT], [NAME], [DOC_NUMBER] и т.п.
        # \b\w\w+\b — обычные слова длиной ≥2
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
    if kind == "word":
        return word_tfidf
    if kind == "char":
        return char_tfidf
    if kind == "word+char":
        return FeatureUnion([
            ("word", word_tfidf),
            ("char", char_tfidf),
        ])
    raise ValueError(kind)

def make_clf(name: str):
    if name == "linear_svm":
        return LinearSVC(class_weight="balanced")
    if name == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced")
    if name == "nb":
        return MultinomialNB()
    raise ValueError(name)

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

    feature_kinds = ["word", "char", "word+char"]
    clfs = ["linear_svm", "logreg", "nb"]

    rows = []
    for fk in feature_kinds:
        features = make_features(fk)
        for clf_name in clfs:
            clf = make_clf(clf_name)

            model = Pipeline([
                ("tfidf", features),
                ("clf", clf),
            ])

            model.fit(X_train, y_train)
            metrics = eval_model(model, X_test, y_test)

            rows.append({
                "features": fk,
                "clf": clf_name,
                **metrics
            })
            print(rows[-1])

    res = pd.DataFrame(rows).sort_values(["weighted_f1", "macro_f1"], ascending=False)
    print("\n=== RESULTS (sorted by weighted_f1) ===")
    print(res.to_string(index=False))

if __name__ == "__main__":
    run()


# === RESULTS (sorted by weighted_f1) ===
#  features        clf  accuracy  macro_f1  weighted_f1
# word+char linear_svm  0.639437  0.440005     0.617318
#      word linear_svm  0.639437  0.433507     0.617000
#      char linear_svm  0.608451  0.462396     0.592787
# word+char     logreg  0.588732  0.425016     0.569212
#      word     logreg  0.557746  0.399598     0.543854
#      char     logreg  0.535211  0.407050     0.530058
#      word         nb  0.315493  0.079313     0.220283
# word+char         nb  0.278873  0.050399     0.180470
#      char         nb  0.256338  0.043960     0.160601

# naive_bayes показывает наихудние результаты, Linear SVM наилучшие.
# По совокупности метрик лучшие результаты у комбинированного tf-idf

# Linear SVM, Logistic Regression