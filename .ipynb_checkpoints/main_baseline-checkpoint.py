import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test.csv"
TEXT_COL = "text"
LABEL_COL = "label"

def make_base_pipeline():
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

    features = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])

    clf = LinearSVC(class_weight="balanced")

    return Pipeline([
        ("tfidf", features),
        ("clf", clf),
    ])

def eval_model(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "weighted_f1": f1_score(y_test, pred, average="weighted"),
    }

def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train = train_df[TEXT_COL].astype(str)
    y_train = train_df[LABEL_COL].astype(str)
    X_test  = test_df[TEXT_COL].astype(str)
    y_test  = test_df[LABEL_COL].astype(str)

    model = make_base_pipeline()
    model.fit(X_train, y_train)

    metrics = eval_model(model, X_test, y_test)
    print("BASELINE METRICS:", metrics)

if __name__ == "__main__":
    main()

# BASELINE METRICS: {'accuracy': 0.6394366197183099, 'macro_f1': 0.4400051801265337, 'weighted_f1': 0.6173177677259828}