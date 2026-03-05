import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def main():
    df = pd.read_csv("train.csv")
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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

    # Объединяем два пространства признаков (конкатенация)
    features = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",  # полезно при дисбалансе
        n_jobs=None
    )

    model = Pipeline([
        ("features", features),
        ("clf", clf),
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(classification_report(y_test, preds, digits=4))

if __name__ == "__main__":
    main()