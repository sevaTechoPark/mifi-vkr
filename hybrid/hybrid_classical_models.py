import os
import argparse

import pandas as pd
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize


def eval_model(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "balanced_accuracy": round(balanced_accuracy_score(y_test, pred), 6),
        "macro_f1": round(f1_score(y_test, pred, average="macro", zero_division=0), 6),
    }


def build_tfidf_only(vecdir, text_col: str = "text", label_col: str = "label"):
    train_path = os.path.join(vecdir, "y_train.csv")
    test_path = os.path.join(vecdir, "y_test.csv")
    meta_path = os.path.join(vecdir, "meta.json")

    # Для TF-IDF-only нам нужны исходные тексты.
    # Предполагается, что рядом с гибридными векторами лежат исходные train/test CSV.
    # Если их нет — этот baseline можно отключить или передавать пути отдельно.
    raise_if_missing = []
    for p in (train_path, test_path):
        if not os.path.exists(p):
            raise_if_missing.append(p)
    if raise_if_missing:
        raise FileNotFoundError(
            f"Cannot build TF-IDF-only baseline, missing files: {raise_if_missing}"
        )

    # Здесь предполагается, что исходные данные лежат рядом с vecdir.
    # Если у тебя другая структура, проще сделать отдельный скрипт для TF-IDF-only.
    raise NotImplementedError(
        "TF-IDF-only baseline requires access to original train/test texts. "
        "Подстрой этот блок под свою структуру данных."
    )


def run_classical(vecdir: str):
    # Гибридные векторы для линейных моделей
    X_train = sp.load_npz(os.path.join(vecdir, "X_train_hybrid.npz"))
    X_test = sp.load_npz(os.path.join(vecdir, "X_test_hybrid.npz"))
    y_train = pd.read_csv(os.path.join(vecdir, "y_train.csv")).iloc[:, 0].astype(str)
    y_test = pd.read_csv(os.path.join(vecdir, "y_test.csv")).iloc[:, 0].astype(str)

    models = {
        "linear_svc": LinearSVC(
            class_weight="balanced",
            max_iter=10000,
            dual=False,  # если n_samples > n_features; убери/измени при необходимости
        ),
        "logreg": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            n_jobs=-1,
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = eval_model(model, X_test, y_test)
        results.append({"model": name, **metrics})
        print(f"{name}: {metrics}")

    # Третий baseline: MultinomialNB на TF-IDF-only (без BERT)
    # Важно: MultinomialNB требует неотрицательные features.
    # Если хочешь TF-IDF-only baseline в этом же файле, нужно иметь доступ к исходным текстам.
    # Здесь оставлен каркас — его нужно адаптировать под структуру проекта.

    # try:
    #     X_train_tfidf, X_test_tfidf, y_train_tf, y_test_tf = build_tfidf_only(vecdir)
    #     nb_model = MultinomialNB()
    #     nb_model.fit(X_train_tfidf, y_train_tf)
    #     nb_metrics = eval_model(nb_model, X_test_tfidf, y_test_tf)
    #     results.append({"model": "multinomial_nb_tfidf_only", **nb_metrics})
    #     print(f"multinomial_nb_tfidf_only: {nb_metrics}")
    # except Exception as e:
    #     print(f"Skipping MultinomialNB TF-IDF-only baseline: {e}")

    return results


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Run classical linear models on hybrid vectors."
    )
    parser.add_argument("--vecdir", required=True)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    run_classical(args.vecdir)


if __name__ == "__main__":
    main()