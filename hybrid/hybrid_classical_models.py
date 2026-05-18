import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC


def _eval(y_true, y_pred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        return {
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        }


def _fit_eval(name, model, X_train, y_train, X_test, y_test, results):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = _eval(y_test, pred)
    print(f"{name}: {metrics}")
    results.append({"model": name, **metrics})


def run_classical(
    vecdir: str,
    class_weight: str | None = None,
    include_tfidf_only: bool = True,
):
    """
    Запускает набор линейных моделей на гибридных векторах + (опц.) TF-IDF-only baseline.

    class_weight:
      - None      → веса классов выключены (часто лучше на средне-дисбалансированных датасетах)
      - "balanced" → class_weight="balanced" sklearn (старое поведение)
    include_tfidf_only:
      - True  → пытается найти texts_train.csv/texts_test.csv и запустить MultinomialNB+ComplementNB
                на чистом TF-IDF (без BERT-блока) как доп. baseline
    """
    X_train = sp.load_npz(os.path.join(vecdir, "X_train_hybrid.npz"))
    X_test = sp.load_npz(os.path.join(vecdir, "X_test_hybrid.npz"))
    y_train = pd.read_csv(os.path.join(vecdir, "y_train.csv")).iloc[:, 0].astype(str)
    y_test = pd.read_csv(os.path.join(vecdir, "y_test.csv")).iloc[:, 0].astype(str)

    # Подсчёт распределения для безопасного CV
    class_counts = y_train.value_counts()
    min_class_count = int(class_counts.min())
    num_classes = int(class_counts.shape[0])

    results: list = []

    print(f"--- hybrid (TF-IDF + BERT) | class_weight={class_weight!r} ---")
    print(
        f"y_train: {len(y_train)} объектов, {num_classes} классов, "
        f"мин. класс = {min_class_count} пример(а/ов)"
    )

    base_kwargs = {"random_state": 42}
    if class_weight is not None:
        base_kwargs["class_weight"] = class_weight

    _fit_eval(
        "linear_svc",
        LinearSVC(max_iter=10000, dual=False, **base_kwargs),
        X_train, y_train, X_test, y_test, results,
    )

    # Калиброванный SVC требует, чтобы в каждом фолде stratified CV было ≥1 пример каждого класса.
    # Стратификация делит min_class_count на cv фолдов и хочет ≥1 на фолд, т.е. cv ≤ min_class_count.
    # Берём максимум 3, но не больше min_class_count, и не меньше 2.
    if min_class_count >= 2:
        safe_cv = max(2, min(3, min_class_count))
        _fit_eval(
            "linear_svc_calibrated",
            CalibratedClassifierCV(
                LinearSVC(max_iter=10000, dual=False, **base_kwargs),
                method="sigmoid", cv=safe_cv,
            ),
            X_train, y_train, X_test, y_test, results,
        )
        if safe_cv < 3:
            print(f"  (linear_svc_calibrated использовал cv={safe_cv}, потому что мин. класс = {min_class_count})")
    else:
        rare_classes = class_counts[class_counts < 2].index.tolist()
        print(
            f"linear_svc_calibrated: SKIPPED — есть классы с одним примером "
            f"(нужно ≥2 для калибровки). Редкие: {rare_classes[:5]}"
            f"{'...' if len(rare_classes) > 5 else ''}"
        )
        results.append({
            "model": "linear_svc_calibrated",
            "balanced_accuracy": None,
            "macro_f1": None,
            "skipped_reason": f"min_class_count={min_class_count} < 2",
        })

    _fit_eval(
        "logreg",
        LogisticRegression(max_iter=2000, n_jobs=-1, **base_kwargs),
        X_train, y_train, X_test, y_test, results,
    )

    _fit_eval(
        "ridge_classifier",
        RidgeClassifier(**base_kwargs),
        X_train, y_train, X_test, y_test, results,
    )

    if include_tfidf_only:
        train_texts_path = os.path.join(vecdir, "texts_train.csv")
        test_texts_path = os.path.join(vecdir, "texts_test.csv")

        if os.path.exists(train_texts_path) and os.path.exists(test_texts_path):
            print(f"\n--- TF-IDF only baseline ---")
            train_df = pd.read_csv(train_texts_path)
            test_df = pd.read_csv(test_texts_path)

            text_col = train_df.columns[0]
            label_col = train_df.columns[1]

            tfidf = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2),
                min_df=2, max_df=0.98, sublinear_tf=True,
                token_pattern=r"(?u)\b\w\w+\b",
            )
            Xtr = tfidf.fit_transform(train_df[text_col].astype(str))
            Xte = tfidf.transform(test_df[text_col].astype(str))
            ytr = train_df[label_col].astype(str)
            yte = test_df[label_col].astype(str)

            _fit_eval("multinomial_nb_tfidf_only", MultinomialNB(),
                      Xtr, ytr, Xte, yte, results)
            _fit_eval("complement_nb_tfidf_only", ComplementNB(),
                      Xtr, ytr, Xte, yte, results)
            nb_lr = LogisticRegression(max_iter=2000, n_jobs=-1, **base_kwargs)
            _fit_eval("logreg_tfidf_only", nb_lr, Xtr, ytr, Xte, yte, results)

    # Сохраняем сводку
    out_path = os.path.join(vecdir, "classical_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_weight": class_weight,
                "include_tfidf_only": include_tfidf_only,
                "min_class_count": min_class_count,
                "num_classes": num_classes,
                "results": results,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nSaved: {out_path}")
    return results


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vecdir", required=True)
    parser.add_argument("--class-weight", default=None,
                        choices=[None, "balanced"], nargs="?")
    parser.add_argument("--no-tfidf-only", action="store_true",
                        help="Не запускать TF-IDF-only baseline (MultinomialNB и пр.)")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    run_classical(
        args.vecdir,
        class_weight=args.class_weight,
        include_tfidf_only=not args.no_tfidf_only,
    )


if __name__ == "__main__":
    main()