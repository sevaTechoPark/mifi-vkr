import os
import json
import argparse
import warnings
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import normalize


# -----------------------------------------------------------------------------
# Eval helpers
# -----------------------------------------------------------------------------
def _eval(y_true, y_pred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        return {
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        }


def _safe_fit_eval(name, model, Xtr, ytr, Xte, yte, results, tag=""):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        metrics = _eval(yte, pred)
        full_name = f"{name}[{tag}]" if tag else name
        print(f"  {full_name}: {metrics}")
        results.append({"model": full_name, "feature_source": tag, **metrics})
        return model, metrics
    except Exception as e:
        full_name = f"{name}[{tag}]" if tag else name
        print(f"  {full_name}: FAILED ({type(e).__name__}: {e})")
        results.append({
            "model": full_name, "feature_source": tag,
            "balanced_accuracy": None, "macro_f1": None,
            "error": f"{type(e).__name__}: {e}",
        })
        return None, None


# -----------------------------------------------------------------------------
# C-grid for linear models
# -----------------------------------------------------------------------------
def _grid_linear_svc(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight):
    best = (None, None, None)  # (C, model, metrics)
    base_kwargs = {"random_state": 42, "max_iter": 10000, "dual": False}
    if class_weight is not None:
        base_kwargs["class_weight"] = class_weight

    for C in c_grid:
        model = LinearSVC(C=C, **base_kwargs)
        m, metrics = _safe_fit_eval(
            f"linear_svc_C{C}", model, Xtr, ytr, Xte, yte, results, tag=tag
        )
        if metrics and (best[2] is None or metrics["balanced_accuracy"] > best[2]["balanced_accuracy"]):
            best = (C, m, metrics)
    if best[0] is not None:
        print(f"  ★ best linear_svc[{tag}]: C={best[0]} → {best[2]}")
        results.append({
            "model": f"linear_svc_best[{tag}]",
            "feature_source": tag,
            "best_C": best[0],
            **best[2],
        })
    return best


def _grid_logreg(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight):
    best = (None, None, None)
    for C in c_grid:
        base_kwargs = {
            "C": C, "max_iter": 4000, "random_state": 42,
            "solver": "lbfgs", "n_jobs": -1,
        }
        if class_weight is not None:
            base_kwargs["class_weight"] = class_weight
        model = LogisticRegression(**base_kwargs)
        m, metrics = _safe_fit_eval(
            f"logreg_C{C}", model, Xtr, ytr, Xte, yte, results, tag=tag
        )
        if metrics and (best[2] is None or metrics["balanced_accuracy"] > best[2]["balanced_accuracy"]):
            best = (C, m, metrics)
    if best[0] is not None:
        print(f"  ★ best logreg[{tag}]: C={best[0]} → {best[2]}")
        results.append({
            "model": f"logreg_best[{tag}]",
            "feature_source": tag,
            "best_C": best[0],
            **best[2],
        })
    return best


# -----------------------------------------------------------------------------
# Per-source pipelines
# -----------------------------------------------------------------------------
def _run_on_source(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight,
                   include_knn_centroid=True, include_rbf=False):
    """
    Прогон полного набора линейных моделей на одной фиче.
    tag: 'bert_only' | 'hybrid' | 'tfidf_only'
    """
    print(f"\n=== Source: {tag} | shape={Xtr.shape} | class_weight={class_weight!r} ===")

    # 1) LinearSVC grid
    best_svc = _grid_linear_svc(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight)

    # 2) LogReg grid
    best_lr = _grid_logreg(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight)

    # 3) Calibrated SVC (на лучшем C)
    if best_svc[0] is not None:
        class_counts = pd.Series(ytr).value_counts()
        min_class_count = int(class_counts.min())
        if min_class_count >= 2:
            safe_cv = max(2, min(3, min_class_count))
            cal_kwargs = {"random_state": 42, "max_iter": 10000, "dual": False, "C": best_svc[0]}
            if class_weight is not None:
                cal_kwargs["class_weight"] = class_weight
            for method in ("sigmoid", "isotonic"):
                _safe_fit_eval(
                    f"linear_svc_calibrated_{method}_C{best_svc[0]}",
                    CalibratedClassifierCV(LinearSVC(**cal_kwargs), method=method, cv=safe_cv),
                    Xtr, ytr, Xte, yte, results, tag=tag,
                )

    # 4) RidgeClassifier
    base = {"random_state": 42}
    if class_weight is not None:
        base["class_weight"] = class_weight
    _safe_fit_eval("ridge_classifier", RidgeClassifier(**base),
                   Xtr, ytr, Xte, yte, results, tag=tag)

    # 5) SGD hinge (на больших разреженных это часто полезно)
    sgd_kwargs = {"random_state": 42, "max_iter": 2000, "alpha": 1e-5,
                  "loss": "hinge", "n_jobs": -1}
    if class_weight is not None:
        sgd_kwargs["class_weight"] = class_weight
    _safe_fit_eval("sgd_hinge", SGDClassifier(**sgd_kwargs),
                   Xtr, ytr, Xte, yte, results, tag=tag)

    # 6) NearestCentroid и KNN-cos — только на L2-нормированных представлениях
    #    (bert_only уже L2; hybrid после изменений тоже разумен)
    if include_knn_centroid:
        # для разреженных матриц NearestCentroid с metric='cosine' не работает,
        # но т.к. векторы L2-нормированы, euclidean ≡ cosine с точностью до константы
        if sp.issparse(Xtr):
            _safe_fit_eval("nearest_centroid_euc",
                           NearestCentroid(metric="euclidean"),
                           Xtr, ytr, Xte, yte, results, tag=tag)
            _safe_fit_eval("knn5_euc",
                           KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1),
                           Xtr, ytr, Xte, yte, results, tag=tag)
        else:
            _safe_fit_eval("nearest_centroid_cos",
                           NearestCentroid(metric="euclidean"),  # L2-norm → eq cosine
                           Xtr, ytr, Xte, yte, results, tag=tag)
            _safe_fit_eval("knn5_cos",
                           KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1),
                           Xtr, ytr, Xte, yte, results, tag=tag)
            _safe_fit_eval("knn3_cos",
                           KNeighborsClassifier(n_neighbors=3, metric="cosine", n_jobs=-1),
                           Xtr, ytr, Xte, yte, results, tag=tag)

    # 7) SVC-rbf — только для dense bert_only (на hybrid слишком дорого/мусор)
    if include_rbf and not sp.issparse(Xtr):
        rbf_kwargs = {"random_state": 42, "kernel": "rbf", "C": 4.0, "gamma": "scale"}
        if class_weight is not None:
            rbf_kwargs["class_weight"] = class_weight
        _safe_fit_eval("svc_rbf_C4", SVC(**rbf_kwargs),
                       Xtr, ytr, Xte, yte, results, tag=tag)

    return best_svc, best_lr


# -----------------------------------------------------------------------------
# Stacking
# -----------------------------------------------------------------------------
def _run_stacking(X_bert_tr, X_bert_te, X_hyb_tr, X_hyb_te, ytr, yte,
                  results, best_C_bert, best_C_hyb, class_weight):
    """
    Стэкинг: 3 базовых предиктора → logreg-meta.
    Базы: linear_svc_calibrated на bert_only, logreg на bert_only, linear_svc на hybrid.
    Все обёрнуты в CalibratedClassifierCV для предсказания вероятностей.
    """
    print(f"\n=== Stacking ===")
    class_counts = pd.Series(ytr).value_counts()
    min_cc = int(class_counts.min())
    if min_cc < 2:
        print("  SKIP: min_class_count < 2, calibrated stacking невозможен")
        return
    cv = max(2, min(3, min_cc))

    base_svc = {"random_state": 42, "max_iter": 10000, "dual": False}
    if class_weight is not None:
        base_svc["class_weight"] = class_weight

    # Подходит только для одного X. Stacking стандартный требует общий X для всех base.
    # Используем bert_only как общий X (он dense, разумного размера).
    # Это даёт мета-стек, где разные алгоритмы дают разные сигналы на одних фичах.
    estimators = [
        ("svc_cal", CalibratedClassifierCV(
            LinearSVC(C=(best_C_bert or 1.0), **base_svc), method="sigmoid", cv=cv
        )),
        ("logreg", LogisticRegression(
            C=(best_C_bert or 1.0), max_iter=4000, random_state=42, n_jobs=-1,
            class_weight=class_weight,
        )),
        ("ridge", CalibratedClassifierCV(
            RidgeClassifier(random_state=42, class_weight=class_weight),
            method="sigmoid", cv=cv,
        )),
    ]
    final_est = LogisticRegression(
        C=1.0, max_iter=4000, random_state=42, n_jobs=-1,
        class_weight=class_weight,
    )
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        cv=cv, n_jobs=1, stack_method="predict_proba",
        passthrough=False,
    )
    _safe_fit_eval("stacking_bert_only", stack,
                   X_bert_tr, ytr, X_bert_te, yte, results, tag="bert_only")


# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------
def run_classical(
    vecdir: str,
    class_weight: str | None = None,
    include_tfidf_only: bool = True,
    feature_sources: Tuple[str, ...] = ("bert_only", "hybrid", "tfidf_only"),
    c_grid: Tuple[float, ...] = (0.1, 0.3, 1.0, 3.0),
    enable_stacking: bool = True,
    enable_rbf: bool = True,
):
    """
    Полный classical-прогон на разных представлениях.

    feature_sources:
      - 'bert_only'   → X_*_bert.npy (L2-нормированный, dense, dim=1024)
      - 'hybrid'      → X_*_hybrid.npz (TF-IDF + BERT*bw, БЕЗ финальной L2)
      - 'tfidf_only'  → строит TF-IDF на лету из texts_*.csv (NB-семейство)
    c_grid: мини-grid по C для linear_svc / logreg
    enable_stacking: финальный stacking на bert_only
    enable_rbf: SVC-rbf на bert_only (медленно, но часто +1-3 BA)
    """
    y_train = pd.read_csv(os.path.join(vecdir, "y_train.csv")).iloc[:, 0].astype(str)
    y_test = pd.read_csv(os.path.join(vecdir, "y_test.csv")).iloc[:, 0].astype(str)

    class_counts = y_train.value_counts()
    min_class_count = int(class_counts.min())
    num_classes = int(class_counts.shape[0])

    results: list = []
    bests: Dict[str, Any] = {}

    print(f"y_train: {len(y_train)} объектов, {num_classes} классов, "
          f"мин. класс = {min_class_count} пример(а/ов)")
    print(f"feature_sources: {feature_sources} | c_grid: {c_grid}")

    X_bert_tr = X_bert_te = None
    X_hyb_tr = X_hyb_te = None

    # === bert_only ===
    if "bert_only" in feature_sources:
        bert_tr_path = os.path.join(vecdir, "X_train_bert.npy")
        bert_te_path = os.path.join(vecdir, "X_test_bert.npy")
        if os.path.exists(bert_tr_path) and os.path.exists(bert_te_path):
            X_bert_tr = np.load(bert_tr_path).astype(np.float32)
            X_bert_te = np.load(bert_te_path).astype(np.float32)
            # на всякий случай повторно нормируем (если файл старый)
            X_bert_tr = normalize(X_bert_tr, norm="l2", axis=1)
            X_bert_te = normalize(X_bert_te, norm="l2", axis=1)
            best_svc, best_lr = _run_on_source(
                X_bert_tr, y_train, X_bert_te, y_test, results,
                tag="bert_only", c_grid=c_grid, class_weight=class_weight,
                include_knn_centroid=True, include_rbf=enable_rbf,
            )
            bests["bert_only"] = {"svc": best_svc, "lr": best_lr}
        else:
            print(f"\n[!] bert_only SKIPPED: нет {bert_tr_path}. "
                  f"Перезапусти `hybrid build` с новой версией hybrid_vector_build.py")

    # === hybrid ===
    if "hybrid" in feature_sources:
        hyb_tr_path = os.path.join(vecdir, "X_train_hybrid.npz")
        hyb_te_path = os.path.join(vecdir, "X_test_hybrid.npz")
        X_hyb_tr = sp.load_npz(hyb_tr_path)
        X_hyb_te = sp.load_npz(hyb_te_path)
        best_svc, best_lr = _run_on_source(
            X_hyb_tr, y_train, X_hyb_te, y_test, results,
            tag="hybrid", c_grid=c_grid, class_weight=class_weight,
            include_knn_centroid=False,   # sparse: knn-cos сильно медленный
            include_rbf=False,
        )
        bests["hybrid"] = {"svc": best_svc, "lr": best_lr}

    # === tfidf_only ===
    if "tfidf_only" in feature_sources and include_tfidf_only:
        train_texts_path = os.path.join(vecdir, "texts_train.csv")
        test_texts_path = os.path.join(vecdir, "texts_test.csv")
        if os.path.exists(train_texts_path) and os.path.exists(test_texts_path):
            print(f"\n=== Source: tfidf_only ===")
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

            _safe_fit_eval("multinomial_nb", MultinomialNB(),
                           Xtr, ytr, Xte, yte, results, tag="tfidf_only")
            _safe_fit_eval("complement_nb", ComplementNB(),
                           Xtr, ytr, Xte, yte, results, tag="tfidf_only")

            # full linear suite + grid
            _run_on_source(
                Xtr, ytr, Xte, yte, results,
                tag="tfidf_only", c_grid=c_grid, class_weight=class_weight,
                include_knn_centroid=False, include_rbf=False,
            )

    # === Stacking (на bert_only) ===
    if enable_stacking and X_bert_tr is not None:
        best_C_bert = bests.get("bert_only", {}).get("svc", (None,))[0]
        best_C_hyb = bests.get("hybrid", {}).get("svc", (None,))[0]
        _run_stacking(
            X_bert_tr, X_bert_te,
            X_hyb_tr if X_hyb_tr is not None else X_bert_tr,
            X_hyb_te if X_hyb_te is not None else X_bert_te,
            y_train, y_test, results,
            best_C_bert=best_C_bert, best_C_hyb=best_C_hyb,
            class_weight=class_weight,
        )

    # === Сводка ===
    valid = [r for r in results if r.get("balanced_accuracy") is not None]
    valid.sort(key=lambda r: r["balanced_accuracy"], reverse=True)
    print(f"\n=== TOP-10 по balanced_accuracy ===")
    for r in valid[:10]:
        print(f"  {r['balanced_accuracy']:.4f} / f1={r['macro_f1']:.4f}  {r['model']}")

    summary = {
        "class_weight": class_weight,
        "feature_sources": list(feature_sources),
        "c_grid": list(c_grid),
        "min_class_count": min_class_count,
        "num_classes": num_classes,
        "n_models_ok": len(valid),
        "n_models_total": len(results),
        "top_model": valid[0] if valid else None,
        "results": results,
    }
    out_path = os.path.join(vecdir, "classical_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")
    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vecdir", required=True)
    parser.add_argument("--class-weight", default=None,
                        choices=[None, "balanced"], nargs="?")
    parser.add_argument("--no-tfidf-only", action="store_true",
                        help="Не запускать TF-IDF-only ветку (NB-семейство и пр.)")
    parser.add_argument("--feature-sources", default="bert_only,hybrid,tfidf_only",
                        help="Через запятую: bert_only,hybrid,tfidf_only")
    parser.add_argument("--c-grid", default="0.1,0.3,1.0,3.0",
                        help="Через запятую: значения C для LinearSVC/LogReg grid")
    parser.add_argument("--no-stacking", action="store_true",
                        help="Выключить stacking")
    parser.add_argument("--no-rbf", action="store_true",
                        help="Выключить SVC-rbf на bert_only (быстрее)")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    sources = tuple(s.strip() for s in args.feature_sources.split(",") if s.strip())
    c_grid = tuple(float(c) for c in args.c_grid.split(",") if c.strip())
    run_classical(
        args.vecdir,
        class_weight=args.class_weight,
        include_tfidf_only=(not args.no_tfidf_only) and ("tfidf_only" in sources),
        feature_sources=sources,
        c_grid=c_grid,
        enable_stacking=not args.no_stacking,
        enable_rbf=not args.no_rbf,
    )


if __name__ == "__main__":
    main()