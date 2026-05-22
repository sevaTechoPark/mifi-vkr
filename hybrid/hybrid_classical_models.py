import os
import json
import argparse
import warnings
from datetime import datetime
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.svm import LinearSVC


# -----------------------------------------------------------------------------
# v17: оставляем только три линейных семейства — LinearSVC, LogisticRegression,
# RidgeClassifier. SGD/RBF/NB/Voting/Stacking удалены. Полный перебор C-grid и
# модификации (L1/L2, калибровка) сохранены.
# Источники остаются snake_case.
# -----------------------------------------------------------------------------
PRETTY_NAMES = {
    "linear_svc": "LinearSVC",
    "linear_svc_l1": "LinearSVC-L1",
    "linear_svc_calibrated_sigmoid": "LinearSVC-CalibratedSigmoid",
    "linear_svc_calibrated_isotonic": "LinearSVC-CalibratedIsotonic",
    "logreg": "LogisticRegression",
    "logreg_l1": "LogisticRegression-L1",
    "ridge_classifier": "RidgeClassifier",
}


def _pretty(model_key: str) -> str:
    """linear_svc → LinearSVC; linear_svc_l1 → LinearSVC-L1; etc."""
    return PRETTY_NAMES.get(model_key, model_key)


def _fmt_model(model_key: str, tag: str = "", C: float | None = None, extra: str = "") -> str:
    """
    Финальное имя модели:
      'LinearSVC[bert_only] C=3.0'
      'LogisticRegression-L1[hybrid] C=1.0'
    """
    pretty = _pretty(model_key)
    name = f"{pretty}[{tag}]" if tag else pretty
    parts = [name]
    if C is not None:
        parts.append(f"C={C}")
    if extra:
        parts.append(extra)
    return " ".join(parts)


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


def _safe_fit_eval(model_key, model, Xtr, ytr, Xte, yte, results, tag="", C=None, extra=""):
    full_name = _fmt_model(model_key, tag=tag, C=C, extra=extra)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        metrics = _eval(yte, pred)
        print(f"  {full_name}: {metrics}")
        results.append({
            "model": full_name,
            "model_key": model_key,
            "feature_source": tag,
            "C": C,
            **metrics,
        })
        return model, metrics
    except Exception as e:
        print(f"  {full_name}: FAILED ({type(e).__name__}: {e})")
        results.append({
            "model": full_name,
            "model_key": model_key,
            "feature_source": tag,
            "C": C,
            "balanced_accuracy": None, "macro_f1": None,
            "error": f"{type(e).__name__}: {e}",
        })
        return None, None


# -----------------------------------------------------------------------------
# C-grid for linear models
# -----------------------------------------------------------------------------
def _grid_linear_svc(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight):
    best = (None, None, None)
    base_kwargs = {"random_state": 42, "max_iter": 10000, "dual": False}
    if class_weight is not None:
        base_kwargs["class_weight"] = class_weight

    for C in c_grid:
        model = LinearSVC(C=C, **base_kwargs)
        m, metrics = _safe_fit_eval(
            "linear_svc", model, Xtr, ytr, Xte, yte, results, tag=tag, C=C,
        )
        if metrics and (best[2] is None or metrics["balanced_accuracy"] > best[2]["balanced_accuracy"]):
            best = (C, m, metrics)
    if best[0] is not None:
        print(f"  ★ best LinearSVC[{tag}]: C={best[0]} → {best[2]}")
        results.append({
            "model": _fmt_model("linear_svc", tag=tag, C=best[0], extra="★best"),
            "model_key": "linear_svc_best",
            "feature_source": tag,
            "best_C": best[0],
            **best[2],
        })
    return best


def _grid_linear_svc_l1(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight):
    """LinearSVC с penalty=l1 (sparse weights, другая регуляризация)."""
    best = (None, None, None)
    for C in c_grid:
        kwargs = {
            "C": C, "random_state": 42, "max_iter": 10000,
            "penalty": "l1", "dual": False, "loss": "squared_hinge",
        }
        if class_weight is not None:
            kwargs["class_weight"] = class_weight
        model = LinearSVC(**kwargs)
        m, metrics = _safe_fit_eval(
            "linear_svc_l1", model, Xtr, ytr, Xte, yte, results, tag=tag, C=C,
        )
        if metrics and (best[2] is None or metrics["balanced_accuracy"] > best[2]["balanced_accuracy"]):
            best = (C, m, metrics)
    if best[0] is not None:
        print(f"  ★ best LinearSVC-L1[{tag}]: C={best[0]} → {best[2]}")
    return best


def _grid_logreg(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight):
    best = (None, None, None)
    for C in c_grid:
        base_kwargs = {
            "C": C, "max_iter": 4000, "random_state": 42,
            "solver": "lbfgs",
        }
        if class_weight is not None:
            base_kwargs["class_weight"] = class_weight
        model = LogisticRegression(**base_kwargs)
        m, metrics = _safe_fit_eval(
            "logreg", model, Xtr, ytr, Xte, yte, results, tag=tag, C=C,
        )
        if metrics and (best[2] is None or metrics["balanced_accuracy"] > best[2]["balanced_accuracy"]):
            best = (C, m, metrics)
    if best[0] is not None:
        print(f"  ★ best LogisticRegression[{tag}]: C={best[0]} → {best[2]}")
        results.append({
            "model": _fmt_model("logreg", tag=tag, C=best[0], extra="★best"),
            "model_key": "logreg_best",
            "feature_source": tag,
            "best_C": best[0],
            **best[2],
        })
    return best


def _grid_logreg_l1(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight):
    """
    v14: penalty='elasticnet' + l1_ratio=1.0 — новый API sklearn 1.8+,
    эквивалентно penalty='l1', но без FutureWarning.
    """
    best = (None, None, None)
    for C in c_grid:
        kwargs = {
            "C": C, "max_iter": 4000, "random_state": 42,
            "penalty": "elasticnet", "l1_ratio": 1.0, "solver": "saga",
        }
        if class_weight is not None:
            kwargs["class_weight"] = class_weight
        model = LogisticRegression(**kwargs)
        m, metrics = _safe_fit_eval(
            "logreg_l1", model, Xtr, ytr, Xte, yte, results, tag=tag, C=C,
        )
        if metrics and (best[2] is None or metrics["balanced_accuracy"] > best[2]["balanced_accuracy"]):
            best = (C, m, metrics)
    if best[0] is not None:
        print(f"  ★ best LogisticRegression-L1[{tag}]: C={best[0]} → {best[2]}")
    return best


# -----------------------------------------------------------------------------
# Per-source pipeline
# -----------------------------------------------------------------------------
def _run_on_source(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight,
                   include_l1=True):
    print(f"\n=== Source: {tag} | shape={Xtr.shape} | class_weight={class_weight!r} ===")

    best_svc = _grid_linear_svc(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight)
    best_lr = _grid_logreg(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight)

    if include_l1:
        _grid_linear_svc_l1(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight)
        if not sp.issparse(Xtr):
            _grid_logreg_l1(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight)

    # Calibrated SVC (sigmoid + isotonic) на лучшем C
    if best_svc[0] is not None:
        class_counts = pd.Series(ytr).value_counts()
        min_class_count = int(class_counts.min())
        if min_class_count >= 2:
            safe_cv = max(2, min(3, min_class_count))
            cal_kwargs = {"random_state": 42, "max_iter": 10000, "dual": False, "C": best_svc[0]}
            if class_weight is not None:
                cal_kwargs["class_weight"] = class_weight
            for method in ("sigmoid", "isotonic"):
                model_key = f"linear_svc_calibrated_{method}"
                _safe_fit_eval(
                    model_key,
                    CalibratedClassifierCV(LinearSVC(**cal_kwargs), method=method, cv=safe_cv),
                    Xtr, ytr, Xte, yte, results, tag=tag, C=best_svc[0],
                )

    # RidgeClassifier
    base = {"random_state": 42}
    if class_weight is not None:
        base["class_weight"] = class_weight
    _safe_fit_eval("ridge_classifier", RidgeClassifier(**base),
                   Xtr, ytr, Xte, yte, results, tag=tag)

    return best_svc, best_lr


# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------
def run_classical(
    vecdir: str,
    class_weight: str | None = "balanced",
    include_tfidf_only: bool = True,
    feature_sources: Tuple[str, ...] = ("bert_only", "hybrid", "tfidf_only"),
    c_grid: Tuple[float, ...] = (0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0),
):
    """
    Полный classical-прогон. Cosine-методы убраны — для них есть отдельный модуль.

    v14+: class_weight по умолчанию 'balanced' — на min_class=1 это критично для macro_f1.
    v16: красивые имена моделей в логах и JSON; имя файла включает basename(vecdir) + datetime.
    v17: оставлены только три линейных семейства (LinearSVC, LogisticRegression,
         RidgeClassifier) со всеми модификациями (L1/L2, калибровка) и полным
         перебором C-grid. SGD/RBF/NB/Voting/Stacking удалены.

    feature_sources:
      - 'bert_only'   → X_*_bert.npy (L2-нормированный, dense, dim=1024)
      - 'hybrid'      → X_*_hybrid_noL2.npz (per-block L2, без финальной L2)
      - 'tfidf_only'  → строит TF-IDF на лету из texts_*.csv
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
    print(f"feature_sources: {feature_sources} | c_grid: {c_grid} | class_weight={class_weight!r}")

    # === bert_only ===
    if "bert_only" in feature_sources:
        bert_tr_path = os.path.join(vecdir, "X_train_bert.npy")
        bert_te_path = os.path.join(vecdir, "X_test_bert.npy")
        if os.path.exists(bert_tr_path) and os.path.exists(bert_te_path):
            X_bert_tr = np.load(bert_tr_path).astype(np.float32)
            X_bert_te = np.load(bert_te_path).astype(np.float32)
            from sklearn.preprocessing import normalize
            X_bert_tr = normalize(X_bert_tr, norm="l2", axis=1)
            X_bert_te = normalize(X_bert_te, norm="l2", axis=1)
            best_svc, best_lr = _run_on_source(
                X_bert_tr, y_train, X_bert_te, y_test, results,
                tag="bert_only", c_grid=c_grid, class_weight=class_weight,
                include_l1=True,
            )
            bests["bert_only"] = {"svc": best_svc, "lr": best_lr}
        else:
            print(f"\n[!] bert_only SKIPPED: нет {bert_tr_path}. "
                  f"Перезапусти `hybrid build`.")

    # === hybrid ===
    if "hybrid" in feature_sources:
        noL2_tr = os.path.join(vecdir, "X_train_hybrid_noL2.npz")
        noL2_te = os.path.join(vecdir, "X_test_hybrid_noL2.npz")
        if os.path.exists(noL2_tr) and os.path.exists(noL2_te):
            print("[hybrid] using X_train_hybrid_noL2.npz (per-block L2, no final L2)")
            X_hyb_tr = sp.load_npz(noL2_tr)
            X_hyb_te = sp.load_npz(noL2_te)
        else:
            print("[hybrid] fallback: using X_train_hybrid.npz (final L2 applied)")
            X_hyb_tr = sp.load_npz(os.path.join(vecdir, "X_train_hybrid.npz"))
            X_hyb_te = sp.load_npz(os.path.join(vecdir, "X_test_hybrid.npz"))
        best_svc, best_lr = _run_on_source(
            X_hyb_tr, y_train, X_hyb_te, y_test, results,
            tag="hybrid", c_grid=c_grid, class_weight=class_weight,
            include_l1=True,
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

            _run_on_source(
                Xtr, ytr, Xte, yte, results,
                tag="tfidf_only", c_grid=c_grid, class_weight=class_weight,
                include_l1=True,
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

    # v16: имя файла = classical-results-<basename(vecdir)>-<YYYY-MM-DDTHH:MM>.json
    vec_base = os.path.basename(os.path.normpath(vecdir)) or "vecdir"
    stamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
    out_name = f"classical-results-{vec_base}-{stamp}.json"
    out_path = os.path.join(vecdir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")
    return results


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vecdir", required=True)
    parser.add_argument("--class-weight", default="balanced",
                        choices=["balanced", "none"], nargs="?")
    parser.add_argument("--no-tfidf-only", action="store_true")
    parser.add_argument("--feature-sources", default="bert_only,hybrid,tfidf_only")
    parser.add_argument("--c-grid", default="0.05,0.1,0.3,0.5,1.0,2.0,3.0,5.0")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    sources = tuple(s.strip() for s in args.feature_sources.split(",") if s.strip())
    c_grid = tuple(float(c) for c in args.c_grid.split(",") if c.strip())
    cw = None if (args.class_weight in (None, "none")) else args.class_weight
    run_classical(
        args.vecdir,
        class_weight=cw,
        include_tfidf_only=(not args.no_tfidf_only) and ("tfidf_only" in sources),
        feature_sources=sources,
        c_grid=c_grid,
    )


if __name__ == "__main__":
    main()
