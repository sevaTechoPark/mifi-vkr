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
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import StackingClassifier, VotingClassifier


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
            warnings.simplefilter("ignore", category=FutureWarning)
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
    best = (None, None, None)
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
            f"linear_svc_l1_C{C}", model, Xtr, ytr, Xte, yte, results, tag=tag
        )
        if metrics and (best[2] is None or metrics["balanced_accuracy"] > best[2]["balanced_accuracy"]):
            best = (C, m, metrics)
    if best[0] is not None:
        print(f"  ★ best linear_svc_l1[{tag}]: C={best[0]} → {best[2]}")
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


def _grid_logreg_l1(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight):
    """
    v14: используем penalty='elasticnet' + l1_ratio=1.0 — новый API sklearn 1.8+,
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
            f"logreg_l1_C{C}", model, Xtr, ytr, Xte, yte, results, tag=tag
        )
        if metrics and (best[2] is None or metrics["balanced_accuracy"] > best[2]["balanced_accuracy"]):
            best = (C, m, metrics)
    if best[0] is not None:
        print(f"  ★ best logreg_l1[{tag}]: C={best[0]} → {best[2]}")
    return best


# -----------------------------------------------------------------------------
# Per-source pipeline
# -----------------------------------------------------------------------------
def _run_on_source(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight,
                   include_rbf=False, include_l1=True, include_voting=True):
    print(f"\n=== Source: {tag} | shape={Xtr.shape} | class_weight={class_weight!r} ===")

    best_svc = _grid_linear_svc(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight)
    best_lr = _grid_logreg(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight)

    if include_l1:
        # l1-варианты часто полезны как другая регуляризация
        _grid_linear_svc_l1(Xtr, ytr, Xte, yte, results, tag, c_grid, class_weight)
        if not sp.issparse(Xtr):
            # saga на больших sparse очень медленно — только для dense (bert_only)
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
                _safe_fit_eval(
                    f"linear_svc_calibrated_{method}_C{best_svc[0]}",
                    CalibratedClassifierCV(LinearSVC(**cal_kwargs), method=method, cv=safe_cv),
                    Xtr, ytr, Xte, yte, results, tag=tag,
                )

    # RidgeClassifier
    base = {"random_state": 42}
    if class_weight is not None:
        base["class_weight"] = class_weight
    _safe_fit_eval("ridge_classifier", RidgeClassifier(**base),
                   Xtr, ytr, Xte, yte, results, tag=tag)

    # SGD hinge
    sgd_kwargs = {"random_state": 42, "max_iter": 2000, "alpha": 1e-5,
                  "loss": "hinge"}
    if class_weight is not None:
        sgd_kwargs["class_weight"] = class_weight
    _safe_fit_eval("sgd_hinge", SGDClassifier(**sgd_kwargs),
                   Xtr, ytr, Xte, yte, results, tag=tag)

    # SVC-rbf — только для dense bert_only
    if include_rbf and not sp.issparse(Xtr):
        rbf_kwargs = {"random_state": 42, "kernel": "rbf", "C": 4.0, "gamma": "scale"}
        if class_weight is not None:
            rbf_kwargs["class_weight"] = class_weight
        _safe_fit_eval("svc_rbf_C4", SVC(**rbf_kwargs),
                       Xtr, ytr, Xte, yte, results, tag=tag)

    # VotingClassifier на топ-3 моделях (только если есть best_svc и best_lr)
    if include_voting and best_svc[0] is not None and best_lr[0] is not None:
        class_counts = pd.Series(ytr).value_counts()
        min_cc = int(class_counts.min())
        if min_cc >= 2:
            safe_cv = max(2, min(3, min_cc))
            cw = class_weight
            svc_cal = CalibratedClassifierCV(
                LinearSVC(C=best_svc[0], max_iter=10000, dual=False,
                          random_state=42, class_weight=cw),
                method="sigmoid", cv=safe_cv,
            )
            lr = LogisticRegression(
                C=best_lr[0], max_iter=4000, random_state=42,
                solver="lbfgs", class_weight=cw,
            )
            ridge_cal = CalibratedClassifierCV(
                RidgeClassifier(random_state=42, class_weight=cw),
                method="sigmoid", cv=safe_cv,
            )
            voter = VotingClassifier(
                estimators=[("svc", svc_cal), ("lr", lr), ("ridge", ridge_cal)],
                voting="soft", n_jobs=1,
            )
            _safe_fit_eval(f"voting_soft_top3", voter,
                           Xtr, ytr, Xte, yte, results, tag=tag)

    return best_svc, best_lr


# -----------------------------------------------------------------------------
# Stacking (на bert_only, требует min_class >=2)
# -----------------------------------------------------------------------------
def _run_stacking(X_bert_tr, X_bert_te, ytr, yte, results, best_C_bert, class_weight):
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

    estimators = [
        ("svc_cal", CalibratedClassifierCV(
            LinearSVC(C=(best_C_bert or 1.0), **base_svc), method="sigmoid", cv=cv
        )),
        ("logreg", LogisticRegression(
            C=(best_C_bert or 1.0), max_iter=4000, random_state=42,
            class_weight=class_weight,
        )),
        ("ridge", CalibratedClassifierCV(
            RidgeClassifier(random_state=42, class_weight=class_weight),
            method="sigmoid", cv=cv,
        )),
    ]
    final_est = LogisticRegression(
        C=1.0, max_iter=4000, random_state=42,
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
    class_weight: str | None = "balanced",   # v14: default 'balanced' (было None)
    include_tfidf_only: bool = True,
    feature_sources: Tuple[str, ...] = ("bert_only", "hybrid", "tfidf_only"),
    c_grid: Tuple[float, ...] = (0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0),
    enable_stacking: bool = True,
    enable_rbf: bool = True,
):
    """
    Полный classical-прогон. Cosine-методы убраны — для них есть отдельный модуль.

    v14: class_weight по умолчанию 'balanced' — на min_class=1 это критично для macro_f1.
    Передай явно class_weight=None для прежнего поведения.

    feature_sources:
      - 'bert_only'   → X_*_bert.npy (L2-нормированный, dense, dim=1024)
      - 'hybrid'      → X_*_hybrid_noL2.npz (per-block L2, без финальной L2)
      - 'tfidf_only'  → строит TF-IDF на лету из texts_*.csv (NB + linear suite)
    c_grid: расширенный grid по C (was 0.1-3, now 0.05-5)
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

    X_bert_tr = X_bert_te = None

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
                include_rbf=enable_rbf, include_l1=True, include_voting=True,
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
            include_rbf=False, include_l1=True, include_voting=True,
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

            _run_on_source(
                Xtr, ytr, Xte, yte, results,
                tag="tfidf_only", c_grid=c_grid, class_weight=class_weight,
                include_rbf=False, include_l1=True, include_voting=False,
            )

    # === Stacking (на bert_only) ===
    if enable_stacking and X_bert_tr is not None:
        best_C_bert = bests.get("bert_only", {}).get("svc", (None,))[0]
        _run_stacking(
            X_bert_tr, X_bert_te, y_train, y_test, results,
            best_C_bert=best_C_bert, class_weight=class_weight,
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


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vecdir", required=True)
    # v14: default='balanced' (раньше None)
    parser.add_argument("--class-weight", default="balanced",
                        choices=["balanced", "none"], nargs="?")
    parser.add_argument("--no-tfidf-only", action="store_true")
    parser.add_argument("--feature-sources", default="bert_only,hybrid,tfidf_only")
    parser.add_argument("--c-grid", default="0.05,0.1,0.3,0.5,1.0,2.0,3.0,5.0")
    parser.add_argument("--no-stacking", action="store_true")
    parser.add_argument("--no-rbf", action="store_true")
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
        enable_stacking=not args.no_stacking,
        enable_rbf=not args.no_rbf,
    )


if __name__ == "__main__":
    main()