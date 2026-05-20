import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import pandas as pd
import torch

from cosine_similarity_classification.config import (
    MODEL_DIR,
    BASE_MODEL_NAME,
    TEXT_COLUMN,
    LABEL_COLUMN,
    METHOD,
    MAX_LENGTH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    POOLING,
    CHUNK_AGGREGATION,
    BATCH_SIZE,
    DEVICE,
    KNN_K,
    KNN_TEMPERATURE,
    KNN_K_SWEEP,
    KNN_T_SWEEP,
    CENTROID_TRIM_RATIO,
    CENTROID_TRIM_MODE,
    CENTROID_TRIM_POWER,
    CENTROID_REFINE_ITERS,
    ENSEMBLE_ALPHA,
)
from cosine_similarity_classification.embedder import (
    load_texts_and_labels,
    build_embedder,
    embed_dataframe,
)
from cosine_similarity_classification.classifiers import (
    predict_nearest,
    predict_nearest_sweep,
    predict_centroid,
    predict_centroid_nn_ensemble,
)
from cosine_similarity_classification.metrics import evaluate_predictions


METHOD_CHOICES = ["all", "centroid", "nearest", "centroid_nn"]


# v16: красивые имена методов. method_key (snake_case) — для group/source.
PRETTY_METHOD_NAMES = {
    "nearest": "Cosine-Nearest",
    "centroid": "Cosine-Centroid",
    "centroid_nn": "Cosine-CentroidNN",
}


def _pretty_method(method_key: str) -> str:
    return PRETTY_METHOD_NAMES.get(method_key, method_key)


def _test_has_label_column(test_file, label_col):
    test_head = pd.read_csv(test_file, nrows=1)
    return label_col in test_head.columns


def _detect_device(device=None):
    if device is not None:
        device = str(device).strip().lower()
        if device:
            return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_list(s, cast):
    if not s or not str(s).strip():
        return []
    return [cast(x) for x in str(s).split(",") if str(x).strip()]


# -----------------------------------------------------------------------------
# Per-method runners
# -----------------------------------------------------------------------------
def _run_centroid(train_embs, train_labels, test_embs, *,
                  trim_ratio, trim_mode, trim_power, refine_iters):
    pred_labels, pred_scores, classes, centroids = predict_centroid(
        train_embs=train_embs, train_labels=train_labels, query_embs=test_embs,
        trim_ratio=float(trim_ratio),
        trim_mode=trim_mode,
        trim_power=float(trim_power),
        refine_iters=int(refine_iters),
    )
    extra = {
        "num_classes": int(len(classes)),
        "centroid_dim": int(centroids.shape[1]),
        "centroid_trim_ratio": float(trim_ratio),
        "centroid_trim_mode": trim_mode,
        "centroid_trim_power": float(trim_power),
        "centroid_refine_iters": int(refine_iters),
    }
    return pred_labels, pred_scores, extra


def _run_nearest(train_embs, train_labels, test_embs, test_df, label_col,
                 test_has_labels, *,
                 knn_k, knn_temperature, knn_k_sweep, knn_t_sweep):
    k_list = _parse_list(knn_k_sweep, int) or [int(knn_k)]
    t_list = _parse_list(knn_t_sweep, float) or [float(knn_temperature)]

    sweep = predict_nearest_sweep(
        train_embs=train_embs, train_labels=train_labels, query_embs=test_embs,
        k_list=k_list, t_list=t_list,
    )

    best = sweep[0]
    if test_has_labels:
        from sklearn.metrics import f1_score
        y_true_tmp = test_df[label_col].astype(str).values
        best_f1 = -1.0
        for item in sweep:
            f1 = f1_score(y_true_tmp, item["pred_labels"], average="macro", zero_division=0)
            item["macro_f1"] = float(f1)
            if f1 > best_f1:
                best_f1 = f1
                best = item

    pred_labels = best["pred_labels"]
    pred_scores = best["pred_scores"]
    extra = {
        "knn_k_best": int(best["k"]),
        "knn_temperature_best": float(best["temperature"]),
        "knn_k_sweep": [int(k) for k in k_list],
        "knn_t_sweep": [float(t) for t in t_list],
        "sweep_table": [],  # заполним ниже, чтобы попало в JSON
    }

    if test_has_labels:
        from sklearn.metrics import f1_score, balanced_accuracy_score
        y_true_tmp = test_df[label_col].astype(str).values
        print(f"--- {_pretty_method('nearest')} sweep (k × T) ---")
        for item in sweep:
            ba = balanced_accuracy_score(y_true_tmp, item["pred_labels"])
            f1 = f1_score(y_true_tmp, item["pred_labels"], average="macro", zero_division=0)
            mark = " ← best" if (item["k"] == best["k"] and item["temperature"] == best["temperature"]) else ""
            print(f"  k={item['k']:>3} T={item['temperature']:.3f}: "
                  f"balanced_accuracy={ba:.6f}, macro_f1={f1:.6f}{mark}")
            extra["sweep_table"].append({
                "k": int(item["k"]),
                "temperature": float(item["temperature"]),
                "balanced_accuracy": round(float(ba), 6),
                "macro_f1": round(float(f1), 6),
                "is_best": (item["k"] == best["k"] and item["temperature"] == best["temperature"]),
            })

    return pred_labels, pred_scores, extra


def _run_centroid_nn(train_embs, train_labels, test_embs, *,
                     trim_ratio, trim_mode, trim_power, refine_iters,
                     knn_k, knn_temperature, ensemble_alpha):
    pred_labels, pred_scores, classes = predict_centroid_nn_ensemble(
        train_embs=train_embs, train_labels=train_labels, query_embs=test_embs,
        trim_ratio=float(trim_ratio),
        trim_mode=trim_mode,
        trim_power=float(trim_power),
        refine_iters=int(refine_iters),
        k=int(knn_k),
        temperature=float(knn_temperature),
        alpha=float(ensemble_alpha),
    )
    extra = {
        "num_classes": int(len(classes)),
        "centroid_trim_ratio": float(trim_ratio),
        "centroid_trim_mode": trim_mode,
        "centroid_refine_iters": int(refine_iters),
        "knn_k": int(knn_k),
        "knn_temperature": float(knn_temperature),
        "ensemble_alpha": float(ensemble_alpha),
    }
    return pred_labels, pred_scores, extra


def _eval_and_print(method_key, pred_labels, test_df, label_col, test_has_labels):
    pretty = _pretty_method(method_key)
    if test_has_labels:
        y_true = test_df[label_col].astype(str).values
        eval_metrics = evaluate_predictions(y_true, pred_labels)
        print(f"{pretty}: "
              f"{{'balanced_accuracy': {eval_metrics['balanced_accuracy']:.6f}, "
              f"'macro_f1': {eval_metrics['macro_f1']:.6f}}}")
        return eval_metrics
    else:
        print(f"{pretty}: predictions computed (test has no labels).")
        return {"balanced_accuracy": None, "macro_f1": None}


# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------
def run_from_params(
    train_file,
    test_file,
    model_dir=MODEL_DIR,
    base_model_name=BASE_MODEL_NAME,
    text_col=TEXT_COLUMN,
    label_col=LABEL_COLUMN,
    method=METHOD,
    max_length=MAX_LENGTH,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    pooling=POOLING,
    chunk_aggregation=CHUNK_AGGREGATION,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    knn_k: int = KNN_K,
    knn_temperature: float = KNN_TEMPERATURE,
    knn_k_sweep: str = KNN_K_SWEEP,
    knn_t_sweep: str = KNN_T_SWEEP,
    centroid_trim_ratio: float = CENTROID_TRIM_RATIO,
    centroid_trim_mode: str = CENTROID_TRIM_MODE,
    centroid_trim_power: float = CENTROID_TRIM_POWER,
    centroid_refine_iters: int = CENTROID_REFINE_ITERS,
    ensemble_alpha: float = ENSEMBLE_ALPHA,
    results_out: str | None = None,
):
    train_file = str(Path(train_file).expanduser())
    test_file = str(Path(test_file).expanduser())

    if model_dir is None:
        model_dir = ""
    else:
        model_dir = str(model_dir).strip()
        if model_dir:
            model_dir = str(Path(model_dir).expanduser())

    device = _detect_device(device)
    test_has_labels = _test_has_label_column(test_file, label_col)

    train_df = load_texts_and_labels(train_file, text_col=text_col, label_col=label_col, require_labels=True)
    test_df = load_texts_and_labels(test_file, text_col=text_col, label_col=label_col, require_labels=test_has_labels)

    # === Эмбеддинги считаем ОДИН раз ===
    embedder = build_embedder(
        model_dir=model_dir, base_model_name=base_model_name,
        max_length=max_length, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        pooling=pooling, chunk_aggregation=chunk_aggregation,
        batch_size=batch_size, device=device,
    )

    train_embs = embed_dataframe(train_df, embedder, text_col=text_col)
    test_embs = embed_dataframe(test_df, embedder, text_col=text_col)
    train_labels = train_df[label_col].astype(str).values

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] train_embs.shape={tuple(train_embs.shape)}, test_embs.shape={tuple(test_embs.shape)}")

    if method == "all":
        methods_to_run = ["centroid", "nearest", "centroid_nn"]
    elif method in METHOD_CHOICES:
        methods_to_run = [method]
    else:
        raise ValueError(f"method must be one of {METHOD_CHOICES}")

    per_method_results = {}

    for m in methods_to_run:
        pretty = _pretty_method(m)
        print(f"\n=== Method: {pretty} ===")
        if m == "centroid":
            pred_labels, pred_scores, extra = _run_centroid(
                train_embs, train_labels, test_embs,
                trim_ratio=centroid_trim_ratio, trim_mode=centroid_trim_mode,
                trim_power=centroid_trim_power, refine_iters=centroid_refine_iters,
            )
        elif m == "nearest":
            pred_labels, pred_scores, extra = _run_nearest(
                train_embs, train_labels, test_embs, test_df, label_col, test_has_labels,
                knn_k=knn_k, knn_temperature=knn_temperature,
                knn_k_sweep=knn_k_sweep, knn_t_sweep=knn_t_sweep,
            )
        elif m == "centroid_nn":
            pred_labels, pred_scores, extra = _run_centroid_nn(
                train_embs, train_labels, test_embs,
                trim_ratio=centroid_trim_ratio, trim_mode=centroid_trim_mode,
                trim_power=centroid_trim_power, refine_iters=centroid_refine_iters,
                knn_k=knn_k, knn_temperature=knn_temperature,
                ensemble_alpha=ensemble_alpha,
            )
        else:
            raise ValueError(f"Unknown method: {m}")

        eval_metrics = _eval_and_print(m, pred_labels, test_df, label_col, test_has_labels)

        per_method_results[m] = {
            "model": pretty,                 # v16: красивое имя
            "method": m,                     # snake_case остаётся как источник
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
            "balanced_accuracy": eval_metrics["balanced_accuracy"],
            "macro_f1": eval_metrics["macro_f1"],
            "pred_labels": pred_labels,
            "pred_scores": pred_scores,
            **extra,
        }

    # === Сводная таблица ===
    if len(methods_to_run) > 1 and test_has_labels:
        print("\n=== TOP по macro_f1 ===")
        rows = []
        for m, res in per_method_results.items():
            rows.append((_pretty_method(m), res["balanced_accuracy"], res["macro_f1"]))
        rows.sort(key=lambda r: r[2], reverse=True)
        for pretty, ba, f1 in rows:
            print(f"  {ba:.4f} / f1={f1:.4f}  {pretty}")

    # === Сохранение JSON ===
    # v16: имя файла = cosine-results-<basename(model_dir) or 'default'>-<YYYY-MM-DDTHH:MM>.json
    if results_out is None:
        model_tag = os.path.basename(os.path.normpath(model_dir)) if model_dir else "default"
        if not model_tag:
            model_tag = "default"
        stamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
        out_name = f"cosine-results-{model_tag}-{stamp}.json"
        results_out = str(Path(train_file).expanduser().parent / out_name)

    try:
        out_dump = {
            "model_dir": model_dir,
            "base_model_name": base_model_name,
            "device": device,
            "train_file": train_file,
            "test_file": test_file,
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
            "methods_ran": methods_to_run,
            "results": {
                m: {
                    k: v for k, v in res.items()
                    if k not in ("pred_labels", "pred_scores")
                }
                for m, res in per_method_results.items()
            },
        }
        with open(results_out, "w", encoding="utf-8") as f:
            json.dump(out_dump, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nSaved: {results_out}")
    except Exception as e:
        print(f"[WARN] не удалось сохранить {results_out}: {e}")

    # === Возврат: если был один метод — back-compat ===
    if len(methods_to_run) == 1:
        m = methods_to_run[0]
        res = per_method_results[m]
        components = {
            "method": m,
            "model": _pretty_method(m),
            "model_dir": model_dir,
            "num_train": int(len(train_df)),
            "num_test": int(len(test_df)),
            "train_embeddings_shape": tuple(train_embs.shape),
            "test_embeddings_shape": tuple(test_embs.shape),
            "pred_labels": res["pred_labels"],
            "pred_scores": res["pred_scores"],
        }
        metrics = {
            "method": m,
            "model": _pretty_method(m),
            "train_size": res["train_size"],
            "test_size": res["test_size"],
            "model_dir": model_dir,
            "base_model_name": base_model_name,
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "pooling": pooling,
            "chunk_aggregation": chunk_aggregation,
            "batch_size": int(batch_size),
            "device": device,
            **{k: v for k, v in res.items() if k not in (
                "pred_labels", "pred_scores", "method", "model",
                "train_size", "test_size",
            )},
        }
        return components, metrics

    return {"per_method": per_method_results}, {
        "device": device, "model_dir": model_dir,
        "methods_ran": methods_to_run,
        "per_method_metrics": {
            _pretty_method(m): {
                "balanced_accuracy": r["balanced_accuracy"],
                "macro_f1": r["macro_f1"],
            }
            for m, r in per_method_results.items()
        },
    }


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Cosine similarity classification over embeddings"
    )
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--base-model-name", type=str, default=BASE_MODEL_NAME)
    parser.add_argument("--text-col", type=str, default=TEXT_COLUMN)
    parser.add_argument("--label-col", type=str, default=LABEL_COLUMN)
    parser.add_argument("--method", type=str,
                        choices=METHOD_CHOICES,
                        default=METHOD,
                        help='Default "all" — прогон всех методов. Или укажи один: '
                             'centroid / nearest / centroid_nn.')

    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--pooling", type=str,
                        choices=["mean", "cls", "max", "mean_max"], default=POOLING)
    parser.add_argument("--chunk-aggregation", type=str,
                        choices=["mean", "max", "mean_max"], default=CHUNK_AGGREGATION)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)

    parser.add_argument("--knn-k", type=int, default=KNN_K)
    parser.add_argument("--knn-temperature", type=float, default=KNN_TEMPERATURE)
    parser.add_argument("--knn-k-sweep", type=str, default=KNN_K_SWEEP)
    parser.add_argument("--knn-t-sweep", type=str, default=KNN_T_SWEEP)

    parser.add_argument("--centroid-trim", type=float, default=CENTROID_TRIM_RATIO)
    parser.add_argument("--centroid-trim-mode", type=str,
                        choices=["hard", "soft"], default=CENTROID_TRIM_MODE)
    parser.add_argument("--centroid-trim-power", type=float, default=CENTROID_TRIM_POWER)
    parser.add_argument("--centroid-refine-iters", type=int, default=CENTROID_REFINE_ITERS)

    parser.add_argument("--ensemble-alpha", type=float, default=ENSEMBLE_ALPHA)

    parser.add_argument("--results-out", type=str, default=None,
                        help="Куда сохранить JSON. По умолчанию рядом с train.csv: "
                             "cosine-results-<model_dir basename or 'default'>-<datetime>.json")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    print(f"[ARGS] {vars(args)}")

    run_from_params(
        train_file=args.train,
        test_file=args.test,
        model_dir=args.model_dir,
        base_model_name=args.base_model_name,
        text_col=args.text_col,
        label_col=args.label_col,
        method=args.method,
        max_length=args.max_length,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        pooling=args.pooling,
        chunk_aggregation=args.chunk_aggregation,
        batch_size=args.batch_size,
        device=args.device,
        knn_k=args.knn_k,
        knn_temperature=args.knn_temperature,
        knn_k_sweep=args.knn_k_sweep,
        knn_t_sweep=args.knn_t_sweep,
        centroid_trim_ratio=args.centroid_trim,
        centroid_trim_mode=args.centroid_trim_mode,
        centroid_trim_power=args.centroid_trim_power,
        centroid_refine_iters=args.centroid_refine_iters,
        ensemble_alpha=args.ensemble_alpha,
        results_out=args.results_out,
    )


if __name__ == "__main__":
    main()