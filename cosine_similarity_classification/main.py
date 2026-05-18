import argparse
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
)
from cosine_similarity_classification.embedder import (
    load_texts_and_labels,
    build_embedder,
    embed_dataframe,
)
from cosine_similarity_classification.classifiers import (
    predict_nearest,
    predict_centroid,
)
from cosine_similarity_classification.metrics import evaluate_predictions


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
):
    train_file = str(Path(train_file).expanduser())
    test_file = str(Path(test_file).expanduser())

    if model_dir is None:
        model_dir = ""
    else:
        model_dir = str(model_dir).strip()
        if model_dir:
            model_dir = str(Path(model_dir).expanduser())
        else:
            model_dir = ""

    device = _detect_device(device)

    test_has_labels = _test_has_label_column(test_file, label_col)

    train_df = load_texts_and_labels(
        train_file,
        text_col=text_col,
        label_col=label_col,
        require_labels=True,
    )

    test_df = load_texts_and_labels(
        test_file,
        text_col=text_col,
        label_col=label_col,
        require_labels=test_has_labels,
    )

    embedder = build_embedder(
        model_dir=model_dir,
        base_model_name=base_model_name,
        max_length=max_length,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        pooling=pooling,
        chunk_aggregation=chunk_aggregation,
        batch_size=batch_size,
        device=device,
    )

    train_embs = embed_dataframe(train_df, embedder, text_col=text_col)
    test_embs = embed_dataframe(test_df, embedder, text_col=text_col)

    train_labels = train_df[label_col].astype(str).values

    if method == "nearest":
        pred_labels, pred_scores = predict_nearest(
            train_embs=train_embs,
            train_labels=train_labels,
            query_embs=test_embs,
            k=knn_k,
        )
        extra = {"knn_k": int(min(knn_k, len(train_df)))}
    elif method == "centroid":
        pred_labels, pred_scores, classes, centroids = predict_centroid(
            train_embs=train_embs,
            train_labels=train_labels,
            query_embs=test_embs,
        )
        extra = {
            "num_classes": int(len(classes)),
            "centroid_dim": int(centroids.shape[1]),
        }
    else:
        raise ValueError("method must be one of: ['nearest', 'centroid']")

    metrics = {
        "method": method,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "model_dir": model_dir,
        "base_model_name": base_model_name,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "pooling": pooling,
        "chunk_aggregation": chunk_aggregation,
        "batch_size": int(batch_size),
        "device": device,
        **extra,
    }

    if test_has_labels:
        y_true = test_df[label_col].astype(str).values
        eval_metrics = evaluate_predictions(y_true, pred_labels)
        metrics["balanced_accuracy"] = eval_metrics["balanced_accuracy"]
        metrics["macro_f1"] = eval_metrics["macro_f1"]
    else:
        metrics["balanced_accuracy"] = None
        metrics["macro_f1"] = None

    components = {
        "method": method,
        "model_dir": model_dir,
        "num_train": int(len(train_df)),
        "num_test": int(len(test_df)),
        "train_embeddings_shape": tuple(train_embs.shape),
        "test_embeddings_shape": tuple(test_embs.shape),
        "pred_labels": pred_labels,
        "pred_scores": pred_scores,
    }

    print(f"[INFO] Using device: {device}")

    if test_has_labels:
        print(
            f"{method}: "
            f"{{'balanced_accuracy': {metrics['balanced_accuracy']:.6f}, "
            f"'macro_f1': {metrics['macro_f1']:.6f}}}"
        )
    else:
        print(f"{method}: predictions computed (test has no labels).")

    return components, metrics


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Cosine similarity classification over embeddings"
    )
    parser.add_argument("--train", type=str, required=True, help="Path to train CSV")
    parser.add_argument("--test", type=str, required=True, help="Path to test CSV")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=MODEL_DIR,
        help="Directory with local embedding model weights",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--base-model-name", type=str, default=BASE_MODEL_NAME)
    parser.add_argument("--text-col", type=str, default=TEXT_COLUMN)
    parser.add_argument("--label-col", type=str, default=LABEL_COLUMN)
    parser.add_argument("--method", type=str, choices=["nearest", "centroid"], default=METHOD)

    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--pooling", type=str, choices=["mean", "cls", "max", "mean_max"], default=POOLING)
    parser.add_argument("--chunk-aggregation", type=str, choices=["mean", "max", "mean_max"], default=CHUNK_AGGREGATION)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--knn-k", type=int, default=KNN_K, help="k для метода 'nearest' (kNN cosine, weights=distance)")

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    print(f"[ARGS] {vars(args)}")   # видно сразу, не съел ли shell какой-нибудь --model-dir

    _, metrics = run_from_params(
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
    )

    print()
    print("run_summary:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()