import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from transformers import AutoTokenizer, RobertaModel

from bert_embeddings.embedding_model import LongTextRobertaEmbedder


MODEL_NAME = "ai-forever/ruRoberta-large"
TEXT_COL = "text"
LABEL_COL = "label"

MAX_LENGTH = 512
CHUNK_SIZE = 448
CHUNK_OVERLAP = 96
BERT_BATCH_SIZE = 8
BERT_WEIGHT = 5.0


def load_and_clean_df(path, text_col=TEXT_COL, label_col=LABEL_COL, require_labels=True):
    df = pd.read_csv(path)

    required_cols = [text_col]
    if require_labels:
        required_cols.append(label_col)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df.dropna(subset=[text_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].reset_index(drop=True)

    if require_labels:
        df = df.dropna(subset=[label_col]).copy()
        df[label_col] = df[label_col].astype(str).str.strip()
        df = df[df[label_col] != ""].reset_index(drop=True)

    return df


def build_tfidf(X_train, X_test):
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w\w+\b",
        lowercase=True,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        lowercase=True,
        sublinear_tf=True,
    )

    X_train_word = word_tfidf.fit_transform(X_train)
    X_test_word = word_tfidf.transform(X_test)

    X_train_char = char_tfidf.fit_transform(X_train)
    X_test_char = char_tfidf.transform(X_test)

    X_train_tfidf = sp.hstack([X_train_word, X_train_char]).tocsr()
    X_test_tfidf = sp.hstack([X_test_word, X_test_char]).tocsr()

    X_train_tfidf = normalize(X_train_tfidf, norm="l2")
    X_test_tfidf = normalize(X_test_tfidf, norm="l2")

    return X_train_tfidf, X_test_tfidf, word_tfidf, char_tfidf


def build_embedder(
    model_dir="",
    base_model_name=MODEL_NAME,
    max_length=MAX_LENGTH,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    pooling="mean_max",
    chunk_aggregation="mean_max",
    batch_size=BERT_BATCH_SIZE,
):
    return LongTextRobertaEmbedder(
        model_dir=model_dir,
        base_model_name=base_model_name,
        max_length=max_length,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        pooling=pooling,
        chunk_aggregation=chunk_aggregation,
        batch_size=batch_size,
    )


def document_bert_embeddings_from_model_dir(
    texts,
    model_dir,
    base_model_name=MODEL_NAME,
    max_length=MAX_LENGTH,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    pooling="mean_max",
    chunk_aggregation="mean_max",
    batch_size=BERT_BATCH_SIZE,
):
    embedder = build_embedder(
        model_dir=model_dir,
        base_model_name=base_model_name,
        max_length=max_length,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        pooling=pooling,
        chunk_aggregation=chunk_aggregation,
        batch_size=batch_size,
    )
    return embedder.encode([str(t) for t in texts]).astype(np.float32)


@torch.no_grad()
def document_bert_embeddings_base(texts, tokenizer, model, device, batch_size=BERT_BATCH_SIZE, max_length=MAX_LENGTH):
    all_vecs = []
    texts = [str(t) for t in texts]
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded, return_dict=True)

        mask = encoded["attention_mask"].unsqueeze(-1).type_as(outputs.last_hidden_state)
        masked = outputs.last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts

        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_vecs.append(pooled.cpu().numpy().astype(np.float32))

    return np.vstack(all_vecs)


def run_build(
    train_file,
    test_file,
    outdir,
    model_dir=None,
    device="cpu",
    bert_weight=BERT_WEIGHT,
    base_model_name=MODEL_NAME,
    text_col=TEXT_COL,
    label_col=LABEL_COL,
    max_length=MAX_LENGTH,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    pooling="mean_max",
    chunk_aggregation="mean_max",
    batch_size=BERT_BATCH_SIZE,
):
    os.makedirs(outdir, exist_ok=True)

    train_df = load_and_clean_df(train_file, text_col=text_col, label_col=label_col, require_labels=True)
    test_df = load_and_clean_df(test_file, text_col=text_col, label_col=label_col, require_labels=True)

    X_train = train_df[text_col]
    y_train = train_df[label_col]
    X_test = test_df[text_col]
    y_test = test_df[label_col]

    X_train_tfidf, X_test_tfidf, word_tfidf, char_tfidf = build_tfidf(X_train, X_test)

    if model_dir:
        X_train_bert = document_bert_embeddings_from_model_dir(
            X_train.tolist(),
            model_dir=model_dir,
            base_model_name=base_model_name,
            max_length=max_length,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            pooling=pooling,
            chunk_aggregation=chunk_aggregation,
            batch_size=batch_size,
        )
        X_test_bert = document_bert_embeddings_from_model_dir(
            X_test.tolist(),
            model_dir=model_dir,
            base_model_name=base_model_name,
            max_length=max_length,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            pooling=pooling,
            chunk_aggregation=chunk_aggregation,
            batch_size=batch_size,
        )
    else:
        torch_device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = RobertaModel.from_pretrained(base_model_name).to(torch_device)

        X_train_bert = document_bert_embeddings_base(
            X_train.tolist(),
            tokenizer,
            model,
            torch_device,
            batch_size=batch_size,
            max_length=max_length,
        )
        X_test_bert = document_bert_embeddings_base(
            X_test.tolist(),
            tokenizer,
            model,
            torch_device,
            batch_size=batch_size,
            max_length=max_length,
        )

    scaler_bert = StandardScaler()
    X_train_bert = scaler_bert.fit_transform(X_train_bert).astype(np.float32)
    X_test_bert = scaler_bert.transform(X_test_bert).astype(np.float32)

    X_train_hybrid = sp.hstack([
        X_train_tfidf,
        sp.csr_matrix(X_train_bert * bert_weight)
    ]).tocsr()

    X_test_hybrid = sp.hstack([
        X_test_tfidf,
        sp.csr_matrix(X_test_bert * bert_weight)
    ]).tocsr()

    sp.save_npz(os.path.join(outdir, "X_train_hybrid.npz"), X_train_hybrid)
    sp.save_npz(os.path.join(outdir, "X_test_hybrid.npz"), X_test_hybrid)
    pd.Series(y_train).to_csv(os.path.join(outdir, "y_train.csv"), index=False)
    pd.Series(y_test).to_csv(os.path.join(outdir, "y_test.csv"), index=False)

    joblib.dump(word_tfidf, os.path.join(outdir, "word_tfidf.joblib"))
    joblib.dump(char_tfidf, os.path.join(outdir, "char_tfidf.joblib"))
    joblib.dump(scaler_bert, os.path.join(outdir, "scaler_bert.joblib"))

    meta = {
        "base_model_name": base_model_name,
        "model_dir": model_dir,
        "text_col": text_col,
        "label_col": label_col,
        "max_length": max_length,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "pooling": pooling,
        "chunk_aggregation": chunk_aggregation,
        "bert_batch_size": batch_size,
        "bert_weight": bert_weight,
        "train_shape": X_train_hybrid.shape,
        "test_shape": X_test_hybrid.shape,
    }
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Hybrid vectors built.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    return {
        "X_train_shape": tuple(X_train_hybrid.shape),
        "X_test_shape": tuple(X_test_hybrid.shape),
        "meta": meta,
    }


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bert_weight", type=float, default=BERT_WEIGHT)

    parser.add_argument("--base_model_name", default=MODEL_NAME)
    parser.add_argument("--text_col", default=TEXT_COL)
    parser.add_argument("--label_col", default=LABEL_COL)

    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--pooling", choices=["mean", "cls", "max", "mean_max"], default="mean_max")
    parser.add_argument("--chunk_aggregation", choices=["mean", "max", "mean_max"], default="mean_max")
    parser.add_argument("--batch_size", type=int, default=BERT_BATCH_SIZE)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    run_build(
        train_file=args.train,
        test_file=args.test,
        outdir=args.outdir,
        model_dir=args.model_dir,
        device=args.device,
        bert_weight=args.bert_weight,
        base_model_name=args.base_model_name,
        text_col=args.text_col,
        label_col=args.label_col,
        max_length=args.max_length,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        pooling=args.pooling,
        chunk_aggregation=args.chunk_aggregation,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()