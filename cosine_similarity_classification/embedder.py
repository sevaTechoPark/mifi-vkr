import numpy as np
import pandas as pd

from bert_embeddings.embedding_model import LongTextRobertaEmbedder


def load_texts_and_labels(path, text_col="text", label_col="label", require_labels=True):
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
        df[label_col] = df[label_col].astype(str)

    return df


def build_embedder(
    model_dir="",
    base_model_name="ai-forever/ruRoberta-large",
    max_length=512,
    chunk_size=448,
    chunk_overlap=96,
    pooling="mean_max",
    chunk_aggregation="mean_max",
    batch_size=8,
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


def embed_dataframe(df, embedder, text_col="text"):
    texts = df[text_col].tolist()
    embs = embedder.encode(texts)
    return embs


def save_embeddings(path, embs):
    np.save(path, embs)