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
from .config import HybridModelConfig, HybridDataConfig


def load_and_clean_df(path, text_col: str = "text", label_col: str = "label", require_labels: bool = True):
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

    # Нормируем TF‑IDF по строкам
    X_train_tfidf = normalize(X_train_tfidf, norm="l2")
    X_test_tfidf = normalize(X_test_tfidf, norm="l2")

    return X_train_tfidf, X_test_tfidf, word_tfidf, char_tfidf


def build_embedder(model_dir: str = "", model_cfg: HybridModelConfig | None = None) -> LongTextRobertaEmbedder:
    if model_cfg is None:
        model_cfg = HybridModelConfig()

    return LongTextRobertaEmbedder(
        model_dir=model_dir,
        base_model_name=model_cfg.base_model_name,
        max_length=model_cfg.max_length,
        chunk_size=model_cfg.chunk_size,
        chunk_overlap=model_cfg.chunk_overlap,
        pooling=model_cfg.pooling,
        chunk_aggregation=model_cfg.chunk_aggregation,
        batch_size=model_cfg.batch_size,
    )


def document_bert_embeddings_from_model_dir(
    texts,
    model_dir: str,
    model_cfg: HybridModelConfig | None = None,
) -> np.ndarray:
    embedder = build_embedder(model_dir=model_dir, model_cfg=model_cfg)
    embs = embedder.encode([str(t) for t in texts])
    return embs.astype(np.float32)


@torch.no_grad()
def document_bert_embeddings_base(
    texts,
    tokenizer,
    model,
    device,
    batch_size: int = 8,
    max_length: int = 512,
) -> np.ndarray:
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
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded, return_dict=True)

        last_hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).type_as(last_hidden)
        masked = last_hidden * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts

        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_vecs.append(pooled.cpu().numpy().astype(np.float32))

    return np.vstack(all_vecs)


def run_build(
    train_file: str,
    test_file: str,
    outdir: str,
    model_dir: str | None = None,
    device: str = "cpu",
    model_cfg: HybridModelConfig | None = None,
    data_cfg: HybridDataConfig | None = None,
):
    if model_cfg is None:
        model_cfg = HybridModelConfig()
    if data_cfg is None:
        data_cfg = HybridDataConfig()

    os.makedirs(outdir, exist_ok=True)

    train_df = load_and_clean_df(
        train_file,
        text_col=data_cfg.text_col,
        label_col=data_cfg.label_col,
        require_labels=True,
    )
    test_df = load_and_clean_df(
        test_file,
        text_col=data_cfg.text_col,
        label_col=data_cfg.label_col,
        require_labels=True,
    )

    X_train = train_df[data_cfg.text_col]
    y_train = train_df[data_cfg.label_col]
    X_test = test_df[data_cfg.text_col]
    y_test = test_df[data_cfg.label_col]

    X_train_tfidf, X_test_tfidf, word_tfidf, char_tfidf = build_tfidf(X_train, X_test)

    if model_dir:
        X_train_bert = document_bert_embeddings_from_model_dir(
            X_train.tolist(),
            model_dir=model_dir,
            model_cfg=model_cfg,
        )
        X_test_bert = document_bert_embeddings_from_model_dir(
            X_test.tolist(),
            model_dir=model_dir,
            model_cfg=model_cfg,
        )
    else:
        torch_device = torch.device(
            device if torch.cuda.is_available() and device != "cpu" else "cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.base_model_name)
        base_model = RobertaModel.from_pretrained(model_cfg.base_model_name).to(torch_device)

        X_train_bert = document_bert_embeddings_base(
            X_train.tolist(),
            tokenizer,
            base_model,
            torch_device,
            batch_size=model_cfg.batch_size,
            max_length=model_cfg.max_length,
        )
        X_test_bert = document_bert_embeddings_base(
            X_test.tolist(),
            tokenizer,
            base_model,
            torch_device,
            batch_size=model_cfg.batch_size,
            max_length=model_cfg.max_length,
        )

    # StandardScaler по BERT-блоку
    scaler_bert = StandardScaler()
    X_train_bert_scaled = scaler_bert.fit_transform(X_train_bert).astype(np.float32)
    X_test_bert_scaled = scaler_bert.transform(X_test_bert).astype(np.float32)

    # Важно: BERT-блок масштабируется через bert_weight
    X_train_bert_weighted = X_train_bert_scaled * float(model_cfg.bert_weight)
    X_test_bert_weighted = X_test_bert_scaled * float(model_cfg.bert_weight)

    X_train_hybrid = sp.hstack(
        [
            X_train_tfidf,
            sp.csr_matrix(X_train_bert_weighted),
        ]
    ).tocsr()

    X_test_hybrid = sp.hstack(
        [
            X_test_tfidf,
            sp.csr_matrix(X_test_bert_weighted),
        ]
    ).tocsr()

    # Финальная L2‑нормировка гибридных векторов
    X_train_hybrid = normalize(X_train_hybrid, norm="l2")
    X_test_hybrid = normalize(X_test_hybrid, norm="l2")

    sp.save_npz(os.path.join(outdir, "X_train_hybrid.npz"), X_train_hybrid)
    sp.save_npz(os.path.join(outdir, "X_test_hybrid.npz"), X_test_hybrid)
    pd.Series(y_train).to_csv(os.path.join(outdir, "y_train.csv"), index=False)
    pd.Series(y_test).to_csv(os.path.join(outdir, "y_test.csv"), index=False)

    joblib.dump(word_tfidf, os.path.join(outdir, "word_tfidf.joblib"))
    joblib.dump(char_tfidf, os.path.join(outdir, "char_tfidf.joblib"))
    joblib.dump(scaler_bert, os.path.join(outdir, "scaler_bert.joblib"))

    # Немного диагностик по нормам (можно смотреть глазами)
    train_tfidf_norm_mean = float(
        np.mean(np.linalg.norm(X_train_tfidf.toarray(), axis=1))
    ) if X_train_tfidf.shape[0] > 0 else 0.0
    train_bert_norm_mean = float(
        np.mean(np.linalg.norm(X_train_bert_scaled, axis=1))
    ) if X_train_bert_scaled.shape[0] > 0 else 0.0
    train_bert_weighted_norm_mean = float(
        np.mean(np.linalg.norm(X_train_bert_weighted, axis=1))
    ) if X_train_bert_weighted.shape[0] > 0 else 0.0

    # Доля BERT-энергии в гибридных векторах (на train, до финальной нормировки).
    # Это и есть «эффективный bert_weight» — единственная величина, которая
    # реально влияет на cos-similarity после нормировки.
    if X_train_tfidf.shape[0] > 0:
        tfidf_sq = np.asarray(X_train_tfidf.multiply(X_train_tfidf).sum(axis=1)).ravel()
        bert_sq = np.linalg.norm(X_train_bert_weighted, axis=1) ** 2
        denom = tfidf_sq + bert_sq + 1e-12
        bert_share_mean = float(np.mean(bert_sq / denom))
    else:
        bert_share_mean = 0.0

    meta = {
        "base_model_name": model_cfg.base_model_name,
        "model_dir": model_dir,
        "text_col": data_cfg.text_col,
        "label_col": data_cfg.label_col,
        "max_length": model_cfg.max_length,
        "chunk_size": model_cfg.chunk_size,
        "chunk_overlap": model_cfg.chunk_overlap,
        "pooling": model_cfg.pooling,
        "chunk_aggregation": model_cfg.chunk_aggregation,
        "bert_batch_size": model_cfg.batch_size,
        "bert_weight": float(model_cfg.bert_weight),
        "train_shape": X_train_hybrid.shape,
        "test_shape": X_test_hybrid.shape,
        "train_tfidf_norm_mean": train_tfidf_norm_mean,
        "train_bert_norm_mean": train_bert_norm_mean,
        "train_bert_weighted_norm_mean": train_bert_weighted_norm_mean,
        "bert_share_mean": bert_share_mean,
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