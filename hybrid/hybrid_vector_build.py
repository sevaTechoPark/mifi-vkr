"""
Построение гибридного признакового пространства (TF-IDF + BERT).

Из обучающего и тестового CSV формируются:

  * TF-IDF (word n-grams + char_wb n-grams), L2-нормированный;
  * BERT-эмбеддинги документов (через LongTextRobertaEmbedder при
    наличии model_dir, либо через базовую модель ruRoberta при mean-pool);
  * Гибридный вектор как hstack TF-IDF и взвешенного BERT-блока.

На диск сохраняются две версии гибридного вектора и отдельно BERT-блок,
а также артефакты векторизаторов и StandardScaler.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from transformers import AutoTokenizer, RobertaModel

from bert_embeddings.embedding_model import LongTextRobertaEmbedder
from .config import HybridModelConfig, HybridDataConfig


def load_and_clean_df(path, text_col="text", label_col="label", require_labels=True):
    """Прочитать CSV, удалить пустые тексты и (опционально) пустые метки."""
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


def build_tfidf(X_train, X_test, model_cfg: HybridModelConfig):
    """
    Построить TF-IDF из двух подвекторизаторов: word n-grams и char_wb n-grams.

    Возвращает L2-нормированные разреженные матрицы для train/test и оба
    обученных векторизатора.
    """
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(model_cfg.word_ngram_min, model_cfg.word_ngram_max),
        token_pattern=r"(?u)\b\w\w+\b",
        lowercase=True,
        min_df=model_cfg.word_min_df,
        max_df=model_cfg.word_max_df,
        sublinear_tf=True,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(model_cfg.char_ngram_min, model_cfg.char_ngram_max),
        min_df=model_cfg.char_min_df,
        max_df=model_cfg.char_max_df,
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


def build_embedder(model_dir="", model_cfg=None):
    """Сконструировать LongTextRobertaEmbedder по параметрам конфига."""
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


def document_bert_embeddings_from_model_dir(texts, model_dir, model_cfg=None):
    """Эмбеддинги документов через дообученную модель из model_dir."""
    embedder = build_embedder(model_dir=model_dir, model_cfg=model_cfg)
    embs = embedder.encode([str(t) for t in texts])
    return embs.astype(np.float32)


@torch.no_grad()
def document_bert_embeddings_base(
    texts, tokenizer, model, device, batch_size=8, max_length=512,
):
    """
    Эмбеддинги документов через базовую ruRoberta-модель.

    Используется mean-pooling по токенам с учётом attention_mask с
    последующей L2-нормировкой.
    """
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
    train_file,
    test_file,
    outdir,
    model_dir=None,
    device="cpu",
    model_cfg=None,
    data_cfg=None,
):
    """
    Сформировать и сохранить гибридное признаковое пространство.

    Сохраняемые артефакты:

      * X_train_bert.npy / X_test_bert.npy — L2-нормированные эмбеддинги;
      * X_train_hybrid.npz / X_test_hybrid.npz — гибридный вектор с
        финальной L2-нормировкой (используется MLP);
      * X_train_hybrid_noL2.npz / X_test_hybrid_noL2.npz — гибридный
        вектор без финальной L2 (используется classical-моделями);
      * X_train_dense.npy / X_test_dense.npy — опциональное dense-представление
        после TruncatedSVD;
      * texts_*.csv, y_*.csv, *.joblib, meta.json — служебные файлы.
    """
    if model_cfg is None:
        model_cfg = HybridModelConfig()
    if data_cfg is None:
        data_cfg = HybridDataConfig()

    os.makedirs(outdir, exist_ok=True)

    train_df = load_and_clean_df(
        train_file, text_col=data_cfg.text_col, label_col=data_cfg.label_col,
    )
    test_df = load_and_clean_df(
        test_file, text_col=data_cfg.text_col, label_col=data_cfg.label_col,
    )

    X_train = train_df[data_cfg.text_col]
    y_train = train_df[data_cfg.label_col]
    X_test = test_df[data_cfg.text_col]
    y_test = test_df[data_cfg.label_col]

    # Тексты сохраняются отдельно: они нужны TF-IDF-only baseline в
    # модуле classical-моделей.
    train_df[[data_cfg.text_col, data_cfg.label_col]].to_csv(
        os.path.join(outdir, "texts_train.csv"), index=False,
    )
    test_df[[data_cfg.text_col, data_cfg.label_col]].to_csv(
        os.path.join(outdir, "texts_test.csv"), index=False,
    )

    X_train_tfidf, X_test_tfidf, word_tfidf, char_tfidf = build_tfidf(
        X_train, X_test, model_cfg,
    )

    if model_dir:
        X_train_bert = document_bert_embeddings_from_model_dir(
            X_train.tolist(), model_dir=model_dir, model_cfg=model_cfg,
        )
        X_test_bert = document_bert_embeddings_from_model_dir(
            X_test.tolist(), model_dir=model_dir, model_cfg=model_cfg,
        )
    else:
        torch_device = torch.device(
            device if torch.cuda.is_available() and device != "cpu" else "cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.base_model_name)
        base_model = RobertaModel.from_pretrained(model_cfg.base_model_name).to(torch_device)

        X_train_bert = document_bert_embeddings_base(
            X_train.tolist(), tokenizer, base_model, torch_device,
            batch_size=model_cfg.batch_size, max_length=model_cfg.max_length,
        )
        X_test_bert = document_bert_embeddings_base(
            X_test.tolist(), tokenizer, base_model, torch_device,
            batch_size=model_cfg.batch_size, max_length=model_cfg.max_length,
        )

    # StandardScaler по BERT-блоку нужен для базовых mean-pool эмбеддингов.
    # Для дообученного эмбеддера (уже L2-нормирован) стандартизация отключается.
    if getattr(model_cfg, "disable_bert_scaler", False):
        print("[hybrid.build] BERT StandardScaler: DISABLED")
        X_train_bert_scaled = X_train_bert.astype(np.float32)
        X_test_bert_scaled = X_test_bert.astype(np.float32)
        scaler_bert = None
    else:
        print("[hybrid.build] BERT StandardScaler: ENABLED")
        scaler_bert = StandardScaler()
        X_train_bert_scaled = scaler_bert.fit_transform(X_train_bert).astype(np.float32)
        X_test_bert_scaled = scaler_bert.transform(X_test_bert).astype(np.float32)

    # L2-нормировка BERT-блока до конкатенации с TF-IDF. Это делает
    # эффективный вклад BERT в гибридный вектор независимым от размерности
    # TF-IDF-части.
    X_train_bert_l2 = normalize(X_train_bert_scaled, norm="l2", axis=1).astype(np.float32)
    X_test_bert_l2 = normalize(X_test_bert_scaled, norm="l2", axis=1).astype(np.float32)

    # Отдельно сохраняем BERT-блок: classical-модели на нём дают сильный
    # single-source baseline.
    np.save(os.path.join(outdir, "X_train_bert.npy"), X_train_bert_l2)
    np.save(os.path.join(outdir, "X_test_bert.npy"), X_test_bert_l2)

    # Гибридный вектор: TF-IDF (уже L2) + BERT (L2) * bert_weight.
    X_train_bert_weighted = X_train_bert_l2 * float(model_cfg.bert_weight)
    X_test_bert_weighted = X_test_bert_l2 * float(model_cfg.bert_weight)

    X_train_hybrid = sp.hstack(
        [X_train_tfidf, sp.csr_matrix(X_train_bert_weighted)],
    ).tocsr()
    X_test_hybrid = sp.hstack(
        [X_test_tfidf, sp.csr_matrix(X_test_bert_weighted)],
    ).tocsr()

    # Версия с финальной L2-нормировкой — для MLP, где норма входа
    # участвует в обучении.
    X_train_hybrid_l2 = normalize(X_train_hybrid, norm="l2")
    X_test_hybrid_l2 = normalize(X_test_hybrid, norm="l2")

    # Опциональный TruncatedSVD для плотного представления.
    svd = None
    if model_cfg.svd_components and model_cfg.svd_components > 0:
        n_comp = min(
            model_cfg.svd_components,
            X_train_hybrid_l2.shape[1] - 1,
            X_train_hybrid_l2.shape[0] - 1,
        )
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X_train_dense = svd.fit_transform(X_train_hybrid_l2)
        X_test_dense = svd.transform(X_test_hybrid_l2)
        np.save(
            os.path.join(outdir, "X_train_dense.npy"),
            X_train_dense.astype(np.float32),
        )
        np.save(
            os.path.join(outdir, "X_test_dense.npy"),
            X_test_dense.astype(np.float32),
        )
        joblib.dump(svd, os.path.join(outdir, "svd.joblib"))

    # Гибридный вектор сохраняется в двух вариантах:
    #   X_train_hybrid.npz       — с финальной L2 (для MLP);
    #   X_train_hybrid_noL2.npz  — с поблочной L2, но без финальной (для
    #                              линейных моделей: им вреднa финальная L2).
    sp.save_npz(os.path.join(outdir, "X_train_hybrid.npz"), X_train_hybrid_l2)
    sp.save_npz(os.path.join(outdir, "X_test_hybrid.npz"), X_test_hybrid_l2)
    sp.save_npz(os.path.join(outdir, "X_train_hybrid_noL2.npz"), X_train_hybrid)
    sp.save_npz(os.path.join(outdir, "X_test_hybrid_noL2.npz"), X_test_hybrid)

    pd.Series(y_train).to_csv(os.path.join(outdir, "y_train.csv"), index=False)
    pd.Series(y_test).to_csv(os.path.join(outdir, "y_test.csv"), index=False)

    joblib.dump(word_tfidf, os.path.join(outdir, "word_tfidf.joblib"))
    joblib.dump(char_tfidf, os.path.join(outdir, "char_tfidf.joblib"))
    joblib.dump(scaler_bert, os.path.join(outdir, "scaler_bert.joblib"))

    # Доля энергии BERT-блока в гибридном векторе (для диагностики).
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
        "bert_scaler_used": scaler_bert is not None,
        "bert_weight_with_model_dir": float(
            getattr(model_cfg, "bert_weight_with_model_dir", 5.0)
        ),
        "bert_weight_base_model": float(
            getattr(model_cfg, "bert_weight_base_model", 1.0)
        ),
        "svd_components": int(model_cfg.svd_components),
        "train_shape": list(X_train_hybrid.shape),
        "test_shape": list(X_test_hybrid.shape),
        "tfidf_dim": int(X_train_tfidf.shape[1]),
        "bert_dim": int(X_train_bert_scaled.shape[1]),
        "bert_share_mean": bert_share_mean,
        "bert_l2_per_block": True,
        "hybrid_final_l2": True,
        "hybrid_noL2_file": "X_train_hybrid_noL2.npz",
        "bert_only_file": "X_train_bert.npy",
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
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser
