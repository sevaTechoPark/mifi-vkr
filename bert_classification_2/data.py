"""Загрузка, очистка и токенизация датасета для классификации.

Pipeline:
  CSV (train/test) → DataFrame → label2id → label_id → Dataset → токенизация
  с чанкованием → DatasetDict.

Каждый документ превращается в фиксированное число чанков (model_cfg.max_chunks)
по model_cfg.max_length токенов. Недостающие чанки добиваются паддингом,
число валидных чанков сохраняется в поле num_chunks — оно используется
моделью при усреднении эмбеддингов чанков.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, disable_caching
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer

from .config import ModelConfig, DataConfig

# Отключаем кэширование `datasets`: пайплайн быстрый, а файлы кэша занимают место.
disable_caching()


def load_and_prepare_dataframes(
    train_file: str,
    test_file: str,
    text_col: str,
    label_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Читает train/test CSV, оставляет только нужные колонки, чистит пустые строки."""
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    train_df = train_df[[text_col, label_col]].copy().dropna()
    test_df = test_df[[text_col, label_col]].copy().dropna()

    # Приводим к строкам и убираем пробелы по краям.
    for df in (train_df, test_df):
        df[text_col] = df[text_col].astype(str).str.strip()
        df[label_col] = df[label_col].astype(str).str.strip()

    train_df = train_df[
        (train_df[text_col] != "") & (train_df[label_col] != "")
    ].reset_index(drop=True)
    test_df = test_df[
        (test_df[text_col] != "") & (test_df[label_col] != "")
    ].reset_index(drop=True)

    return train_df, test_df


def build_label_mappings(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
):
    """Строит label2id / id2label по обучающему набору меток.

    Падает с ошибкой, если в тесте встретится метка, не виденная в train —
    модель такую метку всё равно не сможет предсказать.
    """
    labels = sorted(train_df[label_col].unique().tolist())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    unknown_test_labels = sorted(set(test_df[label_col].unique()) - set(labels))
    if unknown_test_labels:
        raise ValueError(f"В TEST_FILE есть unseen labels: {unknown_test_labels}")

    return labels, label2id, id2label


def attach_label_ids(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    label2id: Dict[str, int],
):
    """Добавляет колонку `label_id` (int) на основе label2id."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["label_id"] = train_df[label_col].map(label2id)
    test_df["label_id"] = test_df[label_col].map(label2id)

    return train_df, test_df


def compute_class_weights_tensor(
    train_label_ids: np.ndarray,
    num_labels: int,
) -> torch.Tensor:
    """Считает балансировочные веса классов (sklearn `balanced`).

    Передаются в CrossEntropyLoss как `weight=...` — это компенсирует дисбаланс
    в обучающей выборке без необходимости физического oversampling.
    """
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_labels),
        y=train_label_ids,
    )
    return torch.tensor(class_weights, dtype=torch.float)


def build_tokenizer(model_cfg: ModelConfig):
    return AutoTokenizer.from_pretrained(model_cfg.model_name)


def build_tokenize_document_fn(tokenizer, model_cfg: ModelConfig, data_cfg: DataConfig):
    """Возвращает функцию-токенизатор, готовую к использованию в Dataset.map().

    Документ режется на перекрывающиеся окна (sliding window с overlap = stride).
    Число окон ограничивается max_chunks; недостающие позиции добиваются паддингом.
    """

    def tokenize_document(example):
        encoded = tokenizer(
            example[data_cfg.text_col],
            truncation=True,
            padding="max_length",
            max_length=model_cfg.max_length,
            stride=model_cfg.stride,
            return_overflowing_tokens=True,
        )

        input_ids_chunks = encoded["input_ids"][:model_cfg.max_chunks]
        attention_mask_chunks = encoded["attention_mask"][:model_cfg.max_chunks]
        n_chunks = len(input_ids_chunks)

        # Добиваем паддингом до фиксированного числа чанков, чтобы все семплы
        # в батче имели одинаковую форму (упрощает data collator).
        if n_chunks < model_cfg.max_chunks:
            pad_len = model_cfg.max_chunks - n_chunks
            pad_ids = [tokenizer.pad_token_id] * model_cfg.max_length
            pad_mask = [0] * model_cfg.max_length
            input_ids_chunks += [pad_ids] * pad_len
            attention_mask_chunks += [pad_mask] * pad_len

        return {
            "input_ids": input_ids_chunks,
            "attention_mask": attention_mask_chunks,
            "labels": example["label_id"],
            "num_chunks": n_chunks,
        }

    return tokenize_document


def build_dataset_dict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
) -> DatasetDict:
    """Превращает DataFrame'ы в HuggingFace DatasetDict с уже токенизированными чанками."""
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(
                train_df[[data_cfg.text_col, "label_id"]],
                preserve_index=False,
            ),
            "validation": Dataset.from_pandas(
                test_df[[data_cfg.text_col, "label_id"]],
                preserve_index=False,
            ),
        }
    )

    tokenize_document = build_tokenize_document_fn(tokenizer, model_cfg, data_cfg)

    dataset = dataset.map(
        tokenize_document,
        load_from_cache_file=False,
    )
    dataset.set_format(type="python")
    return dataset
