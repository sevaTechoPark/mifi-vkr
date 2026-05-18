from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def load_labeled_text_csv(
    path: str,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].reset_index(drop=True)
    df[label_col] = df[label_col].astype(str)
    return df


def build_training_dataframe(
    train_path: str,
    test_path: str | None = None,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Готовит train-датасет для обучения энкодера БЕЗ test-данных.
    test_path принимается ради обратной совместимости и игнорируется при формировании df.

    Если test_path передан — печатается предупреждение, и тестовая выборка НЕ подмешивается
    в обучающую (иначе энкодер «увидит» тест и downstream-классификаторы получат утечку).
    """
    if test_path is not None:
        print(
            "[bert_embeddings] build_training_dataframe: test_path передан, "
            "но НЕ используется для обучения энкодера (защита от утечки train↔test)."
        )
    return load_labeled_text_csv(train_path, text_col=text_col, label_col=label_col)


def save_texts_parquet(df: pd.DataFrame, out_path: str):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def split_text_to_training_views(
    text: str,
    tokenizer,
    max_length: int = 512,
    chunk_size: int = 448,
    chunk_overlap: int = 128,
    add_global_chunk: bool = True,
) -> list[str]:
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]

    if not token_ids:
        return []

    chunk_size = min(chunk_size, max_length - 2)
    chunk_overlap = min(chunk_overlap, max(0, chunk_size // 2))
    stride = max(1, chunk_size - chunk_overlap)

    views = []
    for start in range(0, len(token_ids), stride):
        end = start + chunk_size
        piece = token_ids[start:end]
        if not piece:
            continue
        text_piece = tokenizer.decode(piece, skip_special_tokens=True).strip()
        if text_piece:
            views.append(text_piece)
        if end >= len(token_ids):
            break

    if add_global_chunk and len(token_ids) > chunk_size:
        head = token_ids[: chunk_size // 2]
        tail = token_ids[-(chunk_size - len(head)) :]
        global_piece = (head + tail)[:chunk_size]
        text_piece = tokenizer.decode(global_piece, skip_special_tokens=True).strip()
        if text_piece:
            views.append(text_piece)

    seen = set()
    deduped = []
    for v in views:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def explode_long_texts_for_training(
    df: pd.DataFrame,
    model_name: str,
    text_col: str = "text",
    label_col: str = "label",
    max_length: int = 512,
    chunk_size: int = 448,
    chunk_overlap: int = 128,
    add_global_chunk: bool = True,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    rows = []
    for _, row in df.iterrows():
        text = row[text_col]
        label = row[label_col]
        views = split_text_to_training_views(
            text=text,
            tokenizer=tokenizer,
            max_length=max_length,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_global_chunk=add_global_chunk,
        )
        if not views:
            continue
        for v in views:
            rows.append({text_col: v, label_col: label})

    out = pd.DataFrame(rows)
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def _sample_positive_pairs(texts: list[str], max_pairs: int, rng: random.Random):
    if len(texts) < 2:
        return []

    all_pairs = list(combinations(range(len(texts)), 2))
    if len(all_pairs) > max_pairs:
        all_pairs = rng.sample(all_pairs, max_pairs)

    return [(texts[i], texts[j], 1.0, 1) for i, j in all_pairs]


def _sample_negative_pairs(grouped_texts: dict[str, list[str]], max_pairs: int, rng: random.Random):
    labels = list(grouped_texts.keys())
    if len(labels) < 2 or max_pairs <= 0:
        return []

    neg_pairs = []
    attempts = 0
    max_attempts = max_pairs * 30

    while len(neg_pairs) < max_pairs and attempts < max_attempts:
        attempts += 1
        label_a, label_b = rng.sample(labels, 2)
        texts_a = grouped_texts[label_a]
        texts_b = grouped_texts[label_b]
        if not texts_a or not texts_b:
            continue
        a = rng.choice(texts_a)
        b = rng.choice(texts_b)
        neg_pairs.append((a, b, 0.0, 0))

    return neg_pairs


def build_pair_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    max_pairs_per_label: int = 20000,
    max_negative_pairs: int = 20000,
    seed: int = 42,
    balance_positives: bool = True,
) -> pd.DataFrame:
    rng = random.Random(seed)

    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[row[label_col]].append(row[text_col])

    pairs = []

    if balance_positives:
        # лимит на класс = min(глобальный лимит, медиана числа возможных пар по классам).
        # Это снимает перекос в сторону крупных классов: миноры не «тонут».
        per_class_caps = []
        for texts in grouped.values():
            n = len(texts)
            possible = n * (n - 1) // 2
            per_class_caps.append(min(max_pairs_per_label, possible))
        if per_class_caps:
            median_cap = sorted(per_class_caps)[len(per_class_caps) // 2]
            effective_cap = max(1, median_cap)
        else:
            effective_cap = max_pairs_per_label
    else:
        effective_cap = max_pairs_per_label

    for _, texts in grouped.items():
        pairs.extend(_sample_positive_pairs(texts, max_pairs=effective_cap, rng=rng))
    pairs.extend(_sample_negative_pairs(grouped, max_pairs=max_negative_pairs, rng=rng))

    pair_df = pd.DataFrame(pairs, columns=["sentence1", "sentence2", "score", "label"])
    pair_df = pair_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return pair_df