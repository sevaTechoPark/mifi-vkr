from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


DOC_ID_COL = "_doc_id"


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
    """
    Каждому исходному документу присваиваем уникальный _doc_id, чтобы дальше
    в build_pair_dataframe можно было строить ТОЛЬКО кросс-документные позитивы.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    rows = []
    for doc_id, (_, row) in enumerate(df.iterrows()):
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
            rows.append({text_col: v, label_col: label, DOC_ID_COL: int(doc_id)})

    out = pd.DataFrame(rows)
    # dedup по тексту, но СОХРАНЯЕМ _doc_id первой встречи
    out = out.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    return out


def _sample_cross_doc_positive_pairs(
    items: list[tuple[str, int]],
    max_pairs: int,
    rng: random.Random,
):
    """items: list of (text, doc_id). Возвращает только пары, где doc_id различаются."""
    if len(items) < 2:
        return []
    cand = [
        (i, j)
        for i, j in combinations(range(len(items)), 2)
        if items[i][1] != items[j][1]
    ]
    if not cand:
        return []
    if len(cand) > max_pairs:
        cand = rng.sample(cand, max_pairs)
    return [(items[i][0], items[j][0], 1.0, 1) for i, j in cand]


def _sample_negative_pairs(
    grouped_items: dict[str, list[tuple[str, int]]],
    max_pairs: int,
    rng: random.Random,
):
    labels = list(grouped_items.keys())
    if len(labels) < 2 or max_pairs <= 0:
        return []

    neg_pairs = []
    attempts = 0
    max_attempts = max_pairs * 30

    while len(neg_pairs) < max_pairs and attempts < max_attempts:
        attempts += 1
        label_a, label_b = rng.sample(labels, 2)
        items_a = grouped_items[label_a]
        items_b = grouped_items[label_b]
        if not items_a or not items_b:
            continue
        a = rng.choice(items_a)[0]
        b = rng.choice(items_b)[0]
        neg_pairs.append((a, b, 0.0, 0))

    return neg_pairs


def build_pair_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    max_pairs_per_label: int = 5000,
    max_negative_pairs: int = 5000,
    seed: int = 42,
    balance_positives: bool = True,
    cross_document_positives_only: bool = True,
) -> pd.DataFrame:
    """
    Главное отличие от старой версии: позитивы строятся ТОЛЬКО между чанками
    РАЗНЫХ документов одного класса. Это убирает тривиальные positive-пары
    из соседних overlapping-чанков одного письма, которые ломали MNR loss
    (representation collapse).
    """
    rng = random.Random(seed)

    has_doc_id = DOC_ID_COL in df.columns

    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for _, row in df.iterrows():
        doc_id = int(row[DOC_ID_COL]) if has_doc_id else -1
        grouped[row[label_col]].append((row[text_col], doc_id))

    if balance_positives:
        per_class_caps = []
        for items in grouped.values():
            n = len(items)
            possible = n * (n - 1) // 2
            per_class_caps.append(min(max_pairs_per_label, possible))
        if per_class_caps:
            median_cap = sorted(per_class_caps)[len(per_class_caps) // 2]
            effective_cap = max(1, median_cap)
        else:
            effective_cap = max_pairs_per_label
    else:
        effective_cap = max_pairs_per_label

    pairs = []
    use_cross_doc = cross_document_positives_only and has_doc_id

    for _, items in grouped.items():
        if use_cross_doc:
            pairs.extend(
                _sample_cross_doc_positive_pairs(items, max_pairs=effective_cap, rng=rng)
            )
        else:
            if len(items) < 2:
                continue
            cand = list(combinations(range(len(items)), 2))
            if len(cand) > effective_cap:
                cand = rng.sample(cand, effective_cap)
            for i, j in cand:
                pairs.append((items[i][0], items[j][0], 1.0, 1))

    pairs.extend(_sample_negative_pairs(grouped, max_pairs=max_negative_pairs, rng=rng))

    pair_df = pd.DataFrame(pairs, columns=["sentence1", "sentence2", "score", "label"])
    pair_df = pair_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return pair_df