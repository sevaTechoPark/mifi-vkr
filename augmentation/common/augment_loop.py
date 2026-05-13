"""
Общий цикл аугментации по малым классам.
Принимает callable augment_fn(text: str) -> str.
"""
import copy
import os
import pandas as pd
import torch
from tqdm.auto import tqdm
from sentence_transformers import util, SentenceTransformer

from .config import TARGET_PER_CLASS, SIM_LABEL_MIN, SIM_LABEL_MAX
from .embeddings import cos_sim


def normalize_text(text: str) -> str:
    import re
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def run_augmentation_loop(
    df: pd.DataFrame,
    embed_model: SentenceTransformer,
    augment_fn,           # callable(text: str) -> str
    aug_file_path: str,
    augmentation_type: str,
    sim_min: float,
    sim_max: float,
    min_len_ratio: float,
    max_len_ratio: float,
) -> pd.DataFrame:
    counts = df["label"].value_counts()
    small_labels = counts[counts < TARGET_PER_CLASS].sort_values(ascending=True).index
    df_small = df[df["label"].isin(small_labels)]

    TOTAL_AUGMENTED = (TARGET_PER_CLASS * len(small_labels)) - len(df_small)

    print(f"Всего малых классов: {len(small_labels)}")
    print(f"Текущее количество примеров в малых классах: {len(df_small)}")
    print(f"Целевое количество на класс: {TARGET_PER_CLASS}")
    print(f"Нужно сгенерировать аугментаций: {TOTAL_AUGMENTED}")

    aug_rows = []
    if os.path.exists(aug_file_path):
        df_prev = pd.read_csv(aug_file_path)
        aug_rows = df_prev.to_dict(orient="records")
        print(f"Загружено уже сгенерированных примеров: {len(aug_rows)}")
    else:
        print("Генерируем с нуля")

    label_to_texts = {
        label: df_small.loc[df_small["label"] == label, "text"].tolist()
        for label in small_labels
    }
    label_to_embs = {}

    for label in tqdm(small_labels, desc="Labels"):
        texts_orig = label_to_texts[label]
        orig_count = len(texts_orig)

        label_aug_rows = [r for r in aug_rows if r["label"] == label]
        current_count = orig_count + len(label_aug_rows)
        need = TARGET_PER_CLASS - current_count

        print(f"\nLabel: {label} | есть {current_count}, нужно добить: {need}")

        if need <= 0:
            print(f"Лейбл {label} уже заполнен")
            continue

        if label not in label_to_embs:
            label_to_embs[label] = embed_model.encode(
                copy.deepcopy(texts_orig),
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

        label_embs = label_to_embs[label]
        orig_idx = 0
        attempts = 0
        max_attempts = need * 50

        while need > 0 and attempts < max_attempts:
            attempts += 1
            source_text = texts_orig[orig_idx]

            aug_text = augment_fn(source_text)

            # фильтр по длине
            len_src = len(source_text)
            len_aug = len(aug_text)
            len_ratio = len_aug / len_src

            if not (min_len_ratio <= len_ratio <= max_len_ratio):
                print(
                    f"[SKIP] длина aug/source = {len_ratio:.2f} "
                    f"(ожидалось [{min_len_ratio}, {max_len_ratio}])"
                )
                orig_idx = (orig_idx + 1) % orig_count
                continue

            if normalize_text(aug_text) == normalize_text(source_text):
                print(f"[SKIP] идентичен оригиналу")
                orig_idx = (orig_idx + 1) % orig_count
                continue

            original_cosine_sim = cos_sim(embed_model, source_text, aug_text)
            if not (sim_min <= original_cosine_sim <= sim_max):
                print(f"[SKIP] sim с source={original_cosine_sim:.4f}")
                orig_idx = (orig_idx + 1) % orig_count
                continue

            new_emb = embed_model.encode(
                aug_text, convert_to_tensor=True, normalize_embeddings=True
            )
            sims = util.cos_sim(new_emb, label_embs)[0]
            max_label_sim = float(torch.max(sims))

            if not (SIM_LABEL_MIN <= max_label_sim <= SIM_LABEL_MAX):
                print(f"[SKIP] sim с лейблом={max_label_sim:.4f}")
                orig_idx = (orig_idx + 1) % orig_count
                continue

            label_embs = torch.cat([label_embs, new_emb.unsqueeze(0)], dim=0)
            label_to_embs[label] = label_embs

            aug_rows.append({
                "label": label,
                "text": aug_text,
                "source_text": source_text,
                "cosine_sim": original_cosine_sim,
                "max_label_cosine_sim": max_label_sim,
                "augmentation_type": augmentation_type,
            })

            if len(aug_rows) % 10 == 0:
                pd.DataFrame(aug_rows).to_csv(aug_file_path, index=False)
                print(f"Сохранено {len(aug_rows)} примеров")

            orig_idx = (orig_idx + 1) % orig_count
            need -= 1

    pd.DataFrame(aug_rows).to_csv(aug_file_path, index=False)
    print(f"\nИтого аугментированных примеров: {len(aug_rows)}")
    return pd.DataFrame(aug_rows)