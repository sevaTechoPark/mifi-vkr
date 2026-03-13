import random
import re
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.augmentation import (
    first_n_sentences,
    random_window_sentences,
    top_n_sentences_by_tfidf,
    random_delete_words,
    split_sentences,
)

TRAIN_PATH = "../data/train.csv"
TEST_PATH  = "../data/test.csv"

LABEL_COL = "label"
TEXT_COL = "text"
RANDOM_STATE = 42

# аугментация только для классов < THRESH
THRESH = 30
TARGET = 30

# параметры аугментации
N_SENT = 5
DROP_FRAC = 0.05  # 3–8%

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
_PLACEHOLDER_RE = re.compile(r"$$[A-Z_]+$$")

def make_features_word_char():
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)$$[A-Z_]+$$|\b\w\w+\b",
        lowercase=True,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        lowercase=True,
    )
    return FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])

def eval_model(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "weighted_f1": f1_score(y_test, pred, average="weighted"),
    }

def augment_train_df(train_df: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(RANDOM_STATE)

    vc = train_df[LABEL_COL].value_counts()
    small_labels = set(vc[vc < THRESH].index)

    # TF-IDF по предложениям для "top informative" (только по small-классам)
    all_sents = []
    for t in train_df.loc[train_df[LABEL_COL].isin(small_labels), TEXT_COL].astype(str):
        all_sents.extend(split_sentences(t))

    use_top = len(all_sents) >= 10
    sent_vec = TfidfVectorizer(min_df=2, max_df=0.95)
    if use_top:
        sent_vec.fit(all_sents)

    rows = []

    for label, cnt in vc.items():
        df_lbl = train_df[train_df[LABEL_COL] == label]
        # добавляем оригиналы всегда
        for _, r in df_lbl.iterrows():
            rows.append({TEXT_COL: str(r[TEXT_COL]), LABEL_COL: label, "aug": "orig"})

        # если класс не маленький — не аугментируем
        if label not in small_labels:
            continue

        # сколько нужно ДОБАВИТЬ, чтобы стало TARGET
        need = max(0, TARGET - cnt)
        if need == 0:
            continue

        # генерируем по кругу из имеющихся писем класса
        src_texts = df_lbl[TEXT_COL].astype(str).tolist()
        i = 0
        while need > 0:
            text = src_texts[i % len(src_texts)]
            i += 1

            # по одному аугменту за итерацию, циклим по типам
            mode = i % 4  # 0..3
            if mode == 0:
                aug_text = first_n_sentences(text, N_SENT)
                aug_name = "first_n"
            elif mode == 1:
                aug_text = random_window_sentences(text, N_SENT, rng)
                aug_name = "rand_window"
            elif mode == 2 and use_top:
                aug_text = top_n_sentences_by_tfidf(text, N_SENT, sent_vec)
                aug_name = "top_tfidf"
            else:
                aug_text = random_delete_words(text, DROP_FRAC, rng)
                aug_name = "drop_words"

            rows.append({TEXT_COL: aug_text, LABEL_COL: label, "aug": aug_name})
            need -= 1

    return pd.DataFrame(rows)

def run():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_test, y_test = test_df[TEXT_COL].astype(str), test_df[LABEL_COL].astype(str)

    features = make_features_word_char()
    models = {
        "linear_svm": LinearSVC(class_weight="balanced"),
        "logreg": LogisticRegression(max_iter=3000, class_weight="balanced"),
    }

    rows = []

    # baseline
    X_train_base = train_df[TEXT_COL].astype(str)
    y_train_base = train_df[LABEL_COL].astype(str)

    for clf_name, clf in models.items():
        pipe = Pipeline([("tfidf", features), ("clf", clf)])
        pipe.fit(X_train_base, y_train_base)
        rows.append({"variant": "baseline", "clf": clf_name, **eval_model(pipe, X_test, y_test)})

    # augmented (только для классов < THRESH)
    train_aug = augment_train_df(train_df)

    # --- stats: сколько добавили в каждый класс и всего ---
    base_counts = train_df[LABEL_COL].value_counts().sort_index()
    aug_counts = train_aug[LABEL_COL].value_counts().sort_index()

    # выравниваем индексы
    base_counts, aug_counts = base_counts.align(aug_counts, fill_value=0)

    added = (aug_counts - base_counts).astype(int)
    total_added = int(added.sum())

    added_pos = added[added > 0].sort_values(ascending=False)

    print("\nClass counts (before -> after) for augmented classes:")
    for lbl, add_n in added_pos.items():
        print(f"{lbl}: {int(base_counts[lbl])} -> {int(aug_counts[lbl])} (+{int(add_n)})")
    
    X_train_aug = train_aug[TEXT_COL].astype(str)
    y_train_aug = train_aug[LABEL_COL].astype(str)

    for clf_name, clf in models.items():
        pipe = Pipeline([("tfidf", features), ("clf", clf)])
        pipe.fit(X_train_aug, y_train_aug)
        rows.append({"variant": f"aug_lt_{THRESH}", "clf": clf_name, **eval_model(pipe, X_test, y_test)})

    res = pd.DataFrame(rows).sort_values(["weighted_f1", "macro_f1"], ascending=False)
    print("\n=== RESULTS (sorted by weighted_f1) ===")
    print(res.to_string(index=False))

if __name__ == "__main__":
    run()



# Подразделение по информационным технологиям: 1 -> 30 (+29)
# Имущественные вопросы: 1 -> 30 (+29)
# Проект «Трубопроводный транспорт Ещё одного НГКМ»: 1 -> 30 (+29)
# Блок заместителя генерального директора по строительству: 1 -> 30 (+29)
# Проект «Обустройство объектов Новейшей нейти»: 2 -> 30 (+28)
# Блок исполнительного директора по реализации проекта "Большое месторождение": 3 -> 30 (+27)
# Проект "Обустройство площадных объектов НГКМ Поменбше": 4 -> 30 (+26)
# Проект «Обустройство Интересного лицензионного участка»: 5 -> 30 (+25)
# Управление коммуникаций: 5 -> 30 (+25)
# Блок бизнес-директора: 6 -> 30 (+24)
# Проект "Южный": 9 -> 30 (+21)
# Блок заместителя генерального директора по имуществу: 13 -> 30 (+17)
# Проект "Восточный": 14 -> 30 (+16)
# Проект «Обустройство объектов Новой нефти»: 14 -> 30 (+16)
# Блок директора по персоналу: 14 -> 30 (+16)
# Проект "Трубопроводный транспорт Главного НГКМ": 15 -> 30 (+15)
# Управление землеустроительных работ: 18 -> 30 (+12)
# Блок директора по портфелю: 19 -> 30 (+11)
# Блок финансового директора: 20 -> 30 (+10)
# Блок заместителя генерального директора по защите: 20 -> 30 (+10)
# Блок директора по газовым проектам: 24 -> 30 (+6)
# Блок операционного директора: 26 -> 30 (+4)
# Проект "Северная деревня": 29 -> 30 (+1)

# === RESULTS (sorted by weighted_f1) ===
#   variant        clf  accuracy  macro_f1  weighted_f1
#  baseline linear_svm  0.639437  0.440005     0.617318
# aug_lt_30 linear_svm  0.630986  0.434816     0.608380
# aug_lt_30     logreg  0.594366  0.429042     0.574289
#  baseline     logreg  0.588732  0.425016     0.569212

# в текущем виде аугментация не помогла
