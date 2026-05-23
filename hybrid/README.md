# hybrid

Гибридный пайплайн классификации: разреженные TF-IDF признаки (word n-граммы + char_wb n-граммы) объединяются с плотными BERT-эмбеддингами в единый вектор, поверх которого работают линейные классические модели (LinearSVC, LogisticRegression, RidgeClassifier) и нейронный MLP-классификатор (StrongMLP).

Идея: TF-IDF чувствителен к специфической лексике и редким терминам (юридическим аббревиатурам, числам, маркерам формы), BERT — к семантике и контексту. На разных классах побеждает разный сигнал, поэтому конкатенация даёт более устойчивый ансамбль, чем любой из источников по отдельности.

## Как это работает

### Постановка задачи

Чисто BERT-классификация работает хорошо, когда классы различаются семантикой длинного текста. Чисто TF-IDF — когда в текстах есть стабильные ключевые слова (имена сторон, статьи кодекса, типы документов). На реальном корпусе оба сигнала пересекаются лишь частично: один класс хорошо отделяется по словарю, другой — по смыслу. Гибрид:

1. Строит TF-IDF-вектор $v_\text{tf}$ — разреженный, высокоразмерный, L2-нормированный.
2. Строит BERT-вектор $v_\text{bert}$ — плотный, низкоразмерный (1024 для ruRoberta), L2-нормированный.
3. Конкатенирует их с настраиваемым весом для BERT-блока:

$$v_\text{hybrid} = [\, v_\text{tf} \; \Vert \; w_b \cdot v_\text{bert} \,]$$

Линейные модели поверх такого вектора эффективно учатся выбирать важные TF-IDF-координаты (через L1) **и** одновременно опираться на смысловой BERT-блок. MLP — добавляет нелинейность и mixup-регуляризацию для шумных признаков.

### Архитектура

Пайплайн разделён на три независимых этапа — каждый запускается отдельной подкомандой:

```
                      ┌──────────────────────────────────────────────┐
                      │  hybrid build                                │
   train.csv ─────►   │  TF-IDF + BERT → hstack → L2 → save .npz/.npy│  ─────► vecdir/
   test.csv  ─────►   │  StandardScaler? + bert_weight              │           │
                      └──────────────────────────────────────────────┘           │
                                                                                 │
                            ┌───────────────────────────────────┐                │
                            │  hybrid classical (vecdir)        │   ◄────────────┤
                            │  LinearSVC + LogReg + Ridge       │                │
                            │  C-grid + L1 + Calibration        │ ──► JSON       │
                            └───────────────────────────────────┘                │
                                                                                 │
                            ┌───────────────────────────────────┐                │
                            │  hybrid mlp (vecdir)              │   ◄────────────┘
                            │  StrongMLP + FocalLoss + Mixup    │
                            │  AdamW + CosineAnneal             │ ──► JSON
                            └───────────────────────────────────┘
```

Промежуточные артефакты (vecdir) — это разделитель между этапами: после одного `build` можно сколько угодно раз запускать `classical` и `mlp` с разными гиперпараметрами без повторного эмбеддинга.

### TF-IDF блок

Используется композиция двух независимых TfidfVectorizer:

**Word n-grams** (`analyzer="word"`):
- `ngram_range=(1, 2)` — униграммы и биграммы.
- `min_df=2`, `max_df=0.98` — фильтр редчайших и почти-стоп-слов.
- `sublinear_tf=True` — заменяет `tf` на $1 + \log(tf)$, сглаживает влияние очень частых терминов.
- `token_pattern=r"(?u)\b\w\w+\b"` — минимум 2 буквы, поддержка кириллицы.

**Character n-grams** (`analyzer="char_wb"`):
- `ngram_range=(3, 5)` — символьные n-граммы в пределах слова.
- `min_df=2`, `max_df=0.95` — чуть строже, потому что char-словарь и так большой.

`char_wb` (word boundary char-grams) полезен на русском: ловит словоформы, опечатки, суффиксы (`-ость`, `-ение`) — то, что word-униграммы пропустят. Композиция объединяется через `scipy.sparse.hstack`:

$$v_\text{tf} = \text{normalize}_{L2}\left( [\, v_\text{word} \;\Vert\; v_\text{char} \,] \right)$$

Поблочная L2-нормировка означает, что итоговый вектор имеет норму 1 — необходимое условие, чтобы скаляр $w_b$ при BERT-блоке предсказуемо контролировал соотношение энергий.

### BERT блок

Документные эмбеддинги получаются одним из двух способов:

1. **С `--model-dir`**: используется `LongTextRobertaEmbedder` из `bert_embeddings` — тот же пайплайн чанкования и усреднения, что и в основном модуле. Эмбеддинги уже L2-нормированы из-за `Normalize()` слоя в дообученной sentence-transformer модели.

2. **Без `--model-dir`**: используется базовая `ai-forever/ruRoberta-large` (по умолчанию), напрямую через transformers. Mean-pooling с учётом attention_mask + L2-нормировка вручную:

$$v_\text{bert} = \frac{1}{\sum_i m_i} \sum_i m_i h_i, \quad \hat v_\text{bert} = \frac{v_\text{bert}}{\lVert v_\text{bert} \rVert_2}$$

где $h_i$ — last_hidden_state, $m_i$ — маска внимания. Базовая модель не выровнена под задачу и даёт более «сырые» эмбеддинги, поэтому к ней применяется `StandardScaler` (вычитание среднего и деление на std по каждой компоненте), затем повторная L2-нормировка.

### Взвешивание BERT-блока

Ключевой гиперпараметр гибрида — соотношение энергий TF-IDF и BERT. Если $v_\text{tf}$ и $v_\text{bert}$ оба L2-нормированы и BERT входит с весом $w_b$:

$$\lVert v_\text{hybrid} \rVert_2^2 = \lVert v_\text{tf} \rVert_2^2 + w_b^2 \lVert v_\text{bert} \rVert_2^2 = 1 + w_b^2$$

Доля энергии BERT в гибриде:

$$\rho_\text{bert} = \frac{w_b^2}{1 + w_b^2}$$

При $w_b = 1$ доля = 50%. При $w_b = 5$ доля = 96% — BERT доминирует. Это сделано намеренно: дообученные BERT-эмбеддинги сильно информативнее, чем TF-IDF, и должны иметь больший вес.

Дефолтная политика выбора $w_b$:
- **`--model-dir` задан** (дообученный энкодер): $w_b = 5.0$ — BERT-блок доминирует, TF-IDF добавляет редкие лексические сигналы.
- **`--model-dir` не задан** (базовая модель): $w_b = 1.0$ — равный вклад, потому что базовый BERT слабее и нет смысла его переоценивать.

Эта политика применяется только если пользователь не задал `--bert-weight` явно (проверка через `sys.argv`).

В meta.json сохраняется фактическая средняя доля BERT-энергии:

$$\bar\rho_\text{bert} = \frac{1}{N} \sum_{i=1}^{N} \frac{w_b^2 \lVert v_\text{bert}^{(i)} \rVert_2^2}{\lVert v_\text{tf}^{(i)} \rVert_2^2 + w_b^2 \lVert v_\text{bert}^{(i)} \rVert_2^2}$$

Это нужно для диагностики: если фактическая доля сильно отличается от ожидаемой, значит, какой-то блок был не нормирован корректно.

### Две версии гибридного вектора

После hstack делаются **две версии** одного и того же гибрида:

| Файл | L2-нормировка | Для кого |
| --- | --- | --- |
| `X_*_hybrid.npz` | финальная L2 после hstack | MLP |
| `X_*_hybrid_noL2.npz` | только поблочная (TF-IDF и BERT раздельно) | LinearSVC, LogReg |

Зачем две версии? Линейные классификаторы (LinearSVC, LogReg) сами регуляризуют норму через C — финальная L2 для них вредна, потому что обрезает дисперсию между документами разной длины. MLP, наоборот, любит входы с предсказуемой нормой — LayerNorm внутри сети ожидает примерно унитарный масштаб.

### Опциональный TruncatedSVD

Если задано `svd_components > 0`, гибридный sparse-вектор проецируется в плотное низкоразмерное пространство через `TruncatedSVD` (LSA):

$$X_\text{dense} = X_\text{hybrid} \cdot V_k^\top, \quad k = \min(\text{components}, n_\text{features}-1, n_\text{samples}-1)$$

Зачем: MLP на sparse-входе требует материализации в dense → большой расход памяти. SVD даёт компактное dense-представление с сохранением основной структуры. По умолчанию отключено (`svd_components=0`).

### Classical-модели — линейная воронка

Для каждого источника признаков (`bert_only`, `hybrid`, `tfidf_only`) прогоняется фиксированная воронка:

1. **LinearSVC** — линейный SVM, perепебор C по сетке `(0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0)`. `dual=False` для разреженных матриц (sklearn рекомендует), `max_iter=10000` для сходимости.

2. **LogisticRegression** — `solver=lbfgs`, тот же C-grid.

3. **LinearSVC-L1** — `penalty="l1"`, `loss="squared_hinge"`, `dual=False`. L1 даёт sparse-веса: модель сама отбирает важные TF-IDF координаты. Особенно полезно на TF-IDF блоке (десятки тысяч n-грамм, из которых работают сотни).

4. **LogReg-L1** — через `penalty="elasticnet"`, `l1_ratio=1.0`, `solver="saga"` (эквивалент чистой L1 в современном sklearn API). Прогоняется только на dense-входе: saga на огромном sparse сходится медленно.

5. **Калибровка вероятностей** — для лучшего LinearSVC по C запускается `CalibratedClassifierCV` с двумя методами:
   - `sigmoid` (Platt scaling) — параметрическая;
   - `isotonic` — непараметрическая, требует больше данных.

   Условие: в каждом классе ≥ 2 примера (для CV). Калибровка полезна, когда нужны ранжированные вероятности, а не только argmax.

6. **RidgeClassifier** — L2-регуляризованная линейная регрессия, обученная на one-hot. Без C-grid — берётся sklearn-дефолт.

**class_weight**: по умолчанию `"balanced"` (веса = `n_samples / (n_classes * n_c)`). На дисбалансированных корпусах с min_class=1 это критично — без баланcировки модель просто игнорирует редкие классы.

### MLP — StrongMLP с регуляризацией

Архитектура: проекция входа → N остаточных блоков → классификационная голова.

**Входная проекция**:
```
Linear(input_dim → hidden_dim) → LayerNorm → GELU → Dropout
```

**ResidualBlock** (повторяется `num_blocks` раз):
```
x ──┬──► LayerNorm → Linear(dim → 2·dim) → GELU → Dropout
    │                                                   │
    │                                                   ▼
    │    LayerNorm → Linear(2·dim → dim) → Dropout ───►(+) ── output
    └────────────────────────────────────────────────►(+)
```

Расширение «вдоль ширины» (dim → 2·dim → dim) — стандартный приём из transformers (FFN-блок), помогает выучивать нелинейные взаимодействия признаков. LayerNorm перед каждым линейным слоем стабилизирует обучение при больших весах TF-IDF координат.

**Голова**:
```
LayerNorm → Linear(hidden_dim → hidden_dim/2) → GELU → Dropout → Linear(hidden_dim/2 → num_classes)
```

**FocalLoss с label smoothing**:

$$\mathcal{L}_\text{focal} = -\alpha_c (1 - p_t)^\gamma \log p_t$$

где $p_t$ — предсказанная вероятность истинного класса (с применённым label smoothing $\varepsilon=0.05$), $\alpha_c$ — балансированный вес класса, $\gamma$ — гиперпараметр фокуса. При $\gamma=0$ получается обычный CE; при $\gamma>0$ хорошо предсказанные примеры ($p_t \to 1$) штрафуются меньше, плохо предсказанные — сильнее.

Реализация:
```
ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none", label_smoothing=0.05)
pt = exp(-ce)
focal = (1 - pt)^gamma * ce
```

**Mixup в пространстве признаков** (а не входов):

Стандартный mixup смешивает входы. Здесь смешиваются скрытые представления **после** `forward_features` (т.е. после блоков, до головы):

$$\tilde h = \lambda h_a + (1-\lambda) h_b, \quad \tilde y = \lambda y_a + (1-\lambda) y_b, \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$

Почему в feature-space, а не на входе: гибридный вектор разреженный (с нулями на большинстве TF-IDF координат), линейное смешение даёт нестабильные интерполяции в TF-IDF-части. В hidden-space все компоненты плотные — mixup работает корректно. Для soft-меток используется отдельный `_soft_cross_entropy` (обычный F.cross_entropy не принимает one-hot/soft).

Если $\alpha \le 0$, mixup отключается; иначе $\lambda$ берётся из Beta-распределения. При $\alpha = 0.2$ распределение бимодальное (большая часть массы около 0 и 1), т.е. чаще одно из исходных значений доминирует.

**Оптимизация**:
- AdamW, learning_rate=3e-4 (для noisy профиля), weight_decay=1e-2.
- CosineAnnealingLR от lr до min_lr=1e-6 за `epochs` шагов.
- `clip_grad_norm_(max_grad_norm=1.0)` — стандартная защита от взрывов градиента при первых эпохах.

**Early stopping** по `macro_f1` с `patience=12` (для noisy профиля). Лучший чекпойнт фиксируется и используется для финальных метрик.

### Профили MLP

Три заранее заданных набора гиперпараметров:

| Профиль | Назначение | Lr | Dropout | Blocks | Hidden | Focal | LS | Mixup | CW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **noisy** | Разреженные TF-IDF+BERT, гибрид | 3e-4 | 0.4 | 2 | 512 | 1.0 | 0.05 | 0.2 | да |
| **clean** | Плотные L2-нормированные BERT-эмбеддинги | 2e-4 | 0.2 | 1 | 384 | 0.0 | 0.0 | 0.0 | нет |
| **custom** | Промежуточный | 3e-4 | 0.3 | 2 | 512 | 0.5 | 0.03 | 0.1 | да |

Профиль выбирается автоматически от `--features`:
- `features=hybrid` → `noisy`;
- `features=bert_only` → `clean`.

Можно перекрыть через `--profile clean|noisy|custom` и любые отдельные поля через CLI.

### Сохранение результатов

Каждый этап сохраняет JSON в `vecdir`:

| Этап | Файл |
| --- | --- |
| build | `meta.json` (параметры построения + диагностика энергий) |
| classical | `classical-results-<vecdir_base>-<YYYY-MM-DDTHH:MM>.json` |
| mlp | `mlp-results-<vecdir_base>-<YYYY-MM-DDTHH-MM>.json` |

JSON classical содержит все прогоны по сетке C, топ-10 моделей, лучшую конфигурацию. JSON mlp — историю обучения по эпохам, per-class F1 на лучшей эпохе, полную конфигурацию.

## Структура модуля

| Файл | Содержимое |
| --- | --- |
| `config.py` | Датаклассы `HybridModelConfig`, `HybridDataConfig`, `HybridPathConfig`, `HybridMLPConfig` + профили MLP. |
| `hybrid_vector_build.py` | Этап `build`: TF-IDF, BERT, hstack, L2, опциональный SVD. |
| `hybrid_classical_models.py` | Этап `classical`: воронка линейных моделей по трём источникам. |
| `hybrid_mlp.py` | Этап `mlp`: StrongMLP + FocalLoss + mixup. |
| `main.py` | Единый CLI с подкомандами `build`/`classical`/`mlp`. |
| `__init__.py` | Реэкспорт основных классов и функций. |

## CLI

Все три этапа запускаются через `python -m hybrid.main <subcommand>`.

### Этап 1 — build

Минимальный запуск (использует базовую модель ruRoberta, $w_b=1$):

```bash
python -m hybrid.main build \
    --train-file data/train.csv \
    --test-file data/test.csv \
    --output-dir runs/hybrid_v1
```

С дообученным энкодером (автоматически $w_b=5$, scaler отключён):

```bash
python -m hybrid.main build \
    --train-file data/train.csv \
    --test-file data/test.csv \
    --output-dir runs/hybrid_v2 \
    --model-dir /content/drive/MyDrive/vkr/models/xlm-roberta-base-finetuned
```

С опциональным SVD до 512 компонент:

```bash
python -m hybrid.main build ... --svd-components 512
```

### Этап 2 — classical

Полный прогон по трём источникам с балансировкой классов:

```bash
python -m hybrid.main classical --vecdir runs/hybrid_v1
```

С узкой настройкой (только гибрид, без TF-IDF baseline, кастомный C-grid):

```bash
python -m hybrid.main classical \
    --vecdir runs/hybrid_v1 \
    --feature-sources hybrid \
    --c-grid "0.1,0.5,1.0,5.0" \
    --no-tfidf-only
```

### Этап 3 — mlp

Профиль выбирается автоматически от `--features`:

```bash
python -m hybrid.main mlp --vecdir runs/hybrid_v1 --features hybrid
python -m hybrid.main mlp --vecdir runs/hybrid_v1 --features bert_only
```

С переопределением отдельных гиперпараметров:

```bash
python -m hybrid.main mlp \
    --vecdir runs/hybrid_v1 \
    --features hybrid \
    --profile noisy \
    --epochs 60 \
    --learning-rate 2e-4 \
    --mixup-alpha 0.3
```

## Python API

```python
from hybrid.config import HybridModelConfig, HybridDataConfig, hybrid_mlp_config_from_profile
from hybrid.hybrid_vector_build import run_build
from hybrid.hybrid_classical_models import run_classical
from hybrid.hybrid_mlp import run_mlp

# Этап 1
run_build(
    train_file="data/train.csv",
    test_file="data/test.csv",
    outdir="runs/hybrid_v1",
    model_dir="path/to/finetuned",
    device="cuda",
    model_cfg=HybridModelConfig(bert_weight=5.0, svd_components=512),
    data_cfg=HybridDataConfig(),
)

# Этап 2
run_classical("runs/hybrid_v1", class_weight="balanced")

# Этап 3
cfg = hybrid_mlp_config_from_profile("noisy", epochs=60)
run_mlp("runs/hybrid_v1", cfg=cfg, feature_source="hybrid")
```

## Входы

| Файл | Колонки | Формат |
| --- | --- | --- |
| `--train-file` | `text`, `label` | CSV |
| `--test-file` | `text`, `label` | CSV |
| `--model-dir` (опц.) | — | директория с дообученным sentence-transformer |

## Выходы

### После `build` (содержимое `vecdir`)

| Файл | Описание |
| --- | --- |
| `X_train_bert.npy` / `X_test_bert.npy` | L2-нормированные BERT-эмбеддинги (1024-d). |
| `X_train_hybrid.npz` / `X_test_hybrid.npz` | Гибридный sparse-вектор, финальная L2 (для MLP). |
| `X_train_hybrid_noL2.npz` / `X_test_hybrid_noL2.npz` | Гибрид без финальной L2 (для linear-моделей). |
| `X_train_dense.npy` / `X_test_dense.npy` | Опционально: dense-представление после SVD. |
| `texts_train.csv` / `texts_test.csv` | Исходные тексты+метки (для tfidf_only baseline). |
| `y_train.csv` / `y_test.csv` | Метки. |
| `word_tfidf.joblib`, `char_tfidf.joblib` | Обученные векторизаторы. |
| `scaler_bert.joblib` | StandardScaler (`None` для дообученного энкодера). |
| `svd.joblib` | Опционально: SVD-проектор. |
| `meta.json` | Конфигурация + диагностика (`bert_share_mean`, размерности). |

### После `classical`

| Файл | Описание |
| --- | --- |
| `classical-results-<vecdir>-<ts>.json` | Все прогоны линейных моделей с метриками. |

### После `mlp`

| Файл | Описание |
| --- | --- |
| `mlp-results-<vecdir>-<ts>.json` | История обучения по эпохам, лучшие метрики, per-class F1. |

## Ключевые параметры

### HybridModelConfig (build)

| Параметр | Дефолт | Назначение |
| --- | --- | --- |
| `bert_weight` | 1.0 / 5.0 | Вес BERT-блока в гибриде. Авто от `--model-dir`. |
| `disable_bert_scaler` | False | Отключить StandardScaler (авто True для дообученного). |
| `word_ngram_min/max` | 1 / 2 | Диапазон word n-грамм. |
| `char_ngram_min/max` | 3 / 5 | Диапазон char_wb n-грамм. |
| `word_min_df`, `word_max_df` | 2, 0.98 | Фильтр частот word. |
| `char_min_df`, `char_max_df` | 2, 0.95 | Фильтр частот char. |
| `svd_components` | 0 | Размерность SVD (0 = отключено). |

### HybridMLPConfig (mlp, профиль noisy)

| Параметр | Дефолт | Назначение |
| --- | --- | --- |
| `learning_rate` | 3e-4 | LR для AdamW. |
| `weight_decay` | 1e-2 | L2-регуляризация в AdamW. |
| `batch_size` | 128 | — |
| `epochs` | 40 | Максимум эпох. |
| `patience` | 12 | Early stopping по macro_f1. |
| `hidden_dim` | 512 | Ширина скрытого слоя. |
| `num_blocks` | 2 | Кол-во остаточных блоков. |
| `dropout` | 0.4 | Применяется во всех Dropout-слоях. |
| `focal_gamma` | 1.0 | Параметр фокуса в FocalLoss. |
| `label_smoothing` | 0.05 | $\varepsilon$ в label smoothing. |
| `mixup_alpha` | 0.2 | Параметр Beta для mixup. |
| `use_class_weight` | True | Балансировка через `compute_class_weight`. |
| `max_grad_norm` | 1.0 | Клиппинг градиента. |
| `min_lr` | 1e-6 | Минимальный LR для CosineAnnealingLR. |

## Метрики

Везде используются:
- **balanced_accuracy** — основная метрика, устойчивая к дисбалансу.
- **macro_f1** — лучшая эпоха MLP выбирается по нему; classical-модели сравниваются по balanced_accuracy.

Для MLP дополнительно сохраняется **per_class_f1** на лучшей эпохе — полезно для понимания, какие классы модель не выучила.
