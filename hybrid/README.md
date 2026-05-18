# hybrid

Гибридное представление: `TF-IDF(word + char) ⊕ scale * BERT_embedding`
поверх него — линейные модели (LinearSVC, LogReg, RidgeClassifier,
calibrated SVC) и `StrongMLP` (residual MLP с focal loss и mixup).

## Зачем

TF-IDF берёт **поверхностные сигналы** (характерные n-граммы,
char n-grams для опечаток, окончания и редкие термины). BERT берёт
**семантику и контекст**. Конкатенация даёт сильный baseline на
small-data русских классификациях, где ни один из источников по
отдельности не идеален.

## Архитектура
```
text →
TfidfVectorizer(word, 1-2) ─┐
├─→ L2 row-norm → hstack
TfidfVectorizer(char_wb,3-5) ─┘ │
│ scale by bert_weight
LongTextRobertaEmbedder → StandardScaler ────────┴──────────────────────→
│
▼
hstack → L2 row-norm
│
▼
[опц.] TruncatedSVD(256)
│
▼
LinearSVC / LogReg / RidgeClassifier
или StrongMLP
```


## Важный гиперпараметр: `bert_weight`

После L2-нормировки и StandardScaler-а единственное, что отличает
вклад TF-IDF от вклада BERT — это `bert_weight`. Прежний дефолт `5.0`
давал `bert_share_mean ≈ 0.95` — то есть TF-IDF фактически не
участвовал в косинусной близости, и гибрид работал хуже, чем чистый
BERT. Новый дефолт `1.0` ставит блоки в одну весовую категорию;
для проверки реального вклада смотри `meta.json → bert_share_mean`
после `build`.

## Что лежит

| Файл | Назначение |
| --- | --- |
| `config.py` | `HybridModelConfig`, `HybridDataConfig`, `HybridPathConfig`, `HybridMLPConfig` |
| `hybrid_vector_build.py` | Сборка гибридных векторов; сохраняет `*.npz`, `*.joblib`, `meta.json` |
| `hybrid_classical_models.py` | LinearSVC, LinearSVC-calibrated, LogReg, RidgeClassifier + TF-IDF-only baselines (MultinomialNB, ComplementNB, LogReg) |
| `hybrid_mlp.py` | `StrongMLP` (residual + focal + mixup), early stop по `macro_f1` |
| `main.py` | CLI с subcommand-ами: `build`, `classical`, `mlp` |

## TF-IDF — почему word + char_wb

- `word` (1-2 граммы) ловит ключевые слова и биграммы — основной сигнал.
- `char_wb` (3-5 граммы) ловит:
  - окончания в русском (`-ение`, `-овать`, `-овых`),
  - опечатки (если в реальных данных они есть),
  - редкие термины и слитные слова (`OZON-доставка`).

`sublinear_tf=True` (использует `1+log(tf)` вместо `tf`) — индустриальный
дефолт для русских текстов: уменьшает доминирование частых слов.

## Что пишется на диск

После `build`:
```
<vecdir>/
├── X_train_hybrid.npz / X_test_hybrid.npz sparse hybrid vectors
├── X_train_dense.npy / X_test_dense.npy (только если svd_components > 0)
├── y_train.csv / y_test.csv
├── texts_train.csv / texts_test.csv (для TF-IDF-only baseline)
├── word_tfidf.joblib / char_tfidf.joblib
├── scaler_bert.joblib / svd.joblib
└── meta.json bert_share_mean, tfidf_dim, bert_dim, ...
```


После `classical` / `mlp`:
```
<vecdir>/classical_results.json
<vecdir>/mlp_results.json
```


## Использование

### Шаг 1: построить гибридные векторы

С baseline-моделью (`ai-forever/ruRoberta-large` без дообучения):
```bash
python -m hybrid.main build \
    --train-file data/train.csv \
    --test-file  data/test.csv \
    --output-dir runs/hybrid_baseline
```

С кастомной моделью после `bert_embeddings`:
```bash
python -m hybrid.main build \
    --train-file data/train.csv \
    --test-file  data/test.csv \
    --output-dir runs/hybrid_custom \
    --model-dir /content/drive/.../bert_embeddings_best/best_model
```

### Шаг 2a: классические линейные модели

```bash
python -m hybrid.main classical --vecdir runs/hybrid_custom
# linear_svc: {'balanced_accuracy': 0.533, 'macro_f1': 0.500}
# linear_svc_calibrated: {'balanced_accuracy': 0.541, 'macro_f1': 0.494}
# logreg: {'balanced_accuracy': 0.542, 'macro_f1': 0.494}
# ridge_classifier: {'balanced_accuracy': 0.524, 'macro_f1': 0.492}
# multinomial_nb_tfidf_only: ...
# complement_nb_tfidf_only: ...
# logreg_tfidf_only: ...
```

С `class_weight="balanced"` (для сильно дисбалансированных классов):
```bash
python -m hybrid.main classical --vecdir runs/hybrid_custom --class-weight balanced
```

### Шаг 2b: StrongMLP

```bash
python -m hybrid.main mlp --vecdir runs/hybrid_custom \
    --epochs 40 --learning-rate 3e-4 --mixup-alpha 0.2
```

## StrongMLP — что внутри

- `input_proj`: `Linear → LayerNorm → GELU → Dropout`.
- `N=2` residual-блоков: `LayerNorm → Linear(d→2d) → GELU → Dropout → LayerNorm → Linear(2d→d) → Dropout` + residual.
- `head`: `LayerNorm → Linear(d→d/2) → GELU → Dropout → Linear(d/2→num_classes)`.
- Лосс: focal loss (γ=1.0, label_smoothing=0.05). С mixup в feature
  space — soft cross-entropy на смешанных one-hot целях.
- Оптимайзер: AdamW + CosineAnnealingLR с `eta_min=1e-6`.
- Early stop: patience по `macro_f1` (простой ручной счётчик, не
  дублирует никакой PyTorch-стандарт).

## Mixup в feature space

```python
features = model.forward_features(x)         # после input_proj+residual
lam = Beta(α, α).sample()
idx = randperm(B)
mixed = lam * features + (1-lam) * features[idx]
y_soft = lam * onehot(y) + (1-lam) * onehot(y[idx])
loss = soft_CE(model.head(mixed), y_soft)
```

`α=0.2` — мягкий mixup, помогает на малых классах. Если ловишь
плато ниже baseline — попробуй `α=0` (отключить) или `α=0.4`.

## Ожидаемые результаты на тебе (на `[custom_embeder]` после v3)

| Модель | balanced_acc | macro_f1 |
| --- | --- | --- |
| linear_svc | 0.53+ | 0.50+ |
| logreg | 0.54+ | 0.49+ |
| ridge | 0.52+ | 0.49+ |
| MLP | 0.45+ | 0.40+ (mixup, может расти выше) |
| MultinomialNB tfidf-only | 0.25-0.35 | 0.20-0.30 |

Linear-модели обычно > MLP на small-data — это нормально. MLP включай
когда заведомо много примеров и SVD-проекцию (`svd_components=256`).