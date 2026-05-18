# bert_embeddings

Дообучение `ruRoberta-large` под доменные эмбеддинги через
**Sentence-Transformers** + contrastive loss. На выходе — энкодер,
из которого извлекается векторное представление документа для
downstream-задач (`cosine_similarity_classification`, `hybrid`).

## Идея

Из `text / label` строится много пар:
- **positive** — два фрагмента **разных** документов одного класса
- **negative** — фрагменты документов разных классов

И на этих парах оптимизируется `MultipleNegativesRankingLoss` (MNR):
для каждого positive в батче все остальные тексты выступают как
negatives. Это in-batch softmax loss, классический рецепт SBERT.

### Почему именно так

В первой версии positive строились из **всех чанков одного класса**, и
overlapping-чанки одного документа попадали в одну пару. Через MNR это
давало тривиальные положительные пары (косинус ≈ 1 с первого шага),
loss насыщался, а представления коллапсировали в шар. Сейчас в
`build_pair_dataframe` стоит `cross_document_positives_only=True` —
обязательное условие для нормального MNR на длинных русских текстах.

### Mismatch обучение↔инференс — главный фикс прошлой итерации

Раньше энкодер обучался с `sentence_pooling="mean"`, а
`LongTextRobertaEmbedder` на инференсе использовал `pooling="mean_max"`.
Это давало catastrophic drop: `cosine_centroid 0.249 → 0.031` на
кастомных эмбеддингах. Сейчас `EmbeddingConfig.pooling == MLMConfig.sentence_pooling = "mean"` —
обучение и инференс используют один и тот же pooling-режим.

## Что лежит в модуле

| Файл | Назначение |
| --- | --- |
| `config.py` | `MLMConfig` (обучение), `EmbeddingConfig` (инференс) |
| `data_utils.py` | Загрузка CSV, чанкование (`explode_long_texts_for_training`), построение пар (`build_pair_dataframe`) |
| `main.py` | Сборка SentenceTransformer + Trainer, кастомные callbacks, экспорт `best_model/` |
| `mlm_train.py` | Алтернативный путь — чистый MLM (если нужно) |
| `embedding_model.py` | `LongTextRobertaEmbedder` — инференс с чанк-агрегацией |
| `embed_texts.py` | Утилита генерации эмбеддингов из CSV |

## Конфиг

```python
@dataclass
class MLMConfig:
    # ОБУЧЕНИЕ
    sentence_pooling: str = "mean"          # MATCH с EmbeddingConfig.pooling
    train_loss: str = "mnr"                 # softmax | cosent | mnr
    cross_document_positives_only: bool = True
    max_pairs_per_label: int = 5000
    max_negative_pairs: int = 5000
    num_train_epochs: int = 3
    freeze_lower_layers: int = 12           # на 1-й эпохе

@dataclass
class EmbeddingConfig:
    # ИНФЕРЕНС — должен совпадать с обучением
    pooling: str = "mean"
    chunk_aggregation: str = "mean"
    chunk_size: int = 448
    chunk_overlap: int = 128
```

## Кастомные callback-и

- `FreezeLowerLayersCallback(N)` — на 1-й эпохе замораживает embeddings
  и нижние N=12 слоёв из 24. На 2-й — размораживает всё. Помогает не
  сломать pretraining за первые шаги.
- `RollingResumeCheckpoint` — один файл `resume_checkpoint.pt`,
  перезаписывается после каждой эпохи (encoder_state_dict +
  optimizer + scheduler + epoch).

`EarlyStoppingCallback` — стандартный из `transformers`.

## Что пишется на диск
```
<output_dir>/
├── best_model/ ← лучший энкодер (pytorch_model.bin + tokenizer + meta)
├── resume_checkpoint.pt ← один rolling-файл для resume
└── metrics.json ← конфиг + финальные метрики
```

**Важно:** промежуточные чекпоинты HF Trainer-а пишутся в
`tempfile.mkdtemp(prefix="st_trainer_tmp_")` — то есть в системный
`/tmp` инстанса Colab. Это сделано специально, чтобы НЕ заполнять
Google Drive, на который указывает `output_dir`. В конце временная
папка удаляется.

## A100 / L4 параметры

- `bf16=True` авто-включается на Ampere+ (A100, RTX30+), иначе fallback.
- `gradient_checkpointing=True` через `SentenceTransformerTrainingArguments`.
- `tf32=True` для matmul на CUDA.
- `dataloader_num_workers=2`, `pin_memory=True`.
- `train_batch_size=32`, `freeze_lower_layers=12` — типичная картинка
  обучения за 1-1.5 ч на L4 для ruRoberta-large на ~2500 чанках.

## Использование

```bash
python -m bert_embeddings.main \
    --train-file data/train.csv \
    --test-file  data/test.csv \
    --output-dir /content/drive/MyDrive/.../bert_embeddings_best
```

или из ноутбука:

```python
from bert_embeddings.main import run_from_params

components, metrics = run_from_params(
    train_file="data/train.csv",
    test_file="data/test.csv",
    output_dir="/content/drive/MyDrive/.../bert_embeddings_best",
    num_train_epochs=3,
    train_batch_size=32,
)
```

## Использование энкодера на инференсе

```python
from bert_embeddings.embedding_model import LongTextRobertaEmbedder

embedder = LongTextRobertaEmbedder(
    model_dir="/content/drive/.../bert_embeddings_best/best_model",
    base_model_name="ai-forever/ruRoberta-large",
    pooling="mean",            # ОБЯЗАТЕЛЬНО матчить с MLMConfig.sentence_pooling
    chunk_aggregation="mean",
)
embs = embedder.encode(["длинный русский текст ..."])
```
