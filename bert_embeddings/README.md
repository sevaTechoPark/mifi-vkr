# bert_embeddings

Пакет для доменного дообучения `ai-forever/ruRoberta-large` по MLM и построения эмбеддингов длинных русскоязычных документов.[conversation_history:2]

## Структура пакета

- `__init__.py` — экспорт `LongTextRobertaEmbedder`
- `config.py` — датаклассы `MLMConfig` и `EmbeddingConfig` с дефолтами
- `data_utils.py` — сбор корпуса для MLM из train/test CSV
- `main.py` — функция `run_from_params(...)` и логика обучения MLM
- `mlm_train.py` — CLI-обёртка поверх `run_from_params`
- `embedding_model.py` — реализация `LongTextRobertaEmbedder` (chunking + pooling)
- `embed_texts.py` — CLI для генерации эмбеддингов на CSV

## Что делает пайплайн

1. Собирает MLM-корпус как конкатенацию текстов из `train` и `test` CSV по колонке `text`.[conversation_history:2]
2. Делит корпус на train/valid по `val_size` (по умолчанию 2% на валидацию).[conversation_history:2]
3. Дообучает `ai-forever/ruRoberta-large` как Masked Language Model на этом корпусе.[conversation_history:2]
4. Каждые `checkpoint_every_n_epochs` эпох сохраняет encoder-only чекпоинты + лучший чекпоинт по `eval_loss`.[conversation_history:2]
5. Сохраняет финальную модель и метрики (`metrics.json`) в одной выходной директории.[conversation_history:2]
6. Загружает encoder-веса из MLM-чекпоинта и строит эмбеддинги длинных документов через token-based chunking с overlap.[conversation_history:2]
7. Агрегирует chunk embeddings в один document embedding для cosine similarity, кластеризации и построения композитных векторов.[conversation_history:2]

## Основные настройки (config.py)

Все дефолты живут в `config.py` и могут быть переопределены через CLI или из Python.

### MLMConfig

```python
@dataclass
class MLMConfig:
    model_name: str = "ai-forever/ruRoberta-large"
    text_col: str = "text"
    max_length: int = 512
    mlm_probability: float = 0.15
    train_batch_size: int = 4
    eval_batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 15
    warmup_ratio: float = 0.10
    logging_steps: int = 100
    fp16: bool = True
    seed: int = 42
    val_size: float = 0.02
    checkpoint_every_n_epochs: int = 3
    early_stopping_patience: int | None = 3
```

По умолчанию включены:

- уменьшенный `learning_rate=2e-5` для стабильного дообучения `large`‑модели;
- `warmup_ratio=0.10` для более плавного разогрева;
- ранняя остановка по `eval_loss` с `early_stopping_patience=3` и выбором лучшей модели.[conversation_history:2]

### EmbeddingConfig

```python
@dataclass
class EmbeddingConfig:
    max_length: int = 512
    chunk_size: int = 448
    chunk_overlap: int = 96
    pooling: str = "mean_max"
    chunk_aggregation: str = "mean_max"
    batch_size: int = 8
    normalize_chunks: bool = True
    normalize_document: bool = True
    add_global_chunk: bool = True
    base_model_name: str = "ai-forever/ruRoberta-large"
```

Рекомендуемые значения для деловых писем уже зашиты как дефолты.[conversation_history:2]

## Установка

```bash
pip install torch transformers datasets pandas numpy pyarrow
```

(версии `torch` и `transformers` должны быть совместимы с `ai-forever/ruRoberta-large` и `Trainer` из `transformers`).[conversation_history:2]

## Обучение MLM через CLI

Запускать из корня репозитория:

```bash
python -m bert_embeddings.mlm_train \
  --train-file /path/train_paraphrase_2.csv \
  --test-file /path/test.csv \
  --output-dir /path/saved_models/bert_embeddings_best
```

Это минимальная команда: все остальные параметры берутся из `MLMConfig`. Если нужно, их можно переопределить флагами, соответствующими полям датакласса (snake_case → kebab-case):

- `num_train_epochs` → `--num-train-epochs`
- `train_batch_size` → `--train-batch-size`
- `eval_batch_size` → `--eval-batch-size`
- `learning_rate` → `--learning-rate`
- `warmup_ratio` → `--warmup-ratio`
- `val_size` → `--val-size`
- `checkpoint_every_n_epochs` → `--checkpoint-every-n-epochs`
- `early_stopping_patience` → `--early-stopping-patience`

Пример с явными гиперпараметрами:

```bash
python -m bert_embeddings.mlm_train \
  --train-file /path/train_paraphrase_2.csv \
  --test-file /path/test.csv \
  --output-dir /path/saved_models/bert_embeddings_best \
  --num-train-epochs 40 \
  --train-batch-size 4 \
  --eval-batch-size 4 \
  --learning-rate 2e-5 \
  --warmup-ratio 0.10 \
  --val-size 0.05 \
  --checkpoint-every-n-epochs 5 \
  --early-stopping-patience 3
```

Что получится в `output-dir`:

- `final_model/` — финальная MLM‑модель и токенизатор;
- `best_model/` — encoder-only веса лучшей модели по `eval_loss` + токенизатор;
- `checkpoints/epoch_XXX/` — encoder-only чекпоинты каждые `checkpoint_every_n_epochs` эпох;
- `metrics.json` — сводка метрик (`eval_loss`, `perplexity`, `best_eval_loss`, `best_epoch` и все ключевые гиперпараметры).[conversation_history:2]

## Обучение MLM из Python (например, в Colab)

```python
from bert_embeddings.main import run_from_params

components, metrics = run_from_params(
    train_file="/content/train_paraphrase_2.csv",
    test_file="/content/test.csv",
    output_dir="/content/saved_models/bert_embeddings_best",
    num_epochs=40,                    # алиас для num_train_epochs
    checkpoint_every_n_epochs=5,
    learning_rate=2e-5,
    early_stopping_patience=3,
)
```

`run_from_params` использует `MLMConfig` как источник дефолтов и возвращает словарь с путями (`best_model_dir`, `final_model_dir`, `metrics_path`) и метрики последней и лучшей модели.[conversation_history:2]

## Генерация эмбеддингов через CLI

`embed_texts.py` читает CSV с колонкой `text` и для каждого документа строит один вектор.

Запуск:

```bash
python -m bert_embeddings.embed_texts \
  --input-csv /path/test.csv \
  --output-npy /path/test_embs.npy \
  --output-csv /path/test_embs_manifest.csv \
  --model-dir /path/saved_models/bert_embeddings_best/best_model \
  --chunk-size 448 \
  --chunk-overlap 96 \
  --pooling mean_max \
  --chunk-aggregation mean_max \
  --batch-size 8
```

Обязательные параметры:

- `--input-csv` — путь к CSV c колонкой `text`;
- `--output-npy` — куда сохранить матрицу эмбеддингов `N × D`;
- `--model-dir` — директория с encoder‑весами (`best_model` или `final_model`).

Опционально можно переопределить любые поля `EmbeddingConfig`:

- `max_length` → `--max-length`
- `chunk_size` → `--chunk-size`
- `chunk_overlap` → `--chunk-overlap`
- `pooling` → `--pooling` (`mean`, `cls`, `max`, `mean_max`)
- `chunk_aggregation` → `--chunk-aggregation` (`mean`, `max`, `mean_max`)
- `batch_size` → `--batch-size`
- `base_model_name` → `--base-model-name` (если encoder обучался на другой базовой модели).[conversation_history:2]

При указании `--output-csv` дополнительно сохраняется манифест:

- `chunk_count` — сколько чанков на документ;
- `embedding_dim` — размерность эмбеддинга;
- `embedding_norm` — L2‑норма эмбеддинга.

## Встраивание эмбеддера из Python

```python
from bert_embeddings.embedding_model import LongTextRobertaEmbedder
from bert_embeddings.config import EmbeddingConfig

cfg = EmbeddingConfig(
    chunk_size=448,
    chunk_overlap=96,
    pooling="mean_max",
    chunk_aggregation="mean_max",
    batch_size=8,
)

embedder = LongTextRobertaEmbedder(
    model_dir="/path/saved_models/bert_embeddings_best/best_model",
    cfg=cfg,
)

texts = [
    "Первое длинное деловое письмо...",
    "Второе длинное письмо...",
]
embs, chunk_counts = embedder.encode(texts, return_chunk_counts=True)
```

`encode` возвращает `np.ndarray` формы `(N, hidden_size)` с L2‑нормированными эмбеддингами документов (и опционально количество чанков на документ).[conversation_history:2]

## Почему запуск через `python -m`

Папка оформлена как пакет с `__init__.py`, а все пути и импорты внутри написаны относительно `bert_embeddings`. Поэтому рекомендуемый способ запуска — из корня репозитория:

```bash
python -m bert_embeddings.mlm_train ...
python -m bert_embeddings.embed_texts ...
```

Это гарантирует корректную работу импортов и не требует ручного добавления пути в `PYTHONPATH` или `sys.path`.[conversation_history:2]