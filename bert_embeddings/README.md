# bert_embeddings

Пакет для обучения эмбеддингов длинных русскоязычных документов на базе `ai-forever/ruRoberta-large` с supervised fine-tuning по данным `text/label` и последующим построением document embeddings через token-based chunking + pooling + aggregation.

Главная цель пакета — обучить encoder для эмбеддингов, но сохранить совместимый выходной формат для внешних систем:

- `pytorch_model.bin` — PyTorch `state_dict`, сохранённый через `torch.save(...)`;
- ключи в `state_dict` начинаются с префикса `roberta.`;
- downstream-системы могут удалить префикс `roberta.` и загрузить веса в чистый `RobertaModel`.

Пакет сохраняет совместимый внешний интерфейс:

- Python API: `run_from_params(...)`
- CLI для обучения: `python -m bert_embeddings.mlm_train ...`
- CLI для генерации эмбеддингов: `python -m bert_embeddings.embed_texts ...`

---

## Структура пакета

- `__init__.py` — экспорт `LongTextRobertaEmbedder`
- `config.py` — датаклассы `MLMConfig` и `EmbeddingConfig`
- `data_utils.py` — загрузка `text/label`, построение training pairs, chunk-aware views для длинных текстов
- `main.py` — функция `run_from_params(...)` с логикой supervised обучения sentence embeddings
- `mlm_train.py` — CLI-обёртка для обучения (legacy entrypoint name)
- `embedding_model.py` — `LongTextRobertaEmbedder` для длинных документов
- `embed_texts.py` — CLI для генерации эмбеддингов по CSV

---

## Что делает пайплайн

1. Читает `train` и `test` CSV с колонками `text` и `label`.
2. Объединяет их в единый обучающий корпус.
3. Для длинных текстов строит chunk-aware training views, чтобы обучение не ограничивалось только первыми 512 токенами.
4. Строит supervised пары:
   - positive pairs — тексты с одинаковым `label`;
   - negative pairs — тексты с разными `label`.
5. Обучает sentence embedding model на базе `ai-forever/ruRoberta-large`.
6. Сохраняет:
   - `best_model/`
   - `final_model/`
   - `checkpoints/epoch_XXX/`
   - `metrics.json`
7. В каждую директорию модели сохраняет:
   - `pytorch_model.bin`
   - `config.json`
   - tokenizer files
   - `mlm_meta.json`
8. На этапе инференса строит эмбеддинги длинных документов через token-based chunking с overlap и document-level aggregation.

---

## Формат входных данных

Ожидаются CSV-файлы с колонками:

- `text` — текст документа / письма
- `label` — класс документа

Пример:

| text | label |
|------|-------|
| "Добрый день, направляем договор..." | contract |
| "Просим согласовать оплату..." | finance |
| "Повторно направляем коммерческое предложение..." | sales |

Пустые и `NaN`-значения удаляются автоматически.

---

## Почему лимит модели не поднимается выше 512

Базовая модель `ai-forever/ruRoberta-large` используется как обычный RoBERTa encoder, поэтому безопасный single-pass лимит остаётся около 512 токенов.[web:54][web:56]

Если письма длиннее:
- на обучении они разбиваются на training views;
- на инференсе они разбиваются на chunk'и с overlap;
- затем chunk embeddings агрегируются в один document embedding.[web:7][web:10]

Это сделано специально, чтобы:
- сохранить совместимость с `RobertaModel`;
- не ломать формат `pytorch_model.bin` с ключами `roberta.*`;
- не требовать отдельной long-context архитектуры.[web:56][web:63]

---

## Основные настройки

Все дефолты лежат в `config.py`.

### MLMConfig

Несмотря на имя, `MLMConfig` теперь управляет не MLM, а supervised обучением эмбеддингов. Имя оставлено ради обратной совместимости.

```python
@dataclass
class MLMConfig:
    model_name: str = "ai-forever/ruRoberta-large"
    text_col: str = "text"
    label_col: str = "label"

    max_length: int = 512

    train_batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 5
    warmup_ratio: float = 0.1
    logging_steps: int = 50
    fp16: bool = True
    seed: int = 42

    val_size: float = 0.1
    checkpoint_every_n_epochs: int = 1
    early_stopping_patience: int = 3

    max_pairs_per_label: int = 20000
    max_negative_pairs: int = 20000
    max_eval_pairs_per_label: int = 4000

    train_pair_strategy: str = "pair_class"
    train_loss: str = "softmax"

    train_chunk_size: int = 448
    train_chunk_overlap: int = 128
    add_global_chunk_to_training: bool = True

    sentence_pooling: str = "mean"
    save_sentence_transformer_artifacts: bool = True
```

Рекомендуемые стартовые значения для длинных деловых писем:

- `max_length=512`
- `train_chunk_size=448`
- `train_chunk_overlap=128`
- `sentence_pooling="mean"`
- `train_loss="softmax"`

### EmbeddingConfig

```python
@dataclass
class EmbeddingConfig:
    max_length: int = 512
    chunk_size: int = 448
    chunk_overlap: int = 128
    pooling: str = "mean_max"
    chunk_aggregation: str = "mean_max"
    batch_size: int = 8
    normalize_chunks: bool = True
    normalize_document: bool = True
    add_global_chunk: bool = True
    base_model_name: str = "ai-forever/ruRoberta-large"
```

Рекомендуемые значения для длинных писем:

- `chunk_size=448`
- `chunk_overlap=128`
- `pooling="mean_max"`
- `chunk_aggregation="mean_max"`
- `add_global_chunk=True`

---

## Установка

```bash
pip install torch transformers sentence-transformers datasets pandas numpy pyarrow scikit-learn
```

Нужны совместимые версии:
- `torch`
- `transformers`
- `sentence-transformers`
- `datasets`

Если обучение идёт на GPU, `fp16` включится автоматически, когда `torch.cuda.is_available()` возвращает `True`.

---

## Обучение из Python

Основной способ запуска из Python:

```python
import os

from bert_embeddings.main import run_from_params


TRAIN_FILE = os.path.join(drive_root, "train.csv")
TEST_FILE = os.path.join(drive_root, "test.csv")
SAVE_DIR = os.path.join(drive_root, "saved_models", "bert_embeddings_best")


components, eval_metrics = run_from_params(
    train_file=TRAIN_FILE,
    test_file=TEST_FILE,
    output_dir=SAVE_DIR,
    num_epochs=15,
    checkpoint_every_n_epochs=8,
)
```

Это **должно сработать**, если:
- `bert_embeddings/main.py` содержит `run_from_params`;
- входные CSV содержат колонки `text` и `label`;
- в окружении установлены зависимости.

### Что возвращает `run_from_params`

Функция возвращает:

- `components` — словарь с путями:
  - `output_dir`
  - `checkpoints_dir`
  - `best_model_dir`
  - `final_model_dir`
  - `metrics_path`
  - `model_name`
- `eval_metrics` — словарь с метриками валидации и служебной информацией.

---

## Обучение через CLI

### Рекомендуемый способ

Запускать из корня репозитория:

```bash
python -m bert_embeddings.mlm_train \
  --train-file ~/papadyk-vkr/data/train_augmented.csv \
  --test-file ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/bert-classification-out
```

Да, **этот способ должен сработать**, если пакет лежит в корректной структуре и запускается из директории, где доступен импорт `bert_embeddings`.

### Почему не `python -m bert_embeddings.main`

Команда:

```bash
python -m bert_embeddings.main \
  --train-file ~/papadyk-vkr/data/train_augmented.csv \
  --test-file ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/bert-classification-out
```

**не сработает в текущем варианте**, потому что `main.py` содержит Python API, но не содержит CLI-парсера аргументов и `__main__` entrypoint.

Если хочется запускать именно так, надо добавить в `bert_embeddings/main.py` функцию CLI-обёртки с `argparse`.

---

## Переопределение гиперпараметров в CLI

Любое небулевое поле `MLMConfig` можно переопределить из CLI через `snake_case -> kebab-case`.

Примеры:

- `num_train_epochs` → `--num-train-epochs`
- `train_batch_size` → `--train-batch-size`
- `eval_batch_size` → `--eval-batch-size`
- `learning_rate` → `--learning-rate`
- `warmup_ratio` → `--warmup-ratio`
- `val_size` → `--val-size`
- `checkpoint_every_n_epochs` → `--checkpoint-every-n-epochs`
- `early_stopping_patience` → `--early-stopping-patience`
- `max_pairs_per_label` → `--max-pairs-per-label`
- `max_negative_pairs` → `--max-negative-pairs`
- `train_chunk_size` → `--train-chunk-size`
- `train_chunk_overlap` → `--train-chunk-overlap`

Пример:

```bash
python -m bert_embeddings.mlm_train \
  --train-file ~/papadyk-vkr/data/train_augmented.csv \
  --test-file ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/bert-classification-out \
  --num-train-epochs 15 \
  --train-batch-size 16 \
  --eval-batch-size 16 \
  --learning-rate 2e-5 \
  --warmup-ratio 0.1 \
  --checkpoint-every-n-epochs 8 \
  --train-chunk-size 448 \
  --train-chunk-overlap 128
```

---

## Что получается в `output_dir`

После обучения в `output_dir` будут:

- `best_model/`
- `final_model/`
- `checkpoints/epoch_XXX/`
- `metrics.json`

### Содержимое `best_model/` и `final_model/`

- `pytorch_model.bin` — `state_dict`, сохранённый через `torch.save(...)`
- `config.json` — конфиг модели
- tokenizer files
- `mlm_meta.json`
- `sentence_transformer/` — дополнительные артефакты Sentence-Transformers (если `save_sentence_transformer_artifacts=True`)

### Важный контракт по `pytorch_model.bin`

`pytorch_model.bin` создаётся так, чтобы downstream мог использовать его как encoder-only checkpoint:

- веса сохранены через `torch.save(state_dict, ...)`;
- ключи начинаются с `roberta.`;
- downstream может удалить `roberta.` и загрузить веса в `RobertaModel`.

---

## Генерация эмбеддингов через CLI

`embed_texts.py` читает CSV с колонкой `text` и сохраняет по одному эмбеддингу на документ.

Пример:

```bash
python -m bert_embeddings.embed_texts \
  --input-csv ~/papadyk-vkr/data/test.csv \
  --output-npy ~/papadyk-vkr/data/test_embs.npy \
  --output-csv ~/papadyk-vkr/data/test_embs_manifest.csv \
  --model-dir ~/papadyk-vkr/bert-classification-out/best_model \
  --chunk-size 448 \
  --chunk-overlap 128 \
  --pooling mean_max \
  --chunk-aggregation mean_max \
  --batch-size 8
```

Обязательные параметры:

- `--input-csv`
- `--output-npy`
- `--model-dir`

Опционально:

- `--output-csv`
- любые поля `EmbeddingConfig`

### Что сохраняется

- `output_npy` — матрица эмбеддингов формы `N x D`
- `output_csv` — optional manifest:
  - `chunk_count`
  - `embedding_dim`
  - `embedding_norm`

---

## Использование из Python для эмбеддингов

```python
from bert_embeddings.embedding_model import LongTextRobertaEmbedder
from bert_embeddings.config import EmbeddingConfig


cfg = EmbeddingConfig(
    chunk_size=448,
    chunk_overlap=128,
    pooling="mean_max",
    chunk_aggregation="mean_max",
    batch_size=8,
)

embedder = LongTextRobertaEmbedder(
    model_dir="/path/to/saved_models/bert_embeddings_best/best_model",
    cfg=cfg,
)

texts = [
    "Первое длинное деловое письмо...",
    "Второе длинное письмо...",
]

embs, chunk_counts = embedder.encode(texts, return_chunk_counts=True)
```

`encode(...)` возвращает:
- `np.ndarray` формы `(N, hidden_size)` с document embeddings;
- опционально `chunk_counts`.

---

## Рекомендации для длинных писем

Для писем, которые часто длиннее 512 токенов, рекомендуется:

### На обучении

- `max_length=512`
- `train_chunk_size=448`
- `train_chunk_overlap=128`
- `add_global_chunk_to_training=True`

### На инференсе

- `chunk_size=448`
- `chunk_overlap=128`
- `pooling="mean_max"`
- `chunk_aggregation="mean_max"`
- `add_global_chunk=True`

### Почему это важно

Если письмо длинное, то без chunk-aware подхода модель увидит только его начало. В этом пакете:
- обучение использует training views от длинных документов;
- инференс использует sliding-window chunking;
- итоговый embedding агрегирует несколько представлений одного письма.[web:7][web:10]

---

## Ограничения

1. `max_length > 512` для `ai-forever/ruRoberta-large` не рекомендуется без отдельного context-extension пайплайна.
2. Если какой-то класс представлен 1–2 примерами, качество supervised pairs будет ограничено.
3. Текущий CLI-обучатель называется `mlm_train.py` только ради обратной совместимости; фактически он обучает sentence embeddings.
4. Команда `python -m bert_embeddings.main ...` не работает без отдельной CLI-обвязки.

---

## Частые проблемы

### 1. `KeyError: "['label'] not in index"`
Во входных CSV нет колонки `label` или она названа иначе.

Решение:
- либо переименовать колонку в `label`,
- либо передать `label_col` через конфиг / CLI.

### 2. `KeyError: "['text'] not in index"`
Во входных CSV нет колонки `text`.

Решение:
- переименовать колонку в `text`,
- либо передать `text_col`.

### 3. `Weights not found: .../pytorch_model.bin`
Указан неправильный `model_dir` для инференса.

Обычно нужно указывать:
- `.../best_model`
или
- `.../final_model`

### 4. `python -m bert_embeddings.main ...` не запускается
Это ожидаемо для текущей реализации.

Используй:
```bash
python -m bert_embeddings.mlm_train ...
```

---

## Минимальный рабочий сценарий

### Python training

```python
import os
from bert_embeddings.main import run_from_params

TRAIN_FILE = os.path.join(drive_root, "train.csv")
TEST_FILE = os.path.join(drive_root, "test.csv")
SAVE_DIR = os.path.join(drive_root, "saved_models", "bert_embeddings_best")

components, eval_metrics = run_from_params(
    train_file=TRAIN_FILE,
    test_file=TEST_FILE,
    output_dir=SAVE_DIR,
    num_epochs=15,
    checkpoint_every_n_epochs=8,
)
```

### CLI training

```bash
python -m bert_embeddings.mlm_train \
  --train-file ~/papadyk-vkr/data/train_augmented.csv \
  --test-file ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/bert-classification-out
```

### CLI embeddings

```bash
python -m bert_embeddings.embed_texts \
  --input-csv ~/papadyk-vkr/data/test.csv \
  --output-npy ~/papadyk-vkr/data/test_embs.npy \
  --model-dir ~/papadyk-vkr/bert-classification-out/best_model
```
