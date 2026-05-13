# hybrid

Пакет для экспериментов с **гибридными признаками** для классификации текстов.  
Гибридный вектор строится как конкатенация:

- sparse TF-IDF признаков по словам и символьным n-граммам;
- dense BERT-эмбеддингов длинных текстов;
- масштабированной BERT-части (масштаб задаётся одним из параметров конфигурации).

Пакет поддерживает три режима:

1. **build** — построить и сохранить hybrid vectors;
2. **classical** — обучить и оценить классические модели на готовых hybrid vectors;
3. **mlp** — обучить и оценить MLP на готовых hybrid vectors.

Запуск предполагается **из корня проекта** через:

```bash
python -m hybrid.main ...
```

---

## Структура

```text
hybrid/
├── __init__.py
├── config.py
├── main.py
├── hybrid_vector_build.py
├── hybrid_classical_models.py
└── hybrid_mlp.py
```

### `config.py`

Хранит dataclass-конфиги для CLI и сборки признаков:

- `HybridModelConfig` — параметры BERT-части и chunking/pooling;
- `HybridDataConfig` — имена колонок датасета;
- `HybridPathConfig` — пути к train/test/output.

Этот файл является **единым источником дефолтных значений** для режима `build`.

### `hybrid_vector_build.py`

Строит hybrid vectors:

- читает `train` и `test` CSV;
- чистит тексты и метки;
- строит word TF-IDF и char TF-IDF;
- получает BERT-эмбеддинги;
- масштабирует dense BERT-часть;
- объединяет sparse TF-IDF и dense BERT в один hybrid vector;
- сохраняет артефакты в `output_dir`.

Если передан `--model-dir`, используются эмбеддинги из fine-tuned модели через `LongTextRobertaEmbedder`.  
Если `--model-dir` не передан, используется базовая `ai-forever/ruRoberta-large`.

### `hybrid_classical_models.py`

Запускает классические модели на уже готовых hybrid vectors:

- `LinearSVC`
- `LogisticRegression`

Скрипт **ничего не сохраняет**, только печатает метрики:
- `balanced_accuracy`
- `macro_f1`

### `hybrid_mlp.py`

Запускает MLP на уже готовых hybrid vectors:

- residual MLP с `LayerNorm`, `GELU`, `Dropout`;
- `FocalLoss` с `class_weights` и `label_smoothing`;
- early stopping по `macro_f1`.

Скрипт **ничего не сохраняет**, только печатает лучший результат.

### `main.py`

Единая CLI-точка входа с subcommands:

- `build`
- `classical`
- `mlp`

Именно этот файл вызывается через:

```bash
python -m hybrid.main ...
```

`main.py` не хранит собственные дефолты для `build`, а собирает конфиги из `config.py` и переопределяет только те значения, которые были явно переданы через CLI.

---

## Что сохраняется

Пользовательский сценарий сделан так, чтобы **не сохранять лишние результаты**.

### При `build`

Сохраняются только артефакты, необходимые для следующих запусков:

- `X_train_hybrid.npz`
- `X_test_hybrid.npz`
- `y_train.csv`
- `y_test.csv`
- `word_tfidf.joblib`
- `char_tfidf.joblib`
- `scaler_bert.joblib`
- `meta.json`

### При `classical`

Ничего не сохраняется.  
Только печатаются метрики для `LinearSVC` и `LogisticRegression`.

### При `mlp`

Ничего не сохраняется.  
Только печатаются метрики по эпохам и лучший результат.

---

## Требования

Минимально нужны:

- `python >= 3.10`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `torch`
- `transformers`
- `joblib`

Если используется fine-tuned embedder через `--model-dir`, должен быть доступен модуль:

```python
from bert_embeddings.embedding_model import LongTextRobertaEmbedder
```

То есть пакет `bert_embeddings` должен быть доступен из текущего project root или из `PYTHONPATH`.

---

## Формат данных

Ожидаются CSV-файлы как минимум с колонками:

- `text`
- `label`

По умолчанию используются именно эти имена, но их можно переопределить через:

- `--text-col`
- `--label-col`

---

## Варианты запуска

## 1. Построение hybrid vectors

### Через fine-tuned embedder

```bash
python -m hybrid.main build \
  --train-file /Users/v.papadyk/ml/mifi-vkr/data/train_paraphrase_2.csv \
  --test-file /Users/v.papadyk/ml/mifi-vkr/data/test.csv \
  --output-dir /Users/v.papadyk/ml/mifi-vkr/hybrid/data/hybrid_vec \
  --model-dir /Users/v.papadyk/ml/mifi-vkr/bert_embeddings/best_model \
  --device cpu
```

Этот режим:

- строит TF-IDF признаки;
- получает dense embeddings через `LongTextRobertaEmbedder`;
- склеивает всё в hybrid vector;
- сохраняет результат в `output_dir`.

### Через базовую ruRoberta без `model_dir`

```bash
python -m hybrid.main build \
  --train-file /Users/v.papadyk/ml/mifi-vkr/data/train_paraphrase_2.csv \
  --test-file /Users/v.papadyk/ml/mifi-vkr/data/test.csv \
  --output-dir /Users/v.papadyk/ml/mifi-vkr/hybrid/data/hybrid_vec \
  --device cpu
```

В этом режиме будет использована базовая `ai-forever/ruRoberta-large`.

### С переопределением параметров chunking / pooling

```bash
python -m hybrid.main build \
  --train-file /Users/v.papadyk/ml/mifi-vkr/data/train_paraphrase_2.csv \
  --test-file /Users/v.papadyk/ml/mifi-vkr/data/test.csv \
  --output-dir /Users/v.papadyk/ml/mifi-vkr/hybrid/data/hybrid_vec \
  --model-dir /Users/v.papadyk/ml/mifi-vkr/bert_embeddings/best_model \
  --device cpu \
  --max-length 512 \
  --chunk-size 448 \
  --chunk-overlap 96 \
  --pooling mean_max \
  --chunk-aggregation mean_max \
  --batch-size 8
```

---

## 2. Classical models на готовых hybrid vectors

```bash
python -m hybrid.main classical \
  --vecdir /Users/v.papadyk/ml/mifi-vkr/hybrid/data/hybrid_vec
```

Будут обучены и оценены:

- `LinearSVC`
- `LogisticRegression`

Результат выводится в консоль, без сохранения `.joblib`, `.csv` и прочих файлов.

---

## 3. MLP на готовых hybrid vectors

```bash
python -m hybrid.main mlp \
  --vecdir /Users/v.papadyk/ml/mifi-vkr/hybrid/data/hybrid_vec \
  --epochs 40 \
  --device cpu
```

Скрипт:

- обучает MLP;
- печатает train loss и метрики по эпохам;
- останавливается по early stopping;
- в конце печатает лучший результат.

Никакие чекпоинты и промежуточные артефакты не сохраняются.

---

## Основные аргументы CLI

### `build`

Обязательные аргументы:

- `--train-file` — путь к train CSV
- `--test-file` — путь к test CSV
- `--output-dir` — директория, куда сохраняются hybrid vectors и служебные артефакты

Дополнительные аргументы:

- `--model-dir` — путь к fine-tuned модели для `LongTextRobertaEmbedder`
- `--device` — `cpu` или `cuda`
- `--base-model-name` — базовая HF-модель, по умолчанию `ai-forever/ruRoberta-large`
- `--text-col` — имя колонки с текстом
- `--label-col` — имя колонки с меткой
- `--max-length` — максимальная длина токенов
- `--chunk-size` — длина чанка для embedder
- `--chunk-overlap` — overlap между чанками
- `--pooling` — pooling на уровне токенов
- `--chunk-aggregation` — агрегация чанков
- `--batch-size` — batch size при построении dense embeddings

Если аргумент не передан, берётся значение из `config.py`.

### `classical`

- `--vecdir` — директория с уже построенными hybrid vectors

### `mlp`

- `--vecdir` — директория с уже построенными hybrid vectors
- `--epochs` — максимальное число эпох
- `--device` — `cpu` или `cuda`

---

## Пример полного пайплайна

### Шаг 1. Построить hybrid vectors

```bash
python -m hybrid.main build \
  --train-file /Users/v.papadyk/ml/mifi-vkr/data/train_augmented_3.csv \
  --test-file /Users/v.papadyk/ml/mifi-vkr/data/test.csv \
  --output-dir /Users/v.papadyk/ml/mifi-vkr/hybrid/data/hybrid_vec \
  --model-dir /Users/v.papadyk/ml/mifi-vkr/bert_embeddings/best_model \
  --device cpu
```

### Шаг 2. Запустить classical baselines

```bash
python -m hybrid.main classical \
  --vecdir /Users/v.papadyk/ml/mifi-vkr/hybrid/data/hybrid_vec
```

### Шаг 3. Запустить MLP

```bash
python -m hybrid.main mlp \
  --vecdir /Users/v.papadyk/ml/mifi-vkr/hybrid/data/hybrid_vec \
  --epochs 40 \
  --device cpu
```

---

## Примечания

- Запуск через `python -m hybrid.main` требует, чтобы команда выполнялась **из корня проекта**, где директория `hybrid/` видна как пакет Python.
- Если `bert_embeddings` лежит в том же проекте, запуск из корня обычно достаточен, чтобы импорт `bert_embeddings.embedding_model` работал корректно.
- Если нужен запуск напрямую как пакета без указания `main`, можно дополнительно добавить `hybrid/__main__.py`, но для текущего сценария это не обязательно.

---

## Идея пайплайна

Этот пакет удобен для быстрых сравнений трёх подходов на одном и том же наборе признаков:

- sparse + dense hybrid vector как общая feature base;
- linear baselines для быстрой проверки качества;
- MLP как более сильный non-linear baseline.

Такой подход позволяет быстро понять:
- хватает ли линейной модели;
- даёт ли dense-компонента прирост;
- имеет ли смысл переходить к более сложной MLP-голове.