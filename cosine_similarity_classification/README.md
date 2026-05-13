# cosine_similarity_classification

Модуль для классификации текстов через эмбеддинги и косинусное сходство.

Он работает в два этапа:
1. Каждый текст преобразуется в вектор с помощью `LongTextRobertaEmbedder`.
2. Для тестовых объектов выбирается либо ближайший обучающий пример (`nearest`), либо ближайший центроид класса (`centroid`).

## Структура модуля

- `config.py` — значения по умолчанию для модели, колонок и параметров чанкинга.
- `embedder.py` — загрузка данных и построение эмбеддера.
- `classifiers.py` — классификация по ближайшему соседу и по центроидам.
- `metrics.py` — расчет `balanced_accuracy`, `macro_f1` и `classification_report`.
- `main.py` — основной pipeline и CLI-запуск.

## Как это работает

### 1. Загрузка данных

Из train- и test-файлов читаются колонки:
- `text`
- `label` (для train обязательно, для test — только если нужна оценка качества)

Пустые тексты удаляются. Если в `test` есть колонка `label`, модуль считает метрики качества. Если `label` нет, он просто строит предсказания.

### 2. Построение эмбеддингов

Используется `LongTextRobertaEmbedder` из `bert_embeddings.embedding_model`.

Для длинных текстов применяется чанкинг:
- текст разбивается на куски длиной `chunk_size`;
- соседние куски могут пересекаться через `chunk_overlap`;
- каждый chunk кодируется моделью `RobertaModel`;
- chunk-эмбеддинги агрегируются в один эмбеддинг документа.

Поддерживаются варианты pooling внутри chunk:
- `mean`
- `cls`
- `max`
- `mean_max`

Поддерживаются варианты агрегации между chunk:
- `mean`
- `max`
- `mean_max`

### 3. Классификация

Есть два режима:

#### `nearest`
Для каждого тестового текста ищется обучающий пример с максимальным cosine similarity. Его метка становится предсказанием.

#### `centroid`
Для каждого класса считается средний эмбеддинг класса (centroid). Для тестового текста выбирается класс с максимальным cosine similarity до центроида.

### 4. Оценка качества

Если в `test` есть колонка `label`, считаются:
- `balanced_accuracy`
- `macro_f1`
- `classification_report`

## Использование base model и локальных весов

Если `--model-dir` **не передан**, эмбеддер использует только базовую модель из `base_model_name`.

Если `--model-dir` **передан**, эмбеддер дополнительно загружает локальные fine-tuned веса из:

```text
<model_dir>/pytorch_model.bin
```

Это удобно для двух режимов работы:
- быстрый запуск на базовой модели;
- запуск на локально дообученных весах.

## Формат входных данных

Ожидаются CSV-файлы.

### Train

Обязательные колонки:
- `text`
- `label`

### Test

Минимально:
- `text`

Для расчета метрик:
- `text`
- `label`

## CLI-запуск

Запуск на базовой модели:

```bash
python -m cosine_similarity_classification.main \
  --train ~/papadyk-vkr/data/train_augmented.csv \
  --test ~/papadyk-vkr/data/test.csv \
  --device cpu
```

Запуск с локальными fine-tuned весами:

```bash
python -m cosine_similarity_classification.main \
  --train ~/papadyk-vkr/data/train_augmented.csv \
  --test ~/papadyk-vkr/data/test.csv \
  --model-dir ~/papadyk-vkr/bert_embeddings/best_model \
  --device cpu
```

## Основные аргументы CLI

| Аргумент | Описание | Пример |
|---|---|---|
| `--train` | Путь к train CSV | `~/papadyk-vkr/data/train_augmented.csv` |
| `--test` | Путь к test CSV | `~/papadyk-vkr/data/test.csv` |
| `--model-dir` | Путь к локальным весам модели | `~/papadyk-vkr/bert_embeddings/best_model` |
| `--device` | Устройство для инференса | `cpu`, `cuda`, `mps` |
| `--method` | Способ классификации | `nearest`, `centroid` |
| `--base-model-name` | Базовая HF-модель | `ai-forever/ruRoberta-large` |
| `--chunk-size` | Размер текстового chunk | `448` |
| `--chunk-overlap` | Перекрытие между chunk | `96` |
| `--pooling` | Pooling внутри chunk | `mean`, `cls`, `max`, `mean_max` |
| `--chunk-aggregation` | Агрегация chunk-эмбеддингов | `mean`, `max`, `mean_max` |
| `--batch-size` | Batch size при кодировании | `8` |

## Пример программного вызова

```python
from cosine_similarity_classification.main import run_from_params

components, metrics = run_from_params(
    train_file="/Users/v.papadyk/ml/mifi-vkr/data/train_augmented.csv",
    test_file="/Users/v.papadyk/ml/mifi-vkr/data/test.csv",
    device="cpu",
)
```

## Что возвращает `run_from_params`

### `components`
Словарь с промежуточными результатами:
- `train_embeddings_shape`
- `test_embeddings_shape`
- `pred_labels`
- `pred_scores`
- `num_train`
- `num_test`

### `metrics`
Словарь с итоговой информацией о запуске:
- `method`
- `train_size`
- `test_size`
- `model_dir`
- `base_model_name`
- `chunk_size`
- `chunk_overlap`
- `pooling`
- `chunk_aggregation`
- `batch_size`
- `device`
- `balanced_accuracy`
- `macro_f1`
- `classification_report`

## Когда использовать какой метод

### `nearest`
Подходит, если:
- важны локальные примеры из train;
- классы сложные и плохо описываются одним средним вектором;
- нужен простой nearest-neighbor baseline.

### `centroid`
Подходит, если:
- нужен более стабильный и быстрый baseline;
- внутри класса тексты достаточно однородны;
- хочется меньше зависеть от отдельных шумных train-примеров.

## Частые проблемы

### Ошибка `Weights not found: pytorch_model.bin`

Причина: передан `--model-dir`, но в указанной директории нет файла `pytorch_model.bin`.

Проверь:
- путь к директории;
- наличие файла `pytorch_model.bin`;
- если локальные веса не нужны — просто не передавай `--model-dir`.

### Нет метрик качества

Причина: в test-файле нет колонки `label`.

В таком случае модуль делает предсказания, но не считает `balanced_accuracy` и `macro_f1`.

### Медленная работа

Возможные причины:
- длинные тексты;
- маленький `batch_size`;
- запуск на CPU вместо GPU.

## Кратко

`cosine_similarity_classification` — это baseline / lightweight classifier поверх эмбеддингов, который не требует отдельного обучения классификационной головы. Он хорошо подходит для быстрых экспериментов, сравнения представлений текста и проверки качества embedding-based подхода.