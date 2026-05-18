# cosine_similarity_classification

Самый простой downstream-классификатор: эмбеддинги документов от
`bert_embeddings` + два метода поверх косинусной близости — nearest
neighbours и centroid.

## Зачем

Это «честный» способ оценить качество выученных эмбеддингов:
если энкодер хорошо разделяет классы, то даже без обучаемой головы
по cosine-метрике должна быть высокая `balanced_accuracy`. На
`[custom_embeder]` после v3-правок центроидный метод даёт `0.564 / 0.499`
— заметно выше всех простых ноутбук-бэйзлайнов в `notebooks/bert/`.

## Методы

### `--method centroid`

1. Эмбеддим train и test через `LongTextRobertaEmbedder`.
2. По train для каждого класса считаем центроид: усреднение L2-норм.
   эмбеддингов всех его примеров.
3. Для каждого test-документа: косинусная близость к каждому центроиду
   → класс с максимальной близостью.

Сильные стороны: устойчив к выбросам, не требует обучения, быстрый.
Слабые: один вектор на класс — для мультимодальных классов хуже kNN.

### `--method nearest`

1. То же эмбеддение train+test.
2. Для каждого test: k ближайших соседей по cosine (`KNN_K`, по умолчанию 5).
3. Голосование с весом `1/(1-cos)` (weights="distance").

Сильные стороны: учитывает мультимодальные классы. Слабые: чувствителен
к шумным train-примерам.

## Что лежит

| Файл | Назначение |
| --- | --- |
| `config.py` | Все дефолты из `bert_embeddings.config.EmbeddingConfig` (single source of truth) |
| `embedder.py` | Загрузка CSV, обёртка над `LongTextRobertaEmbedder` |
| `classifiers.py` | `predict_centroid`, `predict_nearest` (kNN cosine) |
| `metrics.py` | `evaluate_predictions` — только `balanced_accuracy` + `macro_f1` |
| `main.py` | CLI + `run_from_params(...)` |

## Что пишется на диск

**Ничего.** Этот модуль только читает train/test, считает эмбеддинги
в памяти и печатает результат. Если нужно сохранить предикты —
используй возвращаемое `run_from_params`-ом `components["pred_labels"]`.

## Использование

### С базовой моделью

```bash
python -m cosine_similarity_classification.main \
    --train data/train.csv \
    --test data/test.csv \
    --method centroid
# centroid: {'balanced_accuracy': 0.124988, 'macro_f1': 0.094988}
```

### С `[custom_embeder]` после `bert_embeddings`

```bash
python -m cosine_similarity_classification.main \
    --train data/train.csv \
    --test data/test.csv \
    --model-dir /content/drive/.../bert_embeddings_best/best_model \
    --method centroid
# centroid: {'balanced_accuracy': 0.564518, 'macro_f1': 0.498997}
```

Или из ноутбука:

```python
from cosine_similarity_classification.main import run_from_params

components, metrics = run_from_params(
    train_file="data/train.csv",
    test_file="data/test.csv",
    model_dir="/content/drive/.../bert_embeddings_best/best_model",
    method="centroid",
)
print(metrics["balanced_accuracy"], metrics["macro_f1"])
```

## Формат вывода
```
centroid: {'balanced_accuracy': 0.564518, 'macro_f1': 0.498997}
nearest: {'balanced_accuracy': 0.476479, 'macro_f1': 0.475520}
```

Никаких per-class отчётов и `Precision is ill-defined`-предупреждений
больше нет — отчёт убран, `zero_division=0` поставлен везде.

## Pooling/chunk_aggregation

По умолчанию `pooling="mean"`, `chunk_aggregation="mean"` —
**синхронизировано** с обучением в `bert_embeddings`. Если когда-нибудь
переобучаешь энкодер с другим pooling (например, max), не забудь
поменять конфиг здесь же — иначе получишь catastrophic drop.