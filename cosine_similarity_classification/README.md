# cosine_similarity_classification

Классификация длинных текстов через косинусную близость в пространстве эмбеддингов от модуля `bert_embeddings`. Модуль не учит ничего нового: он берёт уже дообученный энкодер, считает векторы для train/test и применяет три геометрических классификатора — по центроидам классов, по ближайшим соседям и их линейный ансамбль. Все три метода работают в одном и том же L2-нормированном пространстве, поэтому скалярное произведение двух векторов равно их косинусной близости.

## Как это работает

### Постановка задачи

После `bert_embeddings` мы имеем энкодер, для которого справедливо: тексты одного класса лежат близко на единичной сфере, тексты разных классов — далеко. Это значит, что классификацию можно делать без дополнительной обучаемой головы: достаточно описать каждый класс геометрически (одной точкой — центроидом, или множеством точек — обучающей выборкой) и для нового текста выбрать класс, к которому он ближе по косинусу.

Преимущества такого подхода: нет переобучения новой головы, нет риска забыть энкодер, инференс — это одно матричное умножение `Q @ T.T`, где `Q` — матрица query-эмбеддингов, `T` — матрица train-эмбеддингов. Все эмбеддинги L2-нормированы, поэтому `Q @ T.T` напрямую содержит косинусы.

### Архитектура

```
train.csv ──► LongTextRobertaEmbedder ──► train_embs (N_tr, H), L2-норма
                                                │
                                                ▼
                                ┌──────────────────────────────┐
                                │   sims = Q @ T.T   (косинусы) │
                                └──────────────────────────────┘
                                                │
                ┌───────────────────────────────┼────────────────────────────────┐
                ▼                               ▼                                ▼
        Centroid                         Nearest (kNN)                    Centroid + NN
        (1 точка / класс)                (soft-voting top-k)              (линейный ансамбль)
                │                               │                                │
                └─────────► argmax по классам ◄─┘                                │
                                                                                 ▼
                                                                 alpha*S_cent + (1-alpha)*S_nn
test.csv ──► LongTextRobertaEmbedder ──► query_embs (N_te, H), L2-норма
```

Энкодер `LongTextRobertaEmbedder` импортируется из `bert_embeddings` с теми же параметрами, что использовались при обучении (`POOLING="mean"`, `CHUNK_AGGREGATION="mean"`, `MAX_LENGTH=512`, `CHUNK_SIZE=512`, `CHUNK_OVERLAP=128`) — иначе геометрия пространства будет другой и косинусы потеряют смысл.

### Centroid — классификация по центру масс класса

Центроид класса $c$ — это среднее всех его train-эмбеддингов, спроецированное обратно на единичную сферу:

$$\mu_c = \frac{1}{|X_c|} \sum_{x \in X_c} x, \quad \hat\mu_c = \frac{\mu_c}{\lVert \mu_c \rVert_2}$$

Для query-вектора $q$ предсказание:

$$\hat y(q) = \arg\max_c \; \langle q, \hat\mu_c \rangle$$

Тривиальный центроид чувствителен к выбросам: один аномальный текст внутри класса сдвигает $\hat\mu_c$ и портит соседей. Поэтому реализованы два режима триминга и опциональное уточнение.

**Hard-trim** — выбрасываем худшие $r$ долей по близости к текущему центроиду:

1. Считаем $s_i = \langle x_i, \hat\mu_c \rangle$ для всех $x_i \in X_c$.
2. Оставляем $n_\text{keep} = \lceil n \cdot (1-r) \rceil$ векторов с максимальным $s_i$ (через `np.argpartition(-sims, n_keep-1)` — O(n) вместо полной сортировки).
3. Пересчитываем $\hat\mu_c$ по оставшимся.

**Soft-trim** — вместо отбрасывания используем взвешенное среднее, где вес = степень близости:

$$w_i = \max(s_i, 0)^p$$

Дополнительно, если `trim_ratio > 0` и в классе хотя бы 3 примера, веса ниже $r$-перцентиля обнуляются — это убирает явные выбросы, но без жёсткого порога. Затем:

$$\hat\mu_c^{(\text{new})} = \text{normalize}\left(\frac{\sum_i w_i x_i}{\sum_i w_i}\right)$$

Параметр $p$ (`CENTROID_TRIM_POWER`) управляет резкостью: $p=2$ — мягкое взвешивание, $p=8$ — почти hard-trim.

**Refine-iterations**: после первого пересчёта центроид можно использовать как новую опорную точку и повторить триминг ещё раз. Реализована ранняя остановка:

$$\lVert \hat\mu_c^{(k+1)} - \hat\mu_c^{(k)} \rVert_2 < 10^{-6} \;\Rightarrow\; \text{break}$$

На практике 1-2 итераций достаточно — дальше центроид перестаёт двигаться. Слишком много итераций может схлопнуть класс к одной плотной подгруппе и ухудшить generalization.

### Nearest — голосование top-k соседей

Для каждого query берём $k$ ближайших train-векторов по косинусу, но голосуем не дискретно (`majority vote`), а взвешенно с температурой:

1. Top-k индексы: `top_idx = np.argpartition(-sims_q, k-1)[:k]` — O(N) операция.
2. Веса (softmax по косинусам соседей):

$$w_j = \frac{\exp(s_j / T)}{\sum_{j'=1}^{k} \exp(s_{j'} / T)}$$

3. Аккумулирование голосов по классам через `np.add.at(votes, class_idx[top_idx], w)` — векторизованная альтернатива циклу.
4. Предсказание: $\hat y = \arg\max_c \text{votes}_c$.

Температура $T$ — ключевой гиперпараметр:

- $T \to 0$: только ближайший сосед получает вес ≈1 (вырождается в 1-NN).
- $T \to \infty$: все $k$ соседей с равным весом (вырождается в majority vote).
- $T \in [0.05, 0.3]$ — компромисс: топовый сосед доминирует, но более дальние тоже корректируют.

В отличие от центроида, kNN устойчив к мультимодальным классам (когда класс — это две разные подгруппы текстов), потому что не пытается описать класс одной точкой.

### Centroid + NN — ансамбль

Centroid даёт оценку «класс-как-целое» (грубая, но устойчивая), kNN — оценку «локальный сосед» (точная, но чувствительная к шуму). Их можно линейно смешать, но напрямую складывать матрицы скоров нельзя: косинусы лежат в $[-1, 1]$, а softmax-голоса — в $[0, 1]$ и обычно сильно скошены. Поэтому реализована per-query min-max нормализация:

$$\tilde S_{q,c} = \frac{S_{q,c} - \min_c S_{q,c}}{\max_c S_{q,c} - \min_c S_{q,c} + \varepsilon}$$

Нормализация делается отдельно для centroid-матрицы и NN-матрицы, по каждой строке query. После этого скоры в одном масштабе $[0,1]$ и их можно смешать:

$$S_{\text{ens}} = \alpha \cdot \tilde S_{\text{cent}} + (1-\alpha) \cdot \tilde S_{\text{nn}}$$

$\alpha=0.5$ — равный вклад, $\alpha=0.7$ — приоритет центроида, $\alpha=0.3$ — приоритет соседей. Предсказание = `argmax` по столбцам $S_{\text{ens}}$.

### Sweep — поиск гиперпараметров на test-выборке

Все три метода имеют гиперпараметры (k, T, trim_ratio, trim_power, mode, alpha), и оптимальная комбинация не известна заранее. Модуль перебирает решётку значений и выбирает лучшее по `balanced_accuracy`.

Главное ускорение sweep — переиспользование матрицы косинусов:

$$\text{sims} = Q \cdot T^\top \in \mathbb{R}^{N_\text{te} \times N_\text{tr}}$$

Эта матрица считается один раз. Далее:

- **kNN sweep** ($|K| \times |T|$ комбинаций): для каждой пары $(k, T)$ берётся уже готовая `sims` — нужны только `argpartition` и softmax.
- **Centroid sweep** ($|mode| \times |ratio| \times |power|$): для hard-режима dimension `power` пропускается (он влияет только на soft-trim) — это убирает 1/2 избыточных запусков.
- **Ensemble sweep**: после нахождения лучшего centroid и лучшего nn, перебирается только $\alpha$.

В CLI решётки задаются как:

```
--centroid-mode-sweep "hard,soft"
--centroid-ratio-sweep "0,0.1,0.15,0.2"
--centroid-power-sweep "2,4,6,8"
--knn-k-sweep "1,3,5,7,9,11,15,21,31"
--knn-temperature-sweep "0.03,0.05,0.1,0.15,0.2,0.3"
--ensemble-alpha-sweep "0.3,0.4,0.5,0.6,0.7"
```

### Predict-only режим

Если test-CSV не содержит колонки `label`, флаг `_test_has_label_column` отключает evaluation: модуль сохраняет только предсказания (`label_pred` колонка), без метрик и без sweep. Полезно для финального применения уже отобранной конфигурации к unlabeled данным.

### Сохранение результатов

Результат — один JSON-файл со структурой:

```
{
  "method": "centroid_nn",
  "model_dir": "...",
  "best_config": {"alpha": 0.5, "k": 7, "T": 0.1, ...},
  "best_metrics": {"balanced_accuracy": ..., "macro_f1": ...},
  "sweep_results": [...],
  "predictions": [{"text_idx": ..., "label_pred": ...}, ...]
}
```

Имя файла формируется автоматически:

```
cosine-results-<model_tag>-<YYYY-MM-DDTHH:MM>.json
```

где `model_tag` — это `Path(model_dir).name` (например, `xlm-roberta-base-finetuned`), а timestamp — момент запуска. Файл кладётся в директорию train-CSV — это удобно: рядом с обучающими данными лежат и метрики моделей, которые их использовали.

## Структура модуля

| Файл | Содержимое |
| --- | --- |
| `config.py` | Все гиперпараметры (имена колонок, MODEL_DIR, KNN_K, TEMPERATURE, TRIM_*, ALPHA, sweep-решётки). |
| `embedder.py` | `load_texts_and_labels`, `build_embedder`, `embed_dataframe` — обёртки над `LongTextRobertaEmbedder`. |
| `classifiers.py` | Три классификатора + sweep-функции (`centroid_predict`, `knn_predict`, `centroid_nn_predict`). |
| `metrics.py` | `evaluate_predictions` — balanced_accuracy + macro_f1, подавление UndefinedMetricWarning. |
| `main.py` | CLI: парсинг аргументов, оркестрация эмбеддинг → классификация → sweep → JSON. |
| `__init__.py` | Реэкспорт публичных функций. |

## CLI

Минимальный запуск (использует все дефолты + перебирает sweep):

```bash
python -m cosine_similarity_classification.main \
    --train train.csv \
    --test test.csv \
    --method all
```

С переопределением модели и узкой настройкой:

```bash
python -m cosine_similarity_classification.main \
    --train data/train.csv \
    --test data/test.csv \
    --method centroid_nn \
    --model-dir /content/drive/MyDrive/vkr/models/xlm-roberta-base-finetuned \
    --centroid-mode soft \
    --centroid-trim-ratio 0.15 \
    --centroid-trim-power 4 \
    --knn-k 7 \
    --knn-temperature 0.1 \
    --ensemble-alpha 0.5
```

`--method` принимает `centroid`, `nearest`, `centroid_nn`, `all` — последнее запускает все три и сравнивает.

## Python API

```python
from cosine_similarity_classification import (
    load_texts_and_labels,
    build_embedder,
    embed_dataframe,
    centroid_predict,
    knn_predict,
    centroid_nn_predict,
    evaluate_predictions,
)

train_df = load_texts_and_labels("train.csv")
test_df = load_texts_and_labels("test.csv")

embedder = build_embedder(model_dir="path/to/finetuned")
train_embs = embed_dataframe(train_df, embedder)
test_embs = embed_dataframe(test_df, embedder)

y_pred, info = centroid_nn_predict(
    train_embs, train_df["label"].values,
    test_embs,
    k=7, temperature=0.1, alpha=0.5,
    trim_mode="soft", trim_ratio=0.15, trim_power=4.0,
)
print(evaluate_predictions(test_df["label"].values, y_pred))
```

## Входы

| Колонка | Тип | Обязательность | Комментарий |
| --- | --- | --- | --- |
| `text` | str | да | Длинный текст. Чанкуется внутри эмбеддера. |
| `label` | str | для train — да; для test — опционально | Если в test нет — режим predict-only. |

## Выходы

| Файл | Описание |
| --- | --- |
| `cosine-results-<tag>-<ts>.json` | Лучшая конфигурация, все sweep-результаты, предсказания. |

Файл сохраняется рядом с train-CSV (`train_file.parent`).

## Ключевые параметры

| Параметр | Дефолт | Назначение |
| --- | --- | --- |
| `MODEL_DIR` | путь к дообученному энкодеру | Откуда грузить веса для эмбеддинга. |
| `KNN_K` | 5 | Сколько соседей учитывать при kNN. |
| `KNN_TEMPERATURE` | 0.1 | Резкость softmax по соседям. |
| `CENTROID_TRIM_MODE` | "soft" | "hard" — отбрасывание, "soft" — взвешивание. |
| `CENTROID_TRIM_RATIO` | 0.15 | Доля худших примеров для отбрасывания / зануления весов. |
| `CENTROID_TRIM_POWER` | 4.0 | Степень для soft-trim весов. |
| `CENTROID_REFINE_ITERS` | 1 | Сколько раз пересчитывать центроид после триминга. |
| `ENSEMBLE_ALPHA` | 0.5 | Вес centroid в ансамбле centroid_nn. |
| `K_SWEEP` | "1,3,5,7,9,11,15,21,31" | Решётка $k$ для sweep. |
| `T_SWEEP` | "0.03,0.05,0.1,0.15,0.2,0.3" | Решётка температуры. |
| `ALPHA_SWEEP` | "0.3,0.4,0.5,0.6,0.7" | Решётка $\alpha$ ансамбля. |

## Метрики

На каждой комбинации sweep вычисляются:

- **Balanced accuracy** — среднее recall по классам, устойчиво к дисбалансу.
- **Macro F1** — невзвешенное среднее F1 по классам.

Лучшая конфигурация выбирается по **balanced accuracy** (это согласовано с метрикой отбора в `bert_classification`). Macro F1 сохраняется как контрольная.
