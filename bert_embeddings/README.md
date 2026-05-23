# bert_embeddings

Модуль доменного дообучения sentence-encoder на базе
`ai-forever/ruRoberta-large` для русскоязычных писем и последующего
построения эмбеддингов длинных документов. Энкодер обучается
контрастивно на парах (sentence1, sentence2), а на инференсе документ
любой длины кодируется через chunk-aware пайплайн с агрегацией.

---

## Как это работает

### Постановка задачи

Базовая ruRoberta-large обучалась как MLM на общедомённом корпусе. Для
прикладной задачи (классификация писем по теме / категории) её сырые
эмбеддинги недостаточно специализированы: косинусное расстояние между
письмами одного класса и писем разных классов плохо разделены. Нужна
доменная подгонка — но без меток на инференсе. Решение — contrastive
fine-tune: «сближаем» представления документов одного класса и
«разводим» представления документов разных классов в эмбеддинг-пространстве,
после чего эмбеддингами можно пользоваться как готовыми признаками
(косинусный классификатор, MLP-голова, гибридные модели).

При этом надо учитывать, что:

- документы заметно длиннее 512 токенов; их нельзя обучать и кодировать
  «как есть»;
- меток внутри класса много, но они не упорядочены — задача похожа на
  metric learning, а не на classification;
- pooling и нормализация при обучении и при инференсе должны совпадать,
  иначе семантика эмбеддингов несовместима.

### Архитектура

```
                    обучение                              инференс
              (build_train_and_eval_pairs)           (LongTextRobertaEmbedder)

  CSV (text, label)                                  длинный текст
        │                                                 │
        ▼                                                 ▼
  чанкование на views                              чанкование с [CLS]/[SEP]
  (chunk_size=448,                                 (chunk_size=448,
   overlap=128,                                     overlap=128,
   + global head+tail)                              + global head+tail)
        │                                                 │
        ▼                                                 ▼
  positive-пары                                    батч чанков
  cross-document one-label                                │
  + negative random other-label                           ▼
        │                                          ruRoberta-large
        ▼                                          (fine-tuned encoder)
  SentenceTransformer                                     │
  Transformer → Pooling("mean") → Normalize               ▼
        │                                          token mean-pool по маске
        ▼                                                 │
  loss: MNR / CoSENT / Softmax                            ▼
        │                                          L2-normalize chunk
        ▼                                                 │
  best_model/  (только энкодер)                           ▼
        │                                          mean / max / mean_max
        ▼                                          по чанкам документа
  resume_checkpoint.pt                                    │
                                                          ▼
                                                   L2-normalize doc
                                                          │
                                                          ▼
                                                       (N, hidden)
```

### Подготовка обучающих данных

Из `train.csv` строится трёхступенчатый пайплайн:

1. **`build_training_dataframe`** — читает CSV, оставляет колонки
   `text/label`, удаляет пустые строки. Test-выборка сознательно не
   используется: дообучение на тестовых текстах привело бы к утечке
   train↔test и завышению метрик в downstream-классификаторе.
2. **`explode_long_texts_for_training`** — каждому документу
   присваивается `_doc_id`, после чего документ режется на
   overlapping-чанки в пространстве токенов (без спецтокенов, так как
   `add_special_tokens=False`):
   - `chunk_size = 448` токенов, `chunk_overlap = 128`,
     `stride = chunk_size − chunk_overlap = 320`;
   - если документ длиннее `chunk_size`, добавляется «глобальный» чанк
     `head + tail` (первые `chunk_size/2` токенов + последние
     `chunk_size − head`) — он даёт энкодеру одновременно начало и
     конец длинного письма за один проход;
   - дубликаты чанков, возникающие на коротких текстах, отбрасываются.
3. **`build_pair_dataframe`** — строит датасет пар. Для каждого класса
   среди его чанков перебираются комбинации `(i, j)` и сохраняются
   только те, у которых `_doc_id` различаются (`cross_document_positives_only=True`).
   Это исключает «тривиальные» позитивы из соседних overlapping-чанков
   одного письма, на которых contrastive loss моментально сходится к
   нулю без полезного сигнала. Negative-пары семплируются случайно
   между разными классами. Для балансировки используется `effective_cap
   = median(per_class_caps)` — медиана допустимых лимитов по классам,
   чтобы крупные классы не доминировали в loss.

Итоговая колонка `score = 1.0` для positive-пар и `0.0` для
negative-пар (используется CoSENT), `label = 1/0` — для бинарного
`BinaryClassificationEvaluator`.

### Доменное дообучение

Модель собирается стандартным `sentence-transformers`:

```
Transformer(ruRoberta-large, max_seq_length=512)
    → Pooling(mode="mean")
    → Normalize()
```

`mean`-pooling и `Normalize` (L2) ставятся осознанно: те же операции
применяются на инференсе в `LongTextRobertaEmbedder`. Несогласованность
этих двух мест (например, обучали с `mean`, а кодируем с `cls`) ломает
геометрию пространства и сводит на нет эффект fine-tune. Параметр
`MLMConfig.sentence_pooling` и `EmbeddingConfig.pooling` должны иметь
одинаковое значение.

### Функции потерь

Поддерживаются три варианта (`MLMConfig.train_loss`):

**MNR — `MultipleNegativesRankingLoss` (по умолчанию).** Использует
только positive-пары $(a_i, p_i)$. Остальные позитивы того же батча
служат in-batch negatives. Цель — максимизировать softmax
по строке матрицы косинусных похожестей:

$$
\mathcal{L}_{\text{MNR}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp\bigl(\text{sim}(a_i, p_i)/\tau\bigr)}{\sum_{j=1}^{N}\exp\bigl(\text{sim}(a_i, p_j)/\tau\bigr)}.
$$

Преимущество: не требует явных negative-пар (на каждый положительный
автоматически приходится `batch_size − 1` отрицательных в одном forward),
а качество растёт с ростом батча. Этот loss используется по умолчанию.

**CoSENT.** Обучается по непрерывным score: пара со score 1.0 должна
иметь косинус выше, чем пара со score 0.0, разница оптимизируется через
логистическую функцию (стабильнее, чем прямой MSE по косинусу).

**Softmax.** Бинарная классификация пар поверх объединённых эмбеддингов
$(u, v, |u−v|)$. Менее метрический, но иногда полезен как baseline.

### Подсказки для оптимизатора

- **`FreezeLowerLayersCallback`** — на первой эпохе замораживает
  `freeze_lower_layers = 12` нижних энкодер-слоёв (из 24) и слой
  эмбеддингов. Это снижает риск разрушить предобученное лексическое
  представление в самом начале обучения, когда градиенты ещё «дикие»
  из-за случайно инициализированных pooling/normalize-голов. Со второй
  эпохи все слои размораживаются.
- **AdamW + warmup_ratio = 0.1**, cosine LR scheduler.
  `learning_rate = 2e-5` — стандартный для дообучения BERT-моделей.
- **bf16 / fp16 / tf32**: bf16 по умолчанию (только если есть CUDA);
  при отсутствии CUDA bf16/fp16 принудительно сбрасываются в `False`
  в `run_from_params`.
- **`gradient_checkpointing=True`** — стандартная экономия VRAM ценой
  ~20–30% скорости.

### Inference: chunk-aware эмбеддинг документа

`LongTextRobertaEmbedder` загружает дообученный энкодер из
`best_model/pytorch_model.bin`. При загрузке снимается префикс `roberta.`,
с которым веса экспортированы из `SentenceTransformer`, и
загружаются в чистый `RobertaModel` через `strict=False`.

Кодирование одного документа:

1. Токенизация без спецтокенов → `token_ids` любой длины.
2. Sliding window: `stride = chunk_size − chunk_overlap = 320`,
   каждый чанк обрамляется `[CLS] ... [SEP]`. Если документ длиннее
   `chunk_size`, добавляется «глобальный» head+tail чанк.
3. Батч из чанков паддится до `max_len = max(len(c))`, прогоняется
   через энкодер.
4. **Token-level pooling** одного чанка: mean / max / mean_max или CLS.
   По умолчанию `mean` — он лучше согласован с обучением:

   $$
   e_c = \frac{\sum_{t} m_t H_{c,t}}{\max(\sum_t m_t, \varepsilon)}.
   $$
5. **L2-нормализация чанка** (`normalize_chunks=True`) — приводит каждый
   эмбеддинг чанка на единичную сферу. Это означает, что последующая
   агрегация неявно усредняет направления, а не длины.
6. **Chunk-level агрегация** по эмбеддингам чанков документа: mean / max
   / mean_max. По умолчанию — `mean`.
7. **L2-нормализация документа** (`normalize_document=True`) — итоговый
   $v_{doc}$ лежит на единичной сфере, поэтому косинусное расстояние
   эквивалентно скалярному произведению, а скейл документа не зависит
   от его длины.

Если документ пустой (после очистки токенизатор вернул пустой список) —
возвращается нулевой эмбеддинг размерности `hidden_size`. Это не
повреждает CSV/`.npy`-выход и позволяет downstream-скриптам аккуратно
обработать такие документы.

### Сохранение результатов

- `best_model/` — папка в формате HuggingFace. Содержит
  `pytorch_model.bin` (веса энкодера с префиксом `roberta.`),
  `config.json`, файлы токенизатора и `mlm_meta.json` (полная
  конфигурация обучения и статистика по данным).
- `resume_checkpoint.pt` — единый rolling-чекпоинт с состоянием
  энкодера, оптимизатора и scheduler-а. Перезаписывается in-place
  каждую эпоху, чтобы Drive не отправлял старые версии в корзину
  (rename = delete+create на Drive-маунтах).
- `metrics.json` — полный snapshot конфигурации + метрики evaluator-а
  на валидации.
- Промежуточные чекпоинты HF Trainer-а (порядка 1.4 GB на эпоху)
  пишутся в системный `tempfile.mkdtemp(...)` и удаляются после
  завершения обучения — на Drive они не попадают.

---

## Структура модуля

| Файл                  | Содержимое                                                                |
|-----------------------|---------------------------------------------------------------------------|
| `config.py`           | `MLMConfig` (обучение), `EmbeddingConfig` (инференс)                      |
| `data_utils.py`       | загрузка CSV, чанкование на views, построение cross-document positive-пар |
| `embedding_model.py`  | `LongTextRobertaEmbedder`, `mean_pooling`, `masked_max_pooling`           |
| `main.py`             | `run_from_params`, CLI обучения, коллбеки (freeze, rolling resume)        |
| `embed_texts.py`      | CLI инференса: CSV → `.npy` матрица эмбеддингов                           |

---

## CLI

### Обучение энкодера

```bash
python -m bert_embeddings.main \
    --train-file data/train.csv \
    --output-dir runs/bert_embeddings
```

С переопределением гиперпараметров (любое не-bool поле `MLMConfig`):

```bash
python -m bert_embeddings.main \
    --train-file data/train.csv \
    --output-dir runs/bert_embeddings \
    --num-train-epochs 3 \
    --train-batch-size 32 \
    --learning-rate 2e-5 \
    --train-loss mnr \
    --max-pairs-per-label 5000 \
    --freeze-lower-layers 12
```

### Инференс эмбеддингов

```bash
python -m bert_embeddings.embed_texts \
    --input-csv data/test.csv \
    --output-npy runs/bert_embeddings/test_emb.npy \
    --output-csv runs/bert_embeddings/test_emb_manifest.csv \
    --model-dir runs/bert_embeddings/best_model
```

`--output-csv` опциональный: пишет CSV-манифест с `chunk_count`,
`embedding_dim`, `embedding_norm` на каждую строку.

---

## Python API

### Обучение

```python
from bert_embeddings.main import run_from_params

components, metrics = run_from_params(
    train_file="data/train.csv",
    output_dir="runs/bert_embeddings",
    num_train_epochs=3,
    train_loss="mnr",
    cross_document_positives_only=True,
)
print(components["best_model_dir"], metrics)
```

`run_from_params` принимает overrides для любых полей `MLMConfig`;
поддерживаются алиасы `num_epochs → num_train_epochs` и
`batch_size → train_batch_size` для совместимости со старым API.

### Инференс

```python
from bert_embeddings.config import EmbeddingConfig
from bert_embeddings.embedding_model import LongTextRobertaEmbedder

embedder = LongTextRobertaEmbedder(
    model_dir="runs/bert_embeddings/best_model",
    cfg=EmbeddingConfig(pooling="mean", chunk_aggregation="mean"),
)

embs, chunk_counts = embedder.encode(
    ["длинный текст 1", "длинный текст 2"],
    return_chunk_counts=True,
)
# embs.shape == (2, hidden_size); embs нормализованы по L2.
```

`pooling` в `EmbeddingConfig` должен совпадать с
`MLMConfig.sentence_pooling`, который использовался при обучении —
иначе эмбеддинги несовместимы с дообученным энкодером.

---

## Входы

### Обучение

CSV с колонками `text/label` (имена настраиваются через
`MLMConfig.text_col` / `label_col`). Пустые `text` отбрасываются,
метки приводятся к строке.

### Инференс

CSV с колонкой `text` (пустые строки отбрасываются).

---

## Выходы

| Путь                                     | Содержимое                                                                 |
|------------------------------------------|----------------------------------------------------------------------------|
| `<output_dir>/best_model/`               | `pytorch_model.bin`, `config.json`, токенизатор, `mlm_meta.json`           |
| `<output_dir>/resume_checkpoint.pt`      | веса энкодера + state оптимизатора + state scheduler-а (rolling, in-place) |
| `<output_dir>/metrics.json`              | полная конфигурация обучения + метрики evaluator-а на валидации            |
| `--output-npy` (инференс)                | матрица эмбеддингов `(N, hidden)`, dtype=`float32`, L2-нормализованная     |
| `--output-csv` (инференс, опционально)   | манифест: `text`, `chunk_count`, `embedding_dim`, `embedding_norm`         |

---

## Ключевые параметры

`MLMConfig` (`config.py`):

| Параметр                          | По умолчанию               | Смысл                                                                  |
|-----------------------------------|----------------------------|------------------------------------------------------------------------|
| `model_name`                      | `ai-forever/ruRoberta-large` | базовая предобученная модель                                            |
| `max_length`                      | 512                        | длина окна для энкодера                                                |
| `train_chunk_size`                | 448                        | длина чанка при подготовке views (≤ max_length − 2)                    |
| `train_chunk_overlap`             | 128                        | перекрытие соседних чанков                                             |
| `add_global_chunk_to_training`    | True                       | добавлять «глобальный» head+tail чанк                                  |
| `cross_document_positives_only`   | True                       | строить positive-пары только между разными документами                 |
| `max_pairs_per_label`             | 5000                       | потолок positive-пар на класс (балансируется медианой)                 |
| `max_negative_pairs`              | 5000                       | потолок случайных negative-пар                                         |
| `train_loss`                      | `mnr`                      | `mnr` / `cosent` / `softmax`                                           |
| `sentence_pooling`                | `mean`                     | pooling в `SentenceTransformer` (должен совпасть с инференсом)         |
| `learning_rate`                   | 2e-5                       | LR AdamW                                                               |
| `num_train_epochs`                | 3                          | максимум эпох                                                          |
| `early_stopping_patience`         | 3                          | эпох без улучшения метрики                                             |
| `freeze_lower_layers`             | 12                         | нижние слои энкодера, замороженные на первой эпохе                     |
| `bf16` / `fp16` / `tf32`          | True / False / True        | mixed precision (CUDA-only; на CPU сбрасываются в False)               |
| `gradient_checkpointing`          | True                       | экономия VRAM ценой ~20–30% скорости                                   |

`EmbeddingConfig` (`config.py`):

| Параметр              | По умолчанию | Смысл                                                                |
|-----------------------|--------------|----------------------------------------------------------------------|
| `max_length`          | 512          | длина окна энкодера                                                  |
| `chunk_size`          | 448          | длина чанка (без спецтокенов)                                        |
| `chunk_overlap`       | 128          | перекрытие чанков                                                    |
| `pooling`             | `mean`       | token-level pooling (`mean` / `max` / `mean_max` / `cls`)            |
| `chunk_aggregation`   | `mean`       | document-level агрегация эмбеддингов чанков                          |
| `normalize_chunks`    | True         | L2-нормализация каждого чанка                                        |
| `normalize_document`  | True         | L2-нормализация итогового эмбеддинга документа                       |
| `add_global_chunk`    | True         | добавлять «глобальный» head+tail чанк                                |
| `batch_size`          | 8            | размер батча чанков на forward                                       |

---

## Метрики

При `train_loss = mnr` или `softmax` evaluator —
`BinaryClassificationEvaluator`. Лучшая эпоха выбирается по
`eval_valid-binary_cosine_ap` (average precision на бинарной задаче
«одна пара / разные пары» в косинусной метрике).

При `train_loss = cosent` evaluator —
`EmbeddingSimilarityEvaluator`. Лучшая эпоха выбирается по
`eval_valid-sim_spearman_cosine` (корреляция Спирмена между предсказанным
косинусом и эталонным score).

Все метрики сохраняются в `metrics.json` вместе со снимком конфигурации
и статистикой по данным (`raw_doc_count`, `train_view_count`,
`pair_count`, `train_pair_count`, `valid_pair_count`).
