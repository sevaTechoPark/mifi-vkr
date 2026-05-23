# bert_classification

Модуль обучения и оценки классификатора длинных русскоязычных текстов поверх
предобученной модели `ai-forever/ruRoberta-large`. Документ длиннее 512 токенов
разбивается на чанки, эмбеддинги чанков агрегируются, а итоговый вектор
документа подаётся в линейную голову классификации.

---

## Как это работает

### Постановка задачи

Задача — многоклассовая классификация русскоязычных документов
произвольной длины. Базовые BERT-подобные энкодеры имеют жёсткое
ограничение длины входа (для ruRoberta-large — 512 токенов вместе со
спец-токенами `<s>` и `</s>`). Документы заметно длиннее не помещаются в
один проход; распространённый приём — обрезать вход — теряет информацию
из конца документа.

Здесь применяется **chunk-aware mean pooling**: документ делится на
перекрывающиеся окна, каждое окно проходит через энкодер отдельно, а
эмбеддинги окон агрегируются в один вектор документа. Такой подход:

- сохраняет контекст со всех частей документа;
- не требует менять архитектуру предобученной модели и токенизатор;
- стоимость инференса по документу растёт линейно по числу чанков.

### Архитектура

```
        текст документа
              │
              ▼
    токенизация + чанкование
       (max_length=512,
        stride=128,
        max_chunks=6)
              │
              ▼
   ┌─────────────────────────┐
   │   ruRoberta-large       │   ← общий для всех чанков
   │   (encoder, shared)     │
   └─────────────────────────┘
              │
              ▼
   token mean-pool по маске   ← эмбеддинг одного чанка
              │
              ▼
   chunk mean-pool по числу
   валидных чанков документа  ← эмбеддинг документа
              │
              ▼
   LayerNorm → Dropout → Linear(num_labels)
              │
              ▼
            logits
```

### Чанкование документа (sliding window)

Используется sliding window от HuggingFace-токенизатора:
`truncation=True`, `padding="max_length"`, `stride=128`,
`return_overflowing_tokens=True`. Параметры:

- `max_length = 512` — длина одного окна в токенах;
- `stride = 128` — длина перекрытия с предыдущим окном; шаг между
  началами соседних окон равен `max_length − stride = 384` токена;
- `max_chunks = 6` — максимум окон на документ; остальное усекается
  без чтения.

При длине документа L токенов количество чанков считается как
`ceil((L − stride) / (max_length − stride))`. При значениях по умолчанию
один документ покрывает до `max_length + (max_chunks−1)·(max_length−stride) =
512 + 5·384 = 2432` токена. Перекрытие в 128 токенов между соседними
чанками страхует от ситуации, когда важный фрагмент попадает ровно на
границу окна и оба соседних чанка видят его только частично.

Если документ короче — недостающие чанки добиваются паддингом
(`pad_token_id`), а в поле `num_chunks` сохраняется реальное число
валидных окон. Это позволяет держать все семплы в батче одинаковой формы
`(batch, max_chunks, max_length)` без специального data collator-а
переменной длины, при этом «пустые» чанки не вносят вклад в финальный
эмбеддинг документа.

### Двухуровневое усреднение (mean pooling)

В модели выполняется два независимых усреднения с маскированием — это
ключевое отличие от наивного «middle CLS» или max-pool.

**1. Token mean-pool внутри чанка.** Для чанка $c$ с последовательностью
скрытых состояний $H_c \in \mathbb{R}^{T \times d}$ и маской паддинга
$m_c \in \{0,1\}^{T}$ эмбеддинг чанка:

$$
e_c = \frac{\sum_{t=1}^{T} m_{c,t}\, H_{c,t}}{\max(\sum_{t=1}^{T} m_{c,t},\, \varepsilon)}.
$$

Maskedmean устойчивее к длине входа, чем `[CLS]` (особенно для
RoBERTa, где `[CLS]` не предобучается под NSP) и чем max-pool (max-pool
агрессивно реагирует на отдельные «острые» токены).

**2. Chunk mean-pool между чанками.** Имея эмбеддинги чанков
$\{e_1,\dots,e_K\}$ и маску валидных чанков $\mu_k = \mathbb{1}[k < \text{num\_chunks}]$:

$$
v_{doc} = \frac{\sum_{k=1}^{K} \mu_k\, e_k}{\max(\sum_{k=1}^{K} \mu_k,\, \varepsilon)}.
$$

$\varepsilon$ нужен, чтобы при чудом возникшем пустом документе деление не
дало NaN; на практике маска всегда содержит хотя бы одну единицу.
Клампинг знаменателя сделан через `clamp(min=1e-9)`.

### Голова классификации

$$
\hat{y} = \text{Linear}\bigl(\text{Dropout}\bigl(\text{LayerNorm}(v_{doc})\bigr)\bigr).
$$

LayerNorm перед линейным слоем стабилизирует распределение входа
классификатора между эпохами — без него модель чувствительна к масштабу
$v_{doc}$, который зависит от длины документов. Dropout (0.2) применяется
к уже нормализованному вектору. Голова специально неглубокая: на малых
выборках она быстрее переобучается, чем энкодер успевает адаптироваться.

### Функция потерь и балансировка классов

Используется взвешенный CrossEntropy с label smoothing:

$$
\mathcal{L} = -\sum_{i} w_{y_i}\,\bigl[(1-\alpha)\,\log p_{i,y_i} + \frac{\alpha}{C} \sum_{c=1}^{C} \log p_{i,c}\bigr],
$$

где $w_y$ — вес класса, $\alpha$ — `label_smoothing` (0.02 по умолчанию),
$C$ — число классов.

- Веса считаются по обучающей выборке: `sklearn.utils.class_weight
  .compute_class_weight("balanced", ...)`, что эквивалентно
  $w_c = N / (C \cdot n_c)$, где $n_c$ — число объектов класса $c$.
- Веса регистрируются в модели как `buffer` — они переезжают вместе с
  моделью на нужный device, но не попадают в `state_dict` сохранения
  (отфильтровываются в `utils.get_filtered_model_state_dict`), чтобы
  чекпоинт не зависел от конкретной балансировки.
- Label smoothing 0.02 — мягкий: достаточно, чтобы снять
  переуверенность, но не настолько, чтобы заметно ухудшить F1.

### Раздельные learning rate для энкодера и головы

`WeightedChunkTrainer` переопределяет `create_optimizer` и разбивает
параметры на четыре группы:

| Группа               | Префикс имени                | weight_decay | LR             |
|----------------------|------------------------------|--------------|----------------|
| encoder + decay      | `roberta.*` (matrix params)  | `weight_decay` | `lr_encoder` |
| encoder + no_decay   | `roberta.*` (LayerNorm, bias)| 0            | `lr_encoder` |
| head + decay         | всё остальное (matrix)       | `weight_decay` | `lr_head`    |
| head + no_decay      | всё остальное (LayerNorm, bias)| 0          | `lr_head`    |

Используется AdamW c $\beta = (0.9, 0.999)$, $\varepsilon = 10^{-8}$,
cosine LR scheduler и warmup на долю $0.06$ от общего числа шагов
(или абсолютное число шагов, если задано `warmup_steps > 0`). Идея
раздельных LR — энкодеру нужен бережный fine-tune (\~1e-5), голова при
случайной инициализации может (и должна) учиться на более высоком LR;
регулируется без перекомпоновки optimizer-groups.

Кроме того, `freeze_encoder_layers` (12 по умолчанию) и
`freeze_embeddings=True` отключают `requires_grad` у нижних слоёв
энкодера и у слоя эмбеддингов: нижняя половина ruRoberta-large
отвечает за общее лексическое представление и обычно достаточно хороша
«из коробки», а замораживание сильно снижает риск катастрофического
забывания и экономит VRAM. Для текущих гиперпараметров обучаемыми
остаются \~12 верхних энкодер-слоёв + LayerNorm + голова.

### Mixed precision и gradient checkpointing

- **bf16** включается автоматически, если CUDA capability ≥ 8
  (Ampere/Hopper). bf16 имеет тот же диапазон экспоненты, что и fp32,
  поэтому не требует loss scaling и стабильнее fp16.
- **fp16** используется как fallback на более старых GPU
  (`fp16_fallback_on_non_a100=True`).
- **tf32** включается на CUDA для матричных операций — это «бесплатное»
  ускорение matmul без потери точности на практике.
- **gradient_checkpointing** включается через `TrainingArguments` (а не
  в `__init__` модели — иначе несовместимо с актуальным HF Trainer);
  одновременно ставится `use_cache=False`, потому что KV-кэш
  несовместим с обратным проходом по чек-поинтам активаций. Замедляет
  обучение примерно на 20–30%, но позволяет уместить ruRoberta-large +
  6 чанков по 512 токенов в VRAM скромных GPU.

### Сохранение результатов

Чтобы не дублировать веса на диске, обучение использует пару коллбеков
вместо стандартного `save_strategy="epoch"`:

- `BestMetricInMemoryCallback` слушает `on_evaluate` и держит лучший
  state\_dict в RAM (на CPU, через `detach().cpu().clone()`). После
  `trainer.train()` веса откатываются на лучшую эпоху через
  `load_state_dict_into_model(..., strict=False)` (strict=False нужен,
  потому что в модели есть buffer `class_weights`, а в сохранённом
  state\_dict его сознательно нет) и считается финальный `evaluate`.
- `RollingResumeCheckpointCallback` слушает `on_epoch_end` и
  перезаписывает один и тот же файл `resume_checkpoint.pt`. Запись
  делается in-place (open + truncate + write поверх того же inode),
  а не через atomic rename. Это сознательное решение: на Google
  Drive-маунтах rename интерпретируется как delete+create и старая
  версия уходит в корзину, быстро забивая квоту. Trade-off: если
  процесс упадёт во время записи, чекпоинт окажется повреждённым, но
  переписывается он каждую эпоху, так что в худшем случае теряется
  одна эпоха обучения.

На диск пишутся только `metrics.json` (метрики + конфиги) и
`resume_checkpoint.pt` (веса + state оптимизатора + state
scheduler-а — всё необходимое для продолжения обучения).

---

## Структура модуля

| Файл           | Содержимое                                                        |
|----------------|-------------------------------------------------------------------|
| `config.py`    | dataclass-конфиги: `ModelConfig`, `TrainConfig`, `DataConfig`, `PathConfig` |
| `data.py`      | загрузка CSV, label mapping, токенизация с чанкованием, class weights |
| `model.py`     | `ChunkMeanPoolRobertaClassifier`, `ChunkDataCollator`, `build_model` |
| `training.py`  | `WeightedChunkTrainer`, коллбеки, `run_training_pipeline`         |
| `metrics.py`   | `compute_metrics`: balanced accuracy, F1-macro, precision, recall |
| `utils.py`     | seed, фильтрация state\_dict, очистка памяти                      |
| `main.py`      | CLI и Python API (`run_from_params`, `run_from_configs`)          |

---

## CLI

Минимальный запуск:

```bash
python -m bert_classification.main \
    --train-file data/train.csv \
    --test-file  data/test.csv \
    --output-dir runs/bert_classification
```

С переопределением гиперпараметров (любое поле любого dataclass-конфига
доступно как `--kebab-case`):

```bash
python -m bert_classification.main \
    --train-file data/train.csv \
    --test-file  data/test.csv \
    --output-dir runs/bert_classification \
    --num-epochs 8 \
    --batch-size 1 \
    --grad-accum-steps 8 \
    --lr-encoder 1e-5 \
    --lr-head 1e-5 \
    --freeze-encoder-layers 12 \
    --max-chunks 6
```

---

## Python API

```python
from bert_classification.main import run_from_params

components, eval_metrics = run_from_params(
    train_file="data/train.csv",
    test_file="data/test.csv",
    output_dir="runs/bert_classification",
    num_epochs=8,
    freeze_encoder_layers=12,
    bf16=True,
)

print(eval_metrics["eval_balanced_accuracy"], eval_metrics["eval_f1_macro"])
```

`run_from_params` принимает overrides для любых полей `ModelConfig`,
`TrainConfig`, `DataConfig`; неизвестный ключ → `ValueError`.

---

## Входы

CSV-файлы с двумя колонками (имена настраиваются через `DataConfig`):

| Колонка       | По умолчанию | Назначение                                      |
|---------------|--------------|-------------------------------------------------|
| текст         | `text`       | сырой текст документа                           |
| метка класса  | `label`      | строковая метка класса (mapping строится автоматически) |

Никакой ручной предобработки не требуется — токенизация, чанкование и
построение `label2id` происходят внутри пайплайна.

---

## Выходы

В каталоге `--output-dir` после запуска:

| Файл                    | Содержимое                                           |
|-------------------------|------------------------------------------------------|
| `metrics.json`          | `best_epoch`, `best_metrics`, `final_eval_metrics`, `model_config`, `train_config` |
| `resume_checkpoint.pt`  | веса модели + state оптимизатора + state scheduler-а (для продолжения обучения) |

Веса лучшей эпохи пишутся только в `resume_checkpoint.pt`. Чтобы получить
модель для инференса, нужно загрузить веса оттуда — см. функцию
`load_recovery_checkpoint` в `training.py`.

---

## Ключевые параметры (`config.py`)

`ModelConfig`:

| Параметр                  | По умолчанию               | Смысл                                          |
|---------------------------|----------------------------|------------------------------------------------|
| `model_name`              | `ai-forever/ruRoberta-large` | базовая предобученная модель                   |
| `max_length`              | 512                        | длина чанка в токенах                          |
| `stride`                  | 128                        | перекрытие между соседними чанками             |
| `max_chunks`              | 6                          | максимум окон на документ                      |
| `head_dropout`            | 0.2                        | dropout перед линейным слоем                   |
| `label_smoothing`         | 0.02                       | сглаживание целевого распределения             |
| `freeze_encoder_layers`   | 12                         | сколько нижних слоёв энкодера заморозить       |
| `freeze_embeddings`       | True                       | замораживать ли слой эмбеддингов               |

`TrainConfig`:

| Параметр                  | По умолчанию               | Смысл                                          |
|---------------------------|----------------------------|------------------------------------------------|
| `batch_size`              | 1                          | размер мини-батча                              |
| `grad_accum_steps`        | 8                          | накопление градиента (эффективный батч = 8)    |
| `num_epochs`              | 10                         | максимум эпох                                  |
| `early_stopping_patience` | 3                          | сколько эпох без улучшения метрики ждать       |
| `lr_encoder`, `lr_head`   | 1e-5, 1e-5                 | раздельные LR для энкодера и для головы        |
| `weight_decay`            | 0.01                       | L2-регуляризация (AdamW)                       |
| `warmup_ratio`            | 0.06                       | доля шагов warmup от общего числа              |
| `metric_for_best_model`   | `f1_macro`                 | по какой метрике выбираем лучшую эпоху         |
| `bf16` / `fp16_fallback_on_non_a100` | True / True     | mixed precision с автодетектом устройства      |
| `gradient_checkpointing`  | True                       | экономия VRAM ценой скорости                   |

---

## Метрики

`compute_metrics` возвращает на каждой эпохе: `accuracy`,
`balanced_accuracy`, `f1_macro`, `precision_macro`, `recall_macro`.
По умолчанию лучшая эпоха выбирается по `f1_macro` (можно сменить через
`--metric-for-best-model balanced_accuracy`).

---

## Продолжение обучения с чекпоинта

Файл `resume_checkpoint.pt` содержит всё необходимое для возобновления.
Загрузить его в модель можно через `load_recovery_checkpoint` (см.
`training.py`). Чекпоинт переписывается in-place в одной и той же точке
файловой системы, поэтому в `--output-dir` всегда лежит ровно одна копия.
