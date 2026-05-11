# bert-classification

Пакет для обучения document classifier на базе `ai-forever/ruRoberta-large` с кастомной классификационной головой поверх document embedding. Ключевая часть архитектуры здесь — не только обработка длинных текстов, а именно trainable head вида `dropout -> linear -> GELU -> dropout -> classifier`, которая обучается классифицировать документ целиком по агрегированному представлению текста.

Во время обучения модель использует `CrossEntropyLoss` с `class_weights` и `label_smoothing`, а лучшая версия модели отслеживается по `f1_macro`. Пакет сохраняет recovery-чекпоинты каждые N эпох, отдельно обновляет лучший checkpoint по валидационной метрике и в конце экспортирует именно лучший `state_dict` и токенизатор в ту же директорию.

## Какую задачу решает

Пакет решает задачу классификации длинных русскоязычных документов, которые не помещаются в стандартное ограничение трансформера по длине входа. Основная цель — получить устойчивый document-level classifier с кастомной head и контролируемым train pipeline, а разбиение на чанки здесь выступает как технический механизм построения document embedding для длинного текста.

Документ режется на несколько перекрывающихся окон фиксированной длины, каждое окно проходит через `ruRoberta`, затем токенные представления усредняются внутри чанка, а chunk-эмбеддинги усредняются на уровне всего документа. Уже после этого итоговый document embedding подаётся в классификационную голову, которая и является основной discriminative частью модели в задаче классификации.

## Как устроен пакет

Структура директории:

```text
bert-classification/
├── __init__.py
├── config.py
├── utils.py
├── metrics.py
├── data.py
├── model.py
├── training.py
├── inference.py
└── main.py
```

### `__init__.py`

Точка удобного импорта. Переэкспортирует основные dataclass-конфиги и главные entrypoint-функции, чтобы пакет можно было использовать либо как CLI-модуль, либо как Python API без обращения к внутренним файлам напрямую.

### `config.py`

Содержит конфигурационные dataclass-объекты:

- `ModelConfig` — параметры модели и chunking-логики, например `model_name`, `max_length`, `stride`, `max_chunks`, `head_dropout`, `label_smoothing`.
- `TrainConfig` — гиперпараметры обучения: batch size, accumulation, learning rates, early stopping, шаг сохранения чекпоинтов и целевая метрика.
- `DataConfig` — имена колонок с текстом и меткой.
- `PathConfig` — пути к train/test CSV и общей output-директории.

Эти классы позволяют одинаково удобно запускать обучение и из CLI, и программно из внешнего `main.py`.

### `utils.py`

Вспомогательные утилиты общего назначения:

- `set_global_seed()` — фиксирует seed для `random`, `numpy`, `torch` и CUDA.
- `cleanup_memory()` — запускает `gc.collect()` и очищает CUDA cache.
- `ensure_dir()` — гарантирует создание директории.
- `get_device_map_location()` — возвращает `cuda` или `cpu`.
- `get_filtered_model_state_dict()` — возвращает `state_dict` без `class_weights`, чтобы экспорт и инференс не ломались на лишнем буфере при загрузке.

### `metrics.py`

Содержит `compute_metrics(eval_pred)`, которая считает:

- `balanced_accuracy`
- `f1_macro`

Эта функция передаётся в Hugging Face `Trainer` как `compute_metrics`, поэтому после каждой валидации метрики попадают в `state.log_history`, откуда callbacks могут выбирать лучший чекпоинт.

### `data.py`

Отвечает за подготовку данных от CSV до `DatasetDict`:

- читает `train_file` и `test_file`;
- чистит пустые строки и `NaN`;
- строит `label2id`/`id2label`;
- проверяет, что в `test` нет unseen labels;
- считает `class_weights`;
- создаёт tokenizer;
- токенизирует документы в несколько чанков.

Важная особенность: кэширование preprocessing отключено через `disable_caching()` и `dataset.map(..., load_from_cache_file=False)`, чтобы результаты старых запусков не переиспользовались на новых train/test файлах. Это сделано специально для безопасного переобучения на разных датасетах без скрытого влияния HF datasets cache.

### `model.py`

Содержит две ключевые сущности:

- `ChunkDataCollator` — собирает батч формы `[batch, chunks, seq_len]` и контролирует, что число чанков и длины последовательностей совпадают с конфигом.
- `ChunkMeanPoolRobertaClassifier` — основная модель.

Логика модели такая:

1. Документ заранее режется на `max_chunks` перекрывающихся окон длины `max_length`.
2. Все чанки прогоняются через `RobertaModel`.
3. Для каждого чанка считается mean-pooling по токенам с учётом `attention_mask`.
4. Затем считается mean-pooling по чанкам с учётом `num_chunks`.
5. Итоговый document embedding идёт в head: dropout → linear → GELU → dropout → classifier.

Если переданы `labels`, модель сразу считает `CrossEntropyLoss` с `class_weights` и `label_smoothing`.

### `training.py`

Главный orchestration-модуль обучения. Здесь находится почти вся логика train pipeline:

- подготовка датасета и tokenizer;
- создание Trainer;
- кастомная модель оптимизации с разными learning rate для encoder и head;
- ранняя остановка;
- периодическое сохранение чекпоинтов;
- отслеживание лучшей модели по `f1_macro`;
- финальный экспорт лучшей модели в `pytorch_model.bin`.

Основные части файла:

#### `WeightedChunkTrainer`

Наследник Hugging Face `Trainer`, который переопределяет:

- `compute_loss()` — просто забирает loss из модели;
- `create_optimizer()` — строит `AdamW` с четырьмя группами параметров: encoder decay / encoder no_decay / head decay / head no_decay.

Это нужно, чтобы использовать один LR для `roberta`-энкодера и другой для classification head.

#### `EpochIntervalCheckpointCallback`

Callback, который на каждой завершённой эпохе проверяет, кратна ли эпоха `checkpoint_every_n_epochs`. Если да, сохраняет recovery-чекпоинт вида `checkpoint_epoch_3.pt`, `checkpoint_epoch_6.pt` и так далее.

Внутри recovery-чекпоинта лежат:

- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `epoch`
- `label2id` / `id2label`
- `model_config`
- `metrics`
- при наличии `class_weights`

Такой файл подходит именно для восстановления обучения после сбоя.

#### `BestMetricTrackerCallback`

Callback, который после каждой эпохи смотрит последние `eval_*` метрики в `state.log_history`, берёт `eval_f1_macro` и сравнивает с лучшим предыдущим значением. При улучшении:

- обновляет `best_metric`, `best_epoch`, `best_metrics`;
- сохраняет `best_checkpoint.pt`;
- держит лучший `state_dict` в памяти для последующего финального экспорта.

#### `export_model_bundle()`

Экспортирует inference-ready артефакты в ту же output-директорию:

- `pytorch_model.bin`
- tokenizer files
- `training_meta.json`

В `pytorch_model.bin` сохраняется именно лучший `state_dict`, а не обязательно последний.

#### `load_recovery_checkpoint()`

Функция для восстановления обучения из `checkpoint_epoch_N.pt`. Загружает model/optimizer/scheduler state и возвращает номер эпохи, с которой нужно продолжать.

### `inference.py`

Файл для загрузки экспортированной модели и предсказания на новых текстах:

- `load_export_bundle(output_dir)` — загружает `training_meta.json`, tokenizer и `pytorch_model.bin`.
- `build_inference_batch()` — повторяет ту же chunking-логику, что использовалась при обучении.
- `predict_texts()` — делает inference на списке строк и возвращает предсказанную метку, id класса и вероятности.

Ключевая идея — train-time и inference-time preprocessing идентичны, поэтому предсказания согласованы с обучением.

### `main.py`

Главный entrypoint для запуска.

Поддерживает два режима:

1. **CLI**: `python -m bert-classification.main ...`
2. **Python API**: импорт `run_from_params(...)` или `run_from_configs(...)`

Внутри файла:

- `build_arg_parser()` — описывает CLI-аргументы;
- `build_configs_from_args()` — преобразует args в dataclass-конфиги;
- `run_from_configs()` — запускает обучение из уже готовых конфигов;
- `run_from_params()` — удобный Python API с параметрами как у CLI;
- `cli_main()` — стандартная входная точка для `python -m`.

## Как всё вместе работает

Полный жизненный цикл выглядит так:

1. `main.py` получает параметры из CLI или из внешнего Python-кода.
2. Из параметров собираются `ModelConfig`, `TrainConfig`, `DataConfig`, `PathConfig`.
3. `training.py` вызывает подготовку данных из `data.py`.
4. `data.py` читает CSV, строит label mapping, отключает reuse старого preprocessing cache и создаёт tokenized `DatasetDict`.
5. `training.py` создаёт модель из `model.py`, data collator и кастомный `WeightedChunkTrainer`.
6. Во время обучения:
   - на каждой эпохе считается validation;
   - `BestMetricTrackerCallback` обновляет лучший checkpoint;
   - `EpochIntervalCheckpointCallback` каждые N эпох сохраняет recovery-checkpoint.
7. После завершения обучения `training.py` экспортирует именно лучший `state_dict` и tokenizer в `output_dir`.
8. Для инференса `inference.py` читает этот export bundle и использует ту же chunking-логику, что и при тренировке.

## Что появляется в `output_dir`

После обучения в одной директории лежат и recovery-чекпоинты, и лучший экспорт:

```text
output_dir/
├── checkpoint_epoch_3.pt
├── checkpoint_epoch_6.pt
├── checkpoint_epoch_9.pt
├── best_checkpoint.pt
├── pytorch_model.bin
├── training_meta.json
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
└── ...
```

Назначение файлов:

- `checkpoint_epoch_N.pt` — промежуточное восстановление после сбоев;
- `best_checkpoint.pt` — лучший training-checkpoint по `f1_macro`;
- `pytorch_model.bin` — лучший экспорт для инференса;
- `training_meta.json` — конфиг модели, колонок и label mapping;
- tokenizer files — всё, что нужно для повторной токенизации на inference.

## Запуск из командной строки

Пример запуска:

```bash
python -m bert-classification.main \
  --train-file /path/to/train_augmented.csv \
  --test-file /path/to/test.csv \
  --output-dir /path/to/saved_models/bert-classification_meanpool \
  --num-epochs 20 \
  --checkpoint-every-n-epochs 3
```

Полезные аргументы:

- `--train-file` — путь к train CSV, обязателен.
- `--test-file` — путь к test/validation CSV, обязателен.
- `--output-dir` — одна директория для recovery-checkpoints и финального экспорта.
- `--num-epochs` — максимальное число эпох.
- `--checkpoint-every-n-epochs` — как часто сохранять recovery-checkpoint.
- `--batch-size`, `--grad-accum-steps` — параметры effective batch size.
- `--lr-encoder`, `--lr-head` — раздельные learning rate для encoder и head.
- `--text-col`, `--label-col` — имена колонок в CSV.

## Запуск из внешнего Python-кода

Если на сервере есть глобальный `main.py`, можно не собирать CLI-команду, а импортировать пакет напрямую:

```python
from bert-classification.main import run_from_params


components, eval_metrics = run_from_params(
    train_file="/path/to/train_augmented.csv",
    test_file="/path/to/test.csv",
    output_dir="/path/to/saved_models/bert-classification_meanpool",
    num_epochs=20,
    checkpoint_every_n_epochs=3,
    batch_size=1,
    grad_accum_steps=8,
    lr_encoder=1.2e-5,
    lr_head=3e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    early_stopping_patience=2,
    seed=42,
    text_col="text",
    label_col="label",
)
```

Этот вариант делает ровно то же самое, что CLI-запуск, просто без shell-обвязки.

## Как восстановиться после сбоя

Если обучение упало, можно загрузить один из файлов `checkpoint_epoch_N.pt`. В текущей реализации функция `load_recovery_checkpoint()` уже умеет восстанавливать:

- веса модели;
- состояние оптимизатора;
- состояние scheduler;
- номер эпохи, с которой стоит продолжать.

Базовая логика восстановления такая:

```python
from bert-classification.training import load_recovery_checkpoint

ckpt, start_epoch = load_recovery_checkpoint(
    checkpoint_path="/path/to/output_dir/checkpoint_epoch_6.pt",
    model=trainer.model,
    optimizer=trainer.optimizer,
    scheduler=trainer.lr_scheduler,
    map_location="cuda",
)
```

Дальше внешний orchestration-код может решить, как именно продолжить обучение с `start_epoch`.

## Почему нет циклических импортов

Структура пакета специально разложена по слоям:

- `config.py`, `utils.py`, `metrics.py` — базовый слой;
- `data.py` и `model.py` зависят только от базового слоя;
- `training.py` собирает вместе `data`, `model`, `metrics`, `utils`;
- `main.py` зависит только от `config.py` и `training.py`;
- `inference.py` использует `config.py`, `model.py`, `utils.py`.

Ни один нижележащий модуль не импортирует `main.py`, а `model.py` не импортирует `training.py`, поэтому circular import не возникает.

## Ограничения и принятые решения

- Лучший checkpoint выбирается по `f1_macro`, потому что именно эта метрика используется как основная целевая при дисбалансных multiclass-задачах.
- HF `Trainer`-сохранения отключены (`save_strategy="no"`), чтобы не плодить лишние служебные checkpoint-папки и держать только нужные `.pt` файлы в одной директории.
- Для export используется `state_dict`, а не полный Python-object save, потому что это стандартный и более переносимый способ сохранения модели в PyTorch.
- В export специально не сохраняется `class_weights`, потому что они нужны для train loss, но не нужны для обычного inference и могут мешать при строгой загрузке экспортированных весов.

## Минимальные зависимости

Пакету нужны как минимум:

- `torch`
- `transformers`
- `datasets`
- `pandas`
- `numpy`
- `scikit-learn`

Также нужен доступ к весам `ai-forever/ruRoberta-large` через Hugging Face Hub или заранее закешированные model files.