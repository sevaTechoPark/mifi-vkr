# bert_classification

Пакет для обучения document classifier на базе `ai-forever/ruRoberta-large` для классификации длинных русскоязычных текстов. Модель режет документ на несколько перекрывающихся чанков, получает chunk-level представления через `RobertaModel`, агрегирует их в одно document embedding и затем классифицирует документ с помощью trainable classification head. [file:503][file:501]

Во время обучения используется `CrossEntropyLoss` с `class_weights` и `label_smoothing`, метрики считаются как `balanced_accuracy` и `f1_macro`, а лучшая версия модели отслеживается по `f1_macro`. Пайплайн также умеет сохранять recovery-чекпоинты каждые N эпох, отдельно сохранять лучший training checkpoint, писать `train_history.json` и в конце экспортировать inference-ready bundle с лучшими весами и токенизатором. [file:502][file:505]

## Какую задачу решает

Пакет решает задачу multiclass-классификации длинных документов, которые не помещаются в стандартное ограничение трансформеров по длине входа. Основная цель — получить устойчивый document-level classifier, который корректно работает на длинных текстах за счёт chunking и последующей агрегации chunk embeddings. [file:501][file:503]

Документ разбивается на окна длины `max_length` с перекрытием `stride`, затем каждое окно кодируется `ruRoberta`, после чего происходит mean pooling по токенам внутри чанка и mean pooling по чанкам на уровне документа. Получившийся document embedding подаётся в небольшую MLP-head, которая и обучается на финальную классификацию. [file:501][file:503]

## Структура пакета

```text
bert_classification/
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

Точка удобного импорта для внешнего кода. Переэкспортирует основные dataclass-конфиги и главные entrypoint-функции, чтобы пакет можно было использовать как Python API без обращения к внутренним модулям напрямую. [file:499]

### `config.py`

Содержит все основные dataclass-конфиги проекта. В текущей версии именно `config.py` является **единственным источником дефолтных значений**, а `main.py` только переопределяет их при запуске через CLI или Python API. [file:497][file:500]

- `ModelConfig` — параметры модели и chunking: `model_name`, `max_length`, `stride`, `max_chunks`, `head_dropout`, `label_smoothing`. [file:497]
- `TrainConfig` — параметры обучения: `batch_size`, `grad_accum_steps`, `num_epochs`, `lr_encoder`, `lr_head`, `weight_decay`, `warmup_steps`, `early_stopping_patience`, `checkpoint_every_n_epochs`, `seed`, `dataloader_num_workers`, `max_grad_norm`, `metric_for_best_model`. [file:497]
- `DataConfig` — имена колонок `text` и `label`. [file:497]
- `PathConfig` — пути к train/test CSV и output-директории. [file:497]

Текущие дефолты:

- `model_name="ai-forever/ruRoberta-large"` [file:497]
- `max_length=512`, `stride=256`, `max_chunks=6` [file:497]
- `head_dropout=0.4`, `label_smoothing=0.02` [file:497]
- `batch_size=1`, `grad_accum_steps=8`, `num_epochs=16` [file:497]
- `lr_encoder=8e-6`, `lr_head=2e-5` [file:497]
- `weight_decay=0.02`, `warmup_steps=150` [file:497]
- `early_stopping_patience=3`, `checkpoint_every_n_epochs=4` [file:497]

### `utils.py`

Содержит общие вспомогательные функции, которые используются несколькими модулями. Эти утилиты отвечают за reproducibility, очистку памяти, создание директорий и безопасный экспорт модели без train-only буферов. [file:506]

Основные функции:

- `set_global_seed(seed)` — фиксирует seed для `random`, `numpy`, `torch` и CUDA. [file:506]
- `cleanup_memory()` — вызывает `gc.collect()` и очищает CUDA cache, если доступна GPU. [file:506]
- `ensure_dir(path)` — создаёт директорию, если её ещё нет. [file:506]
- `get_device_map_location()` — возвращает строку `"cuda"` или `"cpu"`. [file:506]
- `get_filtered_model_state_dict(model)` — возвращает `state_dict` без `class_weights`, чтобы train-time buffer не ломал export и inference. [file:506]

### `metrics.py`

Содержит функцию `compute_metrics(eval_pred)`, которая вычисляет две ключевые метрики качества: `balanced_accuracy` и `f1_macro`. Эта функция передаётся в Hugging Face `Trainer`, поэтому результаты валидации автоматически попадают в `state.log_history` и могут использоваться callback’ами при выборе лучшего checkpoint. [file:502][file:505]

### `data.py`

Отвечает за полный путь от CSV-файлов до готового `DatasetDict`, который уже можно передавать в обучение. Здесь происходит загрузка train/test данных, очистка, кодирование меток, вычисление class weights, создание tokenizer и нарезка длинных документов на чанки. [file:501]

Основные функции:

- `load_and_prepare_dataframes()` — читает CSV, убирает `NaN`, пустые строки и лишние пробелы. [file:501]
- `build_label_mappings()` — строит `label2id` и `id2label`, а также проверяет, что в `test` нет unseen labels. [file:501]
- `attach_label_ids()` — добавляет колонку `label_id` в train/test DataFrame. [file:501]
- `compute_class_weights_tensor()` — считает веса классов через `compute_class_weight`. [file:501]
- `build_tokenizer()` — создаёт `AutoTokenizer` для выбранной модели. [file:501]
- `build_tokenize_document_fn()` — создаёт функцию токенизации документа в чанки. [file:501]
- `build_dataset_dict()` — собирает `DatasetDict` и применяет chunking/tokenization ко всему датасету. [file:501]

Важная деталь: preprocessing cache у `datasets` отключён через `disable_caching()` и `dataset.map(..., load_from_cache_file=False)`. Это значит, что старые результаты токенизации не будут молча переиспользоваться между разными экспериментами на новых train/test CSV. [file:501]

### `model.py`

Содержит data collator и саму нейросетевую архитектуру классификатора. Вся модель построена вокруг идеи chunk-level encoding и document-level pooling. [file:503]

Основные части:

- `ChunkDataCollator` — собирает батч формы `[batch, chunks, seq_len]`, проверяет число чанков и длины последовательностей. [file:503]
- `ChunkMeanPoolRobertaClassifier` — основная модель поверх `RobertaModel`. [file:503]
- `build_model()` — фабрика для создания модели из `ModelConfig`, числа классов и label mapping. [file:503]

Логика `ChunkMeanPoolRobertaClassifier`:

1. Документ заранее режется на `max_chunks` перекрывающихся окон длины `max_length`. [file:501][file:503]
2. Все чанки прогоняются через `RobertaModel`. [file:503]
3. Для каждого чанка считается mean pooling по токенам с учётом `attention_mask`. [file:503]
4. Затем считается mean pooling по чанкам с учётом `num_chunks`. [file:503]
5. Итоговый document embedding подаётся в classification head вида `dropout -> linear -> GELU -> LayerNorm -> dropout -> classifier`. [file:503]

Дополнительно в модели включён `gradient_checkpointing`, а `use_cache=False`, чтобы корректно работать в training-режиме и экономить память на больших последовательностях. [file:503]

Если в `forward()` переданы `labels`, модель сразу считает `CrossEntropyLoss` с `class_weights` и `label_smoothing`. Это делает модель совместимой с Hugging Face `Trainer`, который ожидает, что модель сама вернёт `loss` и `logits`. [file:503][file:505]

### `training.py`

Главный orchestration-модуль проекта. Он собирает вместе конфиги, данные, tokenizer, Trainer, callbacks, чекпоинты и финальный export модели. [file:505]

Что делает `training.py`:

- готовит train/test данные через `data.py`; [file:505][file:501]
- создаёт tokenizer и dataset; [file:505][file:501]
- собирает модель через `build_model()`; [file:505][file:503]
- создаёт кастомный `WeightedChunkTrainer`; [file:505]
- настраивает раннюю остановку; [file:505]
- сохраняет recovery-чекпоинты каждые N эпох; [file:505]
- отслеживает лучший checkpoint по `f1_macro`; [file:505]
- сохраняет `train_history.json`; [file:505]
- экспортирует лучший `state_dict` и tokenizer в `output_dir`. [file:505]

#### `WeightedChunkTrainer`

Это наследник Hugging Face `Trainer`, который переопределяет:

- `compute_loss()` — берёт loss прямо из модели; [file:505]
- `create_optimizer()` — строит `AdamW` с отдельными группами параметров для encoder и classification head. [file:505]

Параметры делятся на четыре группы:

- encoder + decay;  
- encoder + no_decay;  
- head + decay;  
- head + no_decay. [file:505]

Это позволяет задавать один learning rate для `roberta`-энкодера и другой — для classification head. Такой подход полезен, когда encoder уже сильно pretrained, а head инициализируется с нуля и должен обучаться быстрее. [file:505]

#### `EpochIntervalCheckpointCallback`

Этот callback проверяет на каждой завершённой эпохе, кратна ли эпоха `checkpoint_every_n_epochs`, и если да — сохраняет recovery-чекпоинт вида `checkpoint_epoch_4.pt`, `checkpoint_epoch_8.pt` и так далее при дефолтном конфиге. Recovery-чекпоинт предназначен именно для возобновления обучения после сбоя, а не для итогового inference. [file:497][file:505]

Внутри recovery-чекпоинта сохраняются:

- `model_state_dict`;  
- `optimizer_state_dict`;  
- `scheduler_state_dict`;  
- `epoch`;  
- `label2id` / `id2label`;  
- `model_config`;  
- `metrics`;  
- при наличии `class_weights`. [file:505]

#### `BestMetricTrackerCallback`

Этот callback после каждой эпохи смотрит последние `eval_*` метрики в `state.log_history`, извлекает `eval_f1_macro` и сравнивает его с лучшим предыдущим значением. Если метрика улучшилась, callback обновляет `best_metric`, `best_epoch`, `best_metrics`, сохраняет `best_checkpoint.pt` и держит лучший `state_dict` в памяти для последующего финального экспорта. [file:505]

#### `save_train_history()`

Сохраняет полный `trainer.state.log_history` в файл `train_history.json`. Это удобно для последующего анализа динамики обучения, сравнения запусков и разбора, на какой эпохе модель начинала деградировать. [file:505]

#### `export_model_bundle()`

Экспортирует inference-ready артефакты в `output_dir`. В экспорт входят `pytorch_model.bin`, tokenizer files и `training_meta.json`, а в качестве весов используется именно лучший `state_dict`, а не обязательно последний чекпоинт обучения. [file:505]

#### `load_recovery_checkpoint()`

Позволяет восстановить обучение из одного из recovery-чекпоинтов `checkpoint_epoch_N.pt`. Функция загружает веса модели, состояние оптимизатора, состояние scheduler и возвращает номер эпохи, с которой логически нужно продолжить обучение. [file:505]

#### `run_training_pipeline()`

Главная функция обучения. Она:

1. создаёт output-директорию; [file:505][file:506]
2. готовит tokenizer, данные, label mappings, class weights, dataset и trainer; [file:505][file:501][file:503]
3. запускает `trainer.train()`; [file:505]
4. считает итоговые метрики на validation; [file:505]
5. печатает `FINAL METRICS` и `BEST METRICS`; [file:505]
6. сохраняет `train_history.json`; [file:505]
7. экспортирует лучший bundle для inference. [file:505]

Если во время обучения ни один callback не сохранил лучший state отдельно, pipeline всё равно экспортирует текущий `state_dict` модели как fallback. [file:505]

### `inference.py`

Отвечает за загрузку экспортированной модели и инференс на новых текстах. Здесь повторяется та же chunking-логика, что использовалась при обучении, чтобы preprocessing на train и inference не расходился. [file:498][file:501]

Основные функции:

- `load_export_bundle(output_dir)` — загружает `training_meta.json`, tokenizer и `pytorch_model.bin`; [file:498]
- `build_inference_batch()` — разбивает новые тексты на чанки так же, как на этапе обучения; [file:498]
- `predict_texts()` — получает предсказанные метки, полный вектор вероятностей и `top_k` наиболее вероятных классов. [file:498]

`predict_texts()` возвращает для каждого текста:

- исходный текст;  
- `pred_label`;  
- `pred_id`;  
- `probs`;  
- `top_k` со списком наиболее вероятных классов и их вероятностей. [file:498]

### `main.py`

Главный entrypoint для запуска обучения. Поддерживает два режима: запуск как CLI-модуль через `python -m bert_classification.main` и программный запуск через Python API. [file:500]

Внутри `main.py` находятся:

- `_override_dataclass_from_args()` — аккуратно применяет только явно переданные override к dataclass-конфигу; [file:500]
- `build_configs_from_args()` — собирает конфиги из CLI-аргументов; [file:500]
- `build_arg_parser()` — автоматически строит CLI по полям `ModelConfig`, `TrainConfig` и `DataConfig`; [file:500]
- `run_from_configs()` — запускает обучение из готовых конфигов; [file:500]
- `run_from_params()` — программный API с `**overrides`; [file:500]
- `cli_main()` — стандартная точка входа для `python -m`. [file:500]

Ключевая особенность текущей версии — в `main.py` больше нет дублирования дефолтов. Все значения по умолчанию берутся из dataclass-конфигов, а CLI и Python API лишь переопределяют нужные поля. [file:500][file:497]

## Как всё вместе работает

Жизненный цикл пайплайна выглядит так:

1. `main.py` получает параметры либо из CLI, либо из внешнего Python-кода. [file:500]
2. Из параметров собираются `ModelConfig`, `TrainConfig`, `DataConfig` и `PathConfig`. Если параметр не передан, берётся значение из `config.py`. [file:500][file:497]
3. `training.py` вызывает подготовку данных через `data.py`. [file:505][file:501]
4. `data.py` читает CSV, чистит данные, строит label mapping, считает class weights, отключает reuse preprocessing cache и создаёт tokenized `DatasetDict`. [file:501]
5. `training.py` создаёт модель из `model.py`, data collator и кастомный `WeightedChunkTrainer`. [file:505][file:503]
6. Во время обучения после каждой эпохи считаются validation metrics, `BestMetricTrackerCallback` обновляет лучший checkpoint, а `EpochIntervalCheckpointCallback` по расписанию сохраняет recovery-чекпоинты. [file:505]
7. После завершения обучения `training.py` сохраняет `train_history.json` и экспортирует лучший `state_dict`, tokenizer и metadata в `output_dir`. [file:505]
8. Для инференса `inference.py` читает export bundle и использует ту же chunking-логику, что использовалась при обучении. [file:498][file:501]

## Что появляется в `output_dir`

После обучения в `output_dir` лежат и training-чекпоинты, и inference-ready export. Типичная структура такая: [file:505]

```text
output_dir/
├── checkpoint_epoch_4.pt
├── checkpoint_epoch_8.pt
├── checkpoint_epoch_12.pt
├── best_checkpoint.pt
├── train_history.json
├── pytorch_model.bin
├── training_meta.json
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
└── ...
```

Назначение файлов:

- `checkpoint_epoch_N.pt` — recovery-чекпоинты для возобновления обучения после сбоя; [file:505]
- `best_checkpoint.pt` — лучший training-checkpoint по `f1_macro`; [file:505]
- `train_history.json` — история логов обучения и валидации по эпохам; [file:505]
- `pytorch_model.bin` — экспорт лучших весов для инференса; [file:505]
- `training_meta.json` — metadata с конфигами, путями, метриками и label mapping; [file:505][file:498]
- tokenizer files — всё, что нужно для повторной токенизации на inference. [file:505][file:498]

## Запуск из командной строки

Пример CLI-запуска:

```bash
python -m bert_classification.main \
  --train-file /path/to/train_augmented.csv \
  --test-file /path/to/test.csv \
  --output-dir /path/to/saved_models/bert_classification_meanpool
```

Такой запуск возьмёт все остальные параметры из `config.py`. [file:500][file:497]

Пример запуска с override нескольких параметров:

```bash
python -m bert_classification.main \
  --train-file /path/to/train_augmented.csv \
  --test-file /path/to/test.csv \
  --output-dir /path/to/saved_models/bert_classification_meanpool \
  --num-epochs 20 \
  --lr-encoder 7e-6 \
  --lr-head 1.8e-5 \
  --checkpoint-every-n-epochs 4
```

Полезные аргументы:

- `--train-file` — путь к train CSV; [file:500]
- `--test-file` — путь к validation/test CSV; [file:500]
- `--output-dir` — директория для recovery-чекпоинтов и финального экспорта; [file:500][file:505]
- `--num-epochs` — максимальное число эпох; [file:500][file:497]
- `--checkpoint-every-n-epochs` — периодичность recovery-чекпоинтов; [file:500][file:497]
- `--batch-size`, `--grad-accum-steps` — effective batch size; [file:500][file:497]
- `--lr-encoder`, `--lr-head` — раздельные learning rate для encoder и head; [file:500][file:497]
- `--warmup-steps` — число warmup steps для scheduler; [file:500][file:497]
- `--text-col`, `--label-col` — имена колонок в CSV. [file:500][file:497]

## Запуск из Python

Пример программного запуска через API:

```python
from bert_classification.main import run_from_params

components, eval_metrics = run_from_params(
    train_file="/path/to/train_augmented.csv",
    test_file="/path/to/test.csv",
    output_dir="/path/to/saved_models/bert_classification_meanpool",
)
```

Пример с override нескольких параметров:

```python
from bert_classification.main import run_from_params

components, eval_metrics = run_from_params(
    train_file="/path/to/train_augmented.csv",
    test_file="/path/to/test.csv",
    output_dir="/path/to/saved_models/bert_classification_meanpool",
    num_epochs=20,
    lr_encoder=7e-6,
    lr_head=1.8e-5,
    head_dropout=0.45,
    label_smoothing=0.01,
    early_stopping_patience=3,
    checkpoint_every_n_epochs=4,
)
```

Если в `run_from_params()` передан неизвестный параметр, будет выброшен `ValueError`. Это защищает от тихих опечаток в названиях override-полей. [file:500]

## Как восстановиться после сбоя

Если обучение прервалось, можно загрузить один из файлов `checkpoint_epoch_N.pt` через `load_recovery_checkpoint()`. Функция уже умеет восстановить веса модели, оптимизатор, scheduler и вернуть эпоху, с которой нужно продолжать. [file:505]

Пример:

```python
from bert_classification.training import load_recovery_checkpoint

ckpt, start_epoch = load_recovery_checkpoint(
    checkpoint_path="/path/to/output_dir/checkpoint_epoch_8.pt",
    model=trainer.model,
    optimizer=trainer.optimizer,
    scheduler=trainer.lr_scheduler,
    map_location="cuda",
)
```

После этого внешний orchestration-код может решить, как именно продолжать обучение с `start_epoch`. [file:505]

## Пример инференса

Пример предсказания на новых текстах:

```python
from bert_classification.inference import predict_texts

results = predict_texts(
    output_dir="/path/to/saved_models/bert_classification_meanpool",
    texts=[
        "Первый длинный документ...",
        "Второй длинный документ...",
    ],
    top_k=3,
)

for row in results:
    print(row["pred_label"], row["top_k"])
```

Этот код загрузит экспортированную модель, заново нарежет тексты на чанки тем же способом, что и при обучении, и вернёт предсказания с вероятностями. [file:498][file:501]

## Почему нет циклических импортов

Структура пакета разложена по слоям зависимостей:

- `config.py`, `utils.py`, `metrics.py` — базовый слой; [file:497][file:506][file:502]
- `data.py` и `model.py` зависят только от базового слоя; [file:501][file:503]
- `training.py` собирает вместе `data`, `model`, `metrics`, `utils`; [file:505]
- `main.py` зависит от `config.py` и `training.py`; [file:500]
- `inference.py` использует `config.py`, `model.py`, `utils.py`. [file:498]

Нижележащие модули не импортируют `main.py`, а `model.py` не зависит от `training.py`, поэтому circular import не возникает. [file:498][file:500][file:503][file:505]

## Ограничения и принятые решения

- Лучшая модель выбирается по `f1_macro`, потому что это основная целевая метрика в текущем train pipeline. [file:497][file:505]
- Ранняя остановка по умолчанию использует `patience=3`. [file:497][file:505]
- HF-служебные checkpoint-папки отключены через `save_strategy="no"`, чтобы не плодить лишние директории и хранить только нужные `.pt`-файлы в одном месте. [file:505]
- Для export используется `state_dict`, а не сериализация всего Python-объекта модели, потому что это более переносимый и стандартный способ для PyTorch. [file:505][file:506]
- `class_weights` не входят в `pytorch_model.bin`, потому что нужны для train-time loss, но не нужны для обычного inference и могут мешать при строгой загрузке весов. [file:506][file:505]
- Scheduler использует `warmup_steps`, а не `warmup_ratio`. [file:497][file:505]
- Источник дефолтных параметров один — `config.py`; CLI и Python API только переопределяют нужные поля. [file:497][file:500]

## Минимальные зависимости

Для работы пакета нужны как минимум:

- `torch`;  
- `transformers`;  
- `datasets`;  
- `pandas`;  
- `numpy`;  
- `scikit-learn`. [file:501][file:503][file:505]

Для запуска в Colab обычно также полезны:

- `accelerate`;  
- `sentencepiece`;  
- `huggingface_hub`, если требуется логин в Hugging Face Hub для доступа к модели. [file:498]

## Colab-заметка

Если пакет запускается из Google Colab, обычно достаточно:

1. установить зависимости;
2. смонтировать Google Drive;
3. при необходимости залогиниться в Hugging Face Hub;
4. добавить репозиторий в `sys.path`;
5. вызвать `run_from_params(...)`.

При таком сценарии не нужно отдельно руками создавать `Trainer`, повторно определять `set_seed()` или дублировать train loop в ноутбуке, потому что seed, данные, модель, callbacks и checkpointing уже инкапсулированы внутри пакета. [file:500][file:505][file:506]