# bert_embeddings

Папка для репозитория с дообучением `ai-forever/ruRoberta-large` через MLM и генерацией эмбеддингов длинных документов.

## Структура
- `__init__.py`
- `config.py`
- `data_utils.py`
- `mlm_train.py`
- `embedding_model.py`
- `embed_texts.py`

## Что делает пайплайн
1. Собирает корпус из `train` и `test` текстов.
2. Дообучает `ai-forever/ruRoberta-large` как MLM на твоём домене.
3. Загружает encoder-веса из MLM-чекпоинта.
4. Строит эмбеддинги длинных документов через token-based chunking + overlap.
5. Агрегирует chunk embeddings в один document embedding для cosine similarity и composite vectors.

## Рекомендуемые параметры
Для твоей задачи хороший старт:
- `chunk_size=448`
- `chunk_overlap=96`
- `pooling=mean_max`
- `chunk_aggregation=mean_max`
- `batch_size=8`

## Установка
```bash
pip install torch transformers datasets pandas numpy pyarrow
```

## Обучение MLM
Запускай из корня репозитория:
```bash
python -m bert_embeddings.mlm_train \
  --train_file /path/train_paraphrase_2.csv \
  --test_file /path/test.csv \
  --save_dir /path/saved_models/ruroberta_for_embeddings \
  --epochs 15 \
  --batch_size 4 \
  --max_length 512
```

## Генерация эмбеддингов
Тоже из корня репозитория:
```bash
python -m bert_embeddings.embed_texts \
  --input_csv /path/test.csv \
  --output_npy /path/test_embs.npy \
  --output_csv /path/test_embs_manifest.csv \
  --model_dir /path/saved_models/ruroberta_for_embeddings \
  --chunk_size 448 \
  --chunk_overlap 96 \
  --pooling mean_max \
  --chunk_aggregation mean_max \
  --batch_size 8
```

## Почему запуск через `python -m`
Папка оформлена как пакет с `__init__.py`, поэтому такой запуск корректно работает из корня репозитория и не ломает импорты между файлами.