# Датасеты

## Формируем train.csv/test.csv по original_data.json

mkdir -p ~/papadyk-vkr/data

Здесь нужно указать абсолютный путь для оригинального json. Дальше можно просто копировать пути.

```
python prepare_dataset.py \
   --original-dataset=/Users/v.papadyk/ml/mifi-vkr/data/original_data.json \
   --data-dir=~/papadyk-vkr/data
```

## Аугментация

Можно запускать параллельно

```
python -m augmentation.paraphrase.main \
   --train-file=~/papadyk-vkr/data/train.csv \
   --output-dir=~/papadyk-vkr/data
```

```
python -m augmentation.backtranslate.main \
   --train-file=~/papadyk-vkr/data/train.csv \
   --output-dir=~/papadyk-vkr/data
```

После завершения двух этих скриптов:

```
python -m utils.merge_augmentations \
   --para-file=~/papadyk-vkr/data/train_paraphrase.csv \
   --bt-file=~/papadyk-vkr/data/train_backtranslate.csv \
   --output-dir=~/papadyk-vkr/data
```

## Суммаризация

Суммаризация не дала прироста, поэтому можно её не запускать.

```
python -m text_summarization.main \
   --input-path=~/papadyk-vkr/data/train_augmented.csv \
   --output-dir=~/papadyk-vkr/data
```

Всё! На этом все файлы сгенерированы и можно запускать классификации.

В директории ~/papadyk-vkr/data:
* train_augmented.csv
* train_augmented_summarized.csv
* train_augmented_original_plus_summary.csv

Можно выполнять классификации по каждому из этих файлов. Метрики выводятся в stdout их надо самому куда-то сохранять.

# Fine-tuned BERT для построение embeddings


mkdir -p ~/papadyk-vkr/bert-embeddings-out-train
mkdir -p ~/papadyk-vkr/bert-embeddings-out-train-augmented

Для датасета train.csv:
```
python -m bert_embeddings.main \
  --train-file ~/papadyk-vkr/data/train.csv \
  --test-file ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/bert-embeddings-out-train
```

Для датасета train_augmented.csv:
```
python -m bert_embeddings.main \
  --train-file ~/papadyk-vkr/data/train_augmented.csv \
  --test-file ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/bert-embeddings-out-train-augmented
```

# Классификация

## I baseline

```
python baseline.py \
   --train-path=~/papadyk-vkr/data/train_augmented.csv \
   --test-path=~/papadyk-vkr/data/test.csv
```

## II Композитные вектора 

Построение композитнных векторов:

Для датасета train.csv:
```
python -m hybrid.main build \
  --train-file ~/papadyk-vkr/data/train.csv \
  --test-file  ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/data/hybrid-vec-custom-train
  --model-dir ~/papadyk-vkr/bert-embeddings-out-train
```

Для датасета train_augmented.csv:
```
python -m hybrid.main build \
  --train ~/papadyk-vkr/data/train_augmented.csv \
  --test ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/data/hybrid-vec-custom-train-augmented \
  --model-dir ~/papadyk-vkr/bert-embeddings-out-train-augmented
```

### Классические модели:

Для датасета train.csv:
```
python -m hybrid.main classical \
  --vecdir ~/papadyk-vkr/data/hybrid-vec-custom-train
```

Для датасета train_augmented.csv:
```
python -m hybrid.main classical \
  --vecdir ~/papadyk-vkr/data/hybrid-vec-custom-train-augmented
```

### MLP:

Для датасета train.csv:
```
python -m hybrid.main mlp \
  --vecdir ~/papadyk-vkr/data/hybrid-vec-custom-train
```

Для датасета train_augmented.csv:
```
python -m hybrid.main mlp \
  --vecdir ~/papadyk-vkr/data/hybrid-vec-custom-train-augmented
```

## III Косинусное расстоение

Для датасета train.csv:
```
python -m cosine_similarity_classification.main \
  --train ~/papadyk-vkr/data/train.csv \
  --test ~/papadyk-vkr/data/test.csv \
  --model-dir ~/papadyk-vkr/bert-embeddings-out-train
```

Для датасета train_augmented.csv:
```
python -m cosine_similarity_classification.main \
  --train ~/papadyk-vkr/data/train_augmented.csv \
  --test ~/papadyk-vkr/data/test.csv \
  --method nearest \
  --model-dir ~/papadyk-vkr/bert-embeddings-out-train-augmented
```

## IV Классификация бертом

mkdir -p ~/papadyk-vkr/bert-classification-out

При перезапусках лучше удалять эту директорию, чтобы кеши от прошлых данных не повлияли.

```
python -m bert_classification.main \
  --train-file ~/papadyk-vkr/data/train_augmented.csv \
  --test-file ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/bert-classification-out
```

----

HYBRID:

python -m hybrid.main build \
  --train-file ~/papadyk-vkr/data/train.csv \
  --test-file  ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/data/hybrid_vec_custom \
  --model-dir ~/papadyk-vkr/bert-embeddings-out

  python -m hybrid.main build \
  --train-file ~/papadyk-vkr/data/train.csv \
  --test-file  ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/data/hybrid_vec_baseline

python -m hybrid.main classical --vecdir ~/papadyk-vkr/data/hybrid_vec_custom
python -m hybrid.main mlp --vecdir ~/papadyk-vkr/data/hybrid_vec_custom

python -m hybrid.main classical --vecdir ~/papadyk-vkr/data/hybrid_vec_baseline
python -m hybrid.main mlp --vecdir ~/papadyk-vkr/data/hybrid_vec_baseline

COSINE:

# запуск на датасете train.csv с его кастомными эмбедингами
python -m cosine_similarity_classification.main \
  --train ~/papadyk-vkr/data/train.csv \
  --test ~/papadyk-vkr/data/test.csv \
  --model-dir ~/papadyk-vkr/bert-embeddings-out-train

# запуск на датасете train_augmented.csv с его кастомными эмбедингами
python -m cosine_similarity_classification.main \
  --train ~/papadyk-vkr/data/train_augmented.csv \
  --test ~/papadyk-vkr/data/test.csv \
  --model-dir ~/papadyk-vkr/bert-embeddings-out-train-augmented

# запуск на дефолтных эмбедингах
python -m cosine_similarity_classification.main \
  --train ~/papadyk-vkr/data/train.csv \
  --test ~/papadyk-vkr/data/test.csv \

HYBRID:

# сборка векторов на кастомных эмбедингах на датасете train.csv
python -m hybrid.main build \
  --train-file ~/papadyk-vkr/data/train.csv \
  --test-file  ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/data/hybrid-vec-custom-train
  --model-dir ~/papadyk-vkr/bert-embeddings-out-train

# сборка векторов на кастомных эмбедингах на датасете train_augmented.csv
python -m hybrid.main build \
  --train-file ~/papadyk-vkr/data/train_augmented.csv \
  --test-file  ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/data/hybrid-vec-custom-train-augmented
    --model-dir ~/papadyk-vkr/bert-embeddings-out-train-augmented

# сборка векторов на дефолтных эмбедингах на датасете train.csv
python -m hybrid.main build \
  --train-file ~/papadyk-vkr/data/train.csv \
  --test-file  ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/data/hybrid-vec-default-train

# сборка векторов на дефолтных эмбедингах на датасете train_augmented.csv
python -m hybrid.main build \
  --train-file ~/papadyk-vkr/data/train_augmented.csv \
  --test-file  ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/data/hybrid-vec-default-train-augmented

# запуск на датасете train.csv с его кастомными эмбедингами
python -m hybrid.main classical --vecdir ~/papadyk-vkr/data/hybrid-vec-custom-train
# запуск на датасете train_augmented.csv с его кастомными эмбедингами
python -m hybrid.main classical --vecdir ~/papadyk-vkr/data/hybrid-vec-custom-train-augmented
# запуск на датасете train.csv с дефолтными эмбедингами
python -m hybrid.main classical --vecdir ~/papadyk-vkr/data/hybrid-vec-default-train
# запуск на датасете train_augmented.csv с дефолтными эмбедингами
python -m hybrid.main classical --vecdir ~/papadyk-vkr/data/hybrid-vec-default-train-augmented

# запуск на датасете train.csv с его кастомными эмбедингами
python -m hybrid.main mlp --vecdir ~/papadyk-vkr/data/hybrid-vec-custom-train
# запуск на датасете train_augmented.csv с его кастомными эмбедингами
python -m hybrid.main mlp --vecdir ~/papadyk-vkr/data/hybrid-vec-custom-train-augmented
# запуск на датасете train.csv с дефолтными эмбедингами
python -m hybrid.main mlp --vecdir ~/papadyk-vkr/data/hybrid-vec-default-train
# запуск на датасете train_augmented.csv с дефолтными эмбедингами
python -m hybrid.main mlp --vecdir ~/papadyk-vkr/data/hybrid-vec-default-train-augmented
