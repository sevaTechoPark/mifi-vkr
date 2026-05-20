## train.csv
* BASELINE METRICS: {'balanced_accuracy': 0.534, 'macro_f1': 0.543}
* cosine_similarity centroid balanced_accuracy: 0.125010, macro_f1: 0.094988
* cosine_similarity nearest balanced_accuracy: 0.176384, macro_f1: 0.177300
* hybrid MLP {'balanced_accuracy': 0.119549, 'macro_f1': 0.091449}
* hybrid classical linear_svc: {'balanced_accuracy': 0.341969, 'macro_f1': 0.314718}
* hybrid classical logreg: {'balanced_accuracy': 0.274202, 'macro_f1': 0.196601}
* [custom_embeder] cosine_similarity centroid balanced_accuracy: 0.564518, macro_f1: 0.498997
* [custom_embeder] cosine_similarity nearest balanced_accuracy: 0.476479, macro_f1: 0.475520
* [custom_embeder] hybrid MLP {'balanced_accuracy': 0.433255, 'macro_f1': 0.326977}
* [custom_embeder] hybrid classical linear_svc: {'balanced_accuracy': 0.533544, 'macro_f1': 0.500058}
* [custom_embeder] hybrid classical logreg: {'balanced_accuracy': 0.54235, 'macro_f1': 0.493934}
* rubert-base-cased AutoModelForSequenceClassification balanced_accuracy: 0.359116, f1_macro: 0.347114
* rubert-base-cased MeanPooling balanced_accuracy: 0.406195, f1_macro: 0.408135
* ruRoberta-large AutoModelForSequenceClassification balanced_accuracy: 0.388449, f1_macro: 0.376954
* ruRoberta-large MeanPooling balanced_accuracy: 0.396386, f1_macro: 0.394374
* ruRoberta-large chunkmean balanced_accuracy: 0.454495, f1_macro: 0.442817

## train_augmented.csv
* BASELINE METRICS: {'balanced_accuracy': 0.489, 'macro_f1': 0.486}
* cosine_similarity centroid balanced_accuracy: 0.102992, macro_f1: 0.102776
* cosine_similarity nearest balanced_accuracy: 0.197040, macro_f1: 0.186326
* hybrid MLP {'balanced_accuracy': 0.212621, 'macro_f1': 0.179966}
* hybrid classical linear_svc: {'balanced_accuracy': 0.394114, 'macro_f1': 0.358195}
* hybrid classical logreg: {'balanced_accuracy': 0.272463, 'macro_f1': 0.232159}
* [custom_embeder] cosine_similarity centroid balanced_accuracy: 0.475639, macro_f1: 0.470315
* [custom_embeder] cosine_similarity nearest balanced_accuracy: 0.475877, macro_f1: 0.487425
* [custom_embeder] hybrid MLP
* [custom_embeder] hybrid classical
* [custom_embeder] hybrid classical
* rubert-base-cased AutoModelForSequenceClassification balanced_accuracy: 0.432780, f1_macro: 0.431439
* rubert-base-cased MeanPooling balanced_accuracy: 0.453594, f1_macro: 0.447740
* ruRoberta-large AutoModelForSequenceClassification balanced_accuracy: 0.506839, f1_macro: 0.487621
* ruRoberta-large MeanPooling balanced_accuracy: 0.479098, f1_macro: 0.469369
* ruRoberta-large chunkmean balanced_accuracy: 0.500587, f1_macro: 0.503661

## train_summarized.csv
* BASELINE METRICS: {'balanced_accuracy': 0.366, 'macro_f1': 0.378}
* rubert-base-cased AutoModelForSequenceClassification balanced_accuracy: 0.192772, f1_macro: 0.183410
* rubert-base-cased MeanPooling balanced_accuracy: 0.242164, f1_macro: 0.240241
* ruRoberta-large AutoModelForSequenceClassification balanced_accuracy: 0.345220, f1_macro: 0.324764
* ruRoberta-large MeanPooling balanced_accuracy: 0.332745, f1_macro: 0.325954

## train_original_plus_summary.csv
* BASELINE METRICS: BASELINE METRICS: {'balanced_accuracy': 0.482, 'macro_f1': 0.497}
* rubert-base-cased AutoModelForSequenceClassification balanced_accuracy: 0.363417, f1_macro: 0.357873
* rubert-base-cased MeanPooling balanced_accuracy: 0.375521, f1_macro: 0.385756
* ruRoberta-large AutoModelForSequenceClassification balanced_accuracy: 0.383546, f1_macro: 0.381091
* ruRoberta-large MeanPooling balanced_accuracy: 0.443646, f1_macro: 0.434780

## train_augmented_summarized.csv
* BASELINE METRICS: {'balanced_accuracy': 0.384, 'macro_f1': 0.395}
* rubert-base-cased AutoModelForSequenceClassification balanced_accuracy: 0.325970, f1_macro: 0.291253
* rubert-base-cased MeanPooling balanced_accuracy: 0.270535, f1_macro: 0.270913
* ruRoberta-large AutoModelForSequenceClassification balanced_accuracy: 0.379313, f1_macro: 0.383067
* ruRoberta-large MeanPooling balanced_accuracy: 0.410907, f1_macro: 0.412532


## train_augmented_original_plus_summary.csv
* BASELINE METRICS: {'balanced_accuracy': 0.467, 'macro_f1': 0.475}
* rubert-base-cased AutoModelForSequenceClassification balanced_accuracy: 0.420246, f1_macro: 0.423175
* rubert-base-cased MeanPooling balanced_accuracy: 0.461001, f1_macro: 0.452567
* ruRoberta-large AutoModelForSequenceClassification balanced_accuracy: 0.477742, f1_macro: 0.458942
* ruRoberta-large MeanPooling balanced_accuracy: 0.511597, f1_macro: 0.498471
* ruRoberta-large chunkmean balanced_accuracy: 0.483281, f1_macro: 0.472848

----

было на датасете train.csv:
[custom_embeder] --noisy mlp {'balanced_accuracy': 0.511481, 'macro_f1': 0.503916}
[custom_embeder] --clean mlp {'balanced_accuracy': 0.420657, 'macro_f1': 0.456348}
[default] --noisy mlp {'balanced_accuracy': 0.277828, 'macro_f1': 0.26883}
[default] --clean mlp {'balanced_accuracy': 0.281334, 'macro_f1': 0.328663}
[custom_embeder] classical linear_svc: {'balanced_accuracy': 0.533544, 'macro_f1': 0.500058}
[custom_embeder] classical logreg: {'balanced_accuracy': 0.54235, 'macro_f1': 0.493934}
[default] classical linear_svc: {'balanced_accuracy': 0.341969, 'macro_f1': 0.314718}
[default] classical logreg: {'balanced_accuracy': 0.274202, 'macro_f1': 0.196601}

было на датасете train.csv:
[custom_embeder] cosine_similarity centroid balanced_accuracy: 0.564518, macro_f1: 0.498997
[custom_embeder] cosine_similarity nearest balanced_accuracy: 0.476479, macro_f1: 0.475520
[default] cosine_similarity centroid balanced_accuracy: 0.125010, macro_f1: 0.094988
[default] cosine_similarity nearest balanced_accuracy: 0.176384, macro_f1: 0.177300

было на датасете train_augmented.csv:
[custom_embeder] cosine_similarity centroid balanced_accuracy: 0.475639, macro_f1: 0.470315
[custom_embeder] cosine_similarity nearest balanced_accuracy: 0.475877, macro_f1: 0.487425
[default] cosine_similarity centroid balanced_accuracy: 0.102992, macro_f1: 0.102776
[default] cosine_similarity nearest balanced_accuracy: 0.197040, macro_f1: 0.186326
----

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

# запуск на датасете train.csv на дефолтных эмбедингах
python -m cosine_similarity_classification.main \
  --train ~/papadyk-vkr/data/train.csv \
  --test ~/papadyk-vkr/data/test.csv

# запуск на датасете train_augmented.csv на дефолтных эмбедингах
python -m cosine_similarity_classification.main \
  --train ~/papadyk-vkr/data/train_augmented.csv \
  --test ~/papadyk-vkr/data/test.csv  

HYBRID:

# сборка векторов на кастомных эмбедингах на датасете train.csv
python -m hybrid.main build \
  --train-file ~/papadyk-vkr/data/train.csv \
  --test-file  ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/data/hybrid-vec-custom-train \
  --model-dir ~/papadyk-vkr/bert-embeddings-out-train

# сборка векторов на кастомных эмбедингах на датасете train_augmented.csv
python -m hybrid.main build \
  --train-file ~/papadyk-vkr/data/train_augmented.csv \
  --test-file  ~/papadyk-vkr/data/test.csv \
  --output-dir ~/papadyk-vkr/data/hybrid-vec-custom-train-augmented \
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

----
