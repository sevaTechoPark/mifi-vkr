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
* cosine_similarity centroid balanced_accuracy: 0.078869, macro_f1: 0.068340
* cosine_similarity nearest balanced_accuracy: 0.175316, macro_f1: 0.176586
* hybrid MLP {'balanced_accuracy': 0.212621, 'macro_f1': 0.179966}
* hybrid classical linear_svc: {'balanced_accuracy': 0.394114, 'macro_f1': 0.358195}
* hybrid classical logreg: {'balanced_accuracy': 0.272463, 'macro_f1': 0.232159}
* [custom_embeder] cosine_similarity centroid
* [custom_embeder] cosine_similarity nearest
* [custom_embeder] hybrid MLP
* [custom_embeder] hybrid classical
* [custom_embeder] hybrid classical
* rubert-base-cased AutoModelForSequenceClassification balanced_accuracy: 0.432780, f1_macro: 0.431439
* rubert-base-cased MeanPooling balanced_accuracy: 0.453594, f1_macro: 0.447740
* ruRoberta-large AutoModelForSequenceClassification balanced_accuracy: 0.506839, f1_macro: 0.487621
* ruRoberta-large MeanPooling balanced_accuracy: 0.479098, f1_macro: 0.469369
* ruRoberta-large chunkmean

----

1) Одна из моделей bert_embeddings/bert_classification удаляет свой resume_checkpoint.pt, он на диске попадает в корзину, и из-за этого заканчивается место, не надо удалять, надо просто перезаписывать. Похоже это делает bert_classification
2) Мои новые метрики:
train.csv:
* cosine_similarity centroid entroid-trim 0 {'balanced_accuracy': 0.125010, 'macro_f1': 0.094988}
* cosine_similarity centroid centroid-trim 0.1 {'balanced_accuracy': 0.128626, 'macro_f1': 0.100025}
* cosine_similarity centroid centroid-trim 0.15 {'balanced_accuracy': 0.132272, 'macro_f1': 0.102786}
* cosine_similarity centroid entroid-trim 0.2 {'balanced_accuracy': 0.131505, 'macro_f1': 0.101600}
* cosine_similarity nearest knn-temperature 0 k=1: balanced_accuracy=0.197657, macro_f1=0.186661 ← best
  k=3: balanced_accuracy=0.105629, macro_f1=0.094179
  k=5: balanced_accuracy=0.099112, macro_f1=0.077578
  k=7: balanced_accuracy=0.070107, macro_f1=0.046456
  k=9: balanced_accuracy=0.061777, macro_f1=0.041187
  k=11: balanced_accuracy=0.054424, macro_f1=0.032175
nearest: {'balanced_accuracy': 0.197657, 'macro_f1': 0.186661}
* cosine_similarity nearest knn-temperature 0.1 k=1: balanced_accuracy=0.197657, macro_f1=0.186661 ← best
  k=3: balanced_accuracy=0.182845, macro_f1=0.174588
  k=5: balanced_accuracy=0.157622, macro_f1=0.153409
  k=7: balanced_accuracy=0.146536, macro_f1=0.148256
  k=9: balanced_accuracy=0.137595, macro_f1=0.138532
  k=11: balanced_accuracy=0.133489, macro_f1=0.137659
nearest: {'balanced_accuracy': 0.197657, 'macro_f1': 0.186661}
* cosine_similarity nearest knn-temperature 0.15 k=1: balanced_accuracy=0.197657, macro_f1=0.186661 ← best
  k=3: balanced_accuracy=0.182845, macro_f1=0.174588
  k=5: balanced_accuracy=0.157622, macro_f1=0.153409
  k=7: balanced_accuracy=0.146536, macro_f1=0.148256
  k=9: balanced_accuracy=0.137595, macro_f1=0.138532
  k=11: balanced_accuracy=0.133489, macro_f1=0.137659
nearest: {'balanced_accuracy': 0.197657, 'macro_f1': 0.186661}
* cosine_similarity nearest knn-temperature 0.2 k=1: balanced_accuracy=0.197657, macro_f1=0.186661 ← best
  k=3: balanced_accuracy=0.182845, macro_f1=0.174588
  k=5: balanced_accuracy=0.157622, macro_f1=0.153409
  k=7: balanced_accuracy=0.146536, macro_f1=0.148256
  k=9: balanced_accuracy=0.137595, macro_f1=0.138532
  k=11: balanced_accuracy=0.133489, macro_f1=0.137659
nearest: {'balanced_accuracy': 0.197657, 'macro_f1': 0.186661}
* [custom_embeder] cosine_similarity centroid entroid-trim 0 {'balanced_accuracy': 0.564518, 'macro_f1': 0.498997}
* [custom_embeder] cosine_similarity centroid centroid-trim 0.1 {'balanced_accuracy': 0.564518, 'macro_f1': 0.498341}
* [custom_embeder] cosine_similarity centroid centroid-trim 0.15 {'balanced_accuracy': 0.578406, 'macro_f1': 0.508218}
* [custom_embeder] cosine_similarity centroid entroid-trim 0.2 {'balanced_accuracy': 0.578406, 'macro_f1': 0.508155}
* [custom_embeder] cosine_similarity nearest knn-temperature 0   k=1: balanced_accuracy=0.488788, macro_f1=0.466559 ← best
  k=3: balanced_accuracy=0.467917, macro_f1=0.461489
  k=5: balanced_accuracy=0.449718, macro_f1=0.455524
  k=7: balanced_accuracy=0.415439, macro_f1=0.422333
  k=9: balanced_accuracy=0.412095, macro_f1=0.419021
  k=11: balanced_accuracy=0.411491, macro_f1=0.420796
nearest: {'balanced_accuracy': 0.488788, 'macro_f1': 0.466559}
* [custom_embeder] cosine_similarity nearest knn-temperature 0.1  k=1: balanced_accuracy=0.488788, macro_f1=0.466559
  k=3: balanced_accuracy=0.501972, macro_f1=0.491218
  k=5: balanced_accuracy=0.503361, macro_f1=0.501170 ← best
  k=7: balanced_accuracy=0.448345, macro_f1=0.446068
  k=9: balanced_accuracy=0.448470, macro_f1=0.447127
  k=11: balanced_accuracy=0.448470, macro_f1=0.447189
nearest: {'balanced_accuracy': 0.503361, 'macro_f1': 0.501170}
* [custom_embeder] cosine_similarity nearest knn-temperature 0.15   k=1: balanced_accuracy=0.488788, macro_f1=0.466559
  k=3: balanced_accuracy=0.501972, macro_f1=0.491218 ← best
  k=5: balanced_accuracy=0.447805, macro_f1=0.445165
  k=7: balanced_accuracy=0.448345, macro_f1=0.446068
  k=9: balanced_accuracy=0.448470, macro_f1=0.447127
  k=11: balanced_accuracy=0.448470, macro_f1=0.449461
nearest: {'balanced_accuracy': 0.501972, 'macro_f1': 0.491218}
* [custom_embeder] cosine_similarity nearest nearest knn-temperature 0.2 k=1: balanced_accuracy=0.488788, macro_f1=0.466559
  k=3: balanced_accuracy=0.501972, macro_f1=0.491218 ← best
  k=5: balanced_accuracy=0.447805, macro_f1=0.445165
  k=7: balanced_accuracy=0.448345, macro_f1=0.446290
  k=9: balanced_accuracy=0.448470, macro_f1=0.449400
  k=11: balanced_accuracy=0.447866, macro_f1=0.448111
nearest: {'balanced_accuracy': 0.501972, 'macro_f1': 0.491218}
* hybrid classical linear_svc: {'balanced_accuracy': 0.259641, 'macro_f1': 0.263827}, logreg: {'balanced_accuracy': 0.131756, 'macro_f1': 0.108775}, ridge_classifier: {'balanced_accuracy': 0.226946, 'macro_f1': 0.236813}, multinomial_nb_tfidf_only: {'balanced_accuracy': 0.103561, 'macro_f1': 0.091129}, complement_nb_tfidf_only: {'balanced_accuracy': 0.321283, 'macro_f1': 0.348292}, logreg_tfidf_only: {'balanced_accuracy': 0.198756, 'macro_f1': 0.204418}
* hybrid noisy {'balanced_accuracy': 0.021645, 'macro_f1': 0.068705}
* hybrid clean {'balanced_accuracy': 0.160523, 'macro_f1': 0.219759}
* [custom_embeder] hybrid classical linear_svc: {'balanced_accuracy': 0.436222, 'macro_f1': 0.442916}, logreg: {'balanced_accuracy': 0.396438, 'macro_f1': 0.407003}, ridge_classifier: {'balanced_accuracy': 0.417613, 'macro_f1': 0.427039}, multinomial_nb_tfidf_only: {'balanced_accuracy': 0.103561, 'macro_f1': 0.091129}, complement_nb_tfidf_only: {'balanced_accuracy': 0.321283, 'macro_f1': 0.348292}, logreg_tfidf_only: {'balanced_accuracy': 0.198756, 'macro_f1': 0.204418}
* [custom_embeder] hybrid noisy {'balanced_accuracy': 0.317739, 'macro_f1': 0.439232}
* [custom_embeder] hybrid clean {'balanced_accuracy': 0.445548, 'macro_f1': 0.372871}

критичный вывод: просели custom_embeder hybrid classical метрики(было linear_svc: {'balanced_accuracy': 0.533544, 'macro_f1': 0.500058}, logreg: {'balanced_accuracy': 0.54235, 'macro_f1': 0.493934})

train_augmented.csv(под этот датасет обучался и свой энкодер)
