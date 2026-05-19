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
* hybrid mlp noisy {'balanced_accuracy': 0.021645, 'macro_f1': 0.068705}
* hybrid mlp clean {'balanced_accuracy': 0.160523, 'macro_f1': 0.219759}
* [custom_embeder] hybrid classical linear_svc: {'balanced_accuracy': 0.436222, 'macro_f1': 0.442916}, logreg: {'balanced_accuracy': 0.396438, 'macro_f1': 0.407003}, ridge_classifier: {'balanced_accuracy': 0.417613, 'macro_f1': 0.427039}, multinomial_nb_tfidf_only: {'balanced_accuracy': 0.103561, 'macro_f1': 0.091129}, complement_nb_tfidf_only: {'balanced_accuracy': 0.321283, 'macro_f1': 0.348292}, logreg_tfidf_only: {'balanced_accuracy': 0.198756, 'macro_f1': 0.204418}
* [custom_embeder] hybrid mlp noisy {'balanced_accuracy': 0.317739, 'macro_f1': 0.439232}
* [custom_embeder] hybrid mlp clean {'balanced_accuracy': 0.445548, 'macro_f1': 0.372871}

критичный вывод: просели custom_embeder hybrid classical метрики(было linear_svc: {'balanced_accuracy': 0.533544, 'macro_f1': 0.500058}, logreg: {'balanced_accuracy': 0.54235, 'macro_f1': 0.493934})

train_augmented.csv(под этот датасет обучался и свой энкодер)
* cosine_similarity centroid entroid-trim 0 {'balanced_accuracy': 0.078869, 'macro_f1': 0.068340}
* cosine_similarity centroid centroid-trim 0.1 centroid: {'balanced_accuracy': 0.102992, 'macro_f1': 0.102776}
* cosine_similarity centroid centroid-trim 0.15 {'balanced_accuracy': 0.092162, 'macro_f1': 0.085607}
* cosine_similarity centroid entroid-trim 0.2 {'balanced_accuracy': 0.102992, 'macro_f1': 0.102776}
* cosine_similarity nearest knn-temperature 0  k=1: balanced_accuracy=0.197040, macro_f1=0.186326 ← best
  k=3: balanced_accuracy=0.103027, macro_f1=0.092587
  k=5: balanced_accuracy=0.091550, macro_f1=0.070845
  k=7: balanced_accuracy=0.063086, macro_f1=0.039819
  k=9: balanced_accuracy=0.083150, macro_f1=0.062339
  k=11: balanced_accuracy=0.080218, macro_f1=0.059250
nearest: {'balanced_accuracy': 0.197040, 'macro_f1': 0.186326}
* cosine_similarity nearest knn-temperature 0.1 k=1: balanced_accuracy=0.197040, macro_f1=0.186326 ← best
  k=3: balanced_accuracy=0.182228, macro_f1=0.173531
  k=5: balanced_accuracy=0.170443, macro_f1=0.171375
  k=7: balanced_accuracy=0.161043, macro_f1=0.163640
  k=9: balanced_accuracy=0.158248, macro_f1=0.163698
  k=11: balanced_accuracy=0.154837, macro_f1=0.161570
nearest: {'balanced_accuracy': 0.197040, 'macro_f1': 0.186326}
* cosine_similarity nearest knn-temperature 0.15 k=1: balanced_accuracy=0.197040, macro_f1=0.186326 ← best
  k=3: balanced_accuracy=0.182228, macro_f1=0.173531
  k=5: balanced_accuracy=0.170443, macro_f1=0.171375
  k=7: balanced_accuracy=0.161043, macro_f1=0.163640
  k=9: balanced_accuracy=0.158248, macro_f1=0.163698
  k=11: balanced_accuracy=0.154837, macro_f1=0.161570
nearest: {'balanced_accuracy': 0.197040, 'macro_f1': 0.186326}
* cosine_similarity nearest knn-temperature 0.2 k=1: balanced_accuracy=0.197040, macro_f1=0.186326 ← best
  k=3: balanced_accuracy=0.182228, macro_f1=0.173531
  k=5: balanced_accuracy=0.170443, macro_f1=0.171375
  k=7: balanced_accuracy=0.161043, macro_f1=0.163640
  k=9: balanced_accuracy=0.158248, macro_f1=0.163698
  k=11: balanced_accuracy=0.154837, macro_f1=0.161570
nearest: {'balanced_accuracy': 0.197040, 'macro_f1': 0.186326}
* [custom_embeder] cosine_similarity centroid centroid-trim 0 {'balanced_accuracy': 0.475639, 'macro_f1': 0.470315}
* [custom_embeder] cosine_similarity centroid centroid-trim 0.1 {'balanced_accuracy': 0.475639, 'macro_f1': 0.470257}
* [custom_embeder] cosine_similarity centroid centroid-trim 0.15 {'balanced_accuracy': 0.475639, 'macro_f1': 0.469022}
* [custom_embeder] cosine_similarity centroid entroid-trim 0.2 {'balanced_accuracy': 0.475639, 'macro_f1': 0.469022}
* [custom_embeder] cosine_similarity nearest knn-temperature 0 k=1: balanced_accuracy=0.475877, macro_f1=0.487425 ← best
  k=3: balanced_accuracy=0.470975, macro_f1=0.479042
  k=5: balanced_accuracy=0.465507, macro_f1=0.473629
  k=7: balanced_accuracy=0.449039, macro_f1=0.452049
  k=9: balanced_accuracy=0.449935, macro_f1=0.453446
  k=11: balanced_accuracy=0.449331, macro_f1=0.453479
nearest: {'balanced_accuracy': 0.475877, 'macro_f1': 0.487425}
* [custom_embeder] cosine_similarity nearest knn-temperature 0.1 k=1: balanced_accuracy=0.475877, macro_f1=0.487425 ← best
  k=3: balanced_accuracy=0.474441, macro_f1=0.484047
  k=5: balanced_accuracy=0.473760, macro_f1=0.479627
  k=7: balanced_accuracy=0.479316, macro_f1=0.484189
  k=9: balanced_accuracy=0.478712, macro_f1=0.481900
  k=11: balanced_accuracy=0.474854, macro_f1=0.473087
nearest: {'balanced_accuracy': 0.475877, 'macro_f1': 0.487425}
* [custom_embeder] cosine_similarity nearest knn-temperature 0.15 k=1: balanced_accuracy=0.475877, macro_f1=0.487425 ← best
  k=3: balanced_accuracy=0.474441, macro_f1=0.484047
  k=5: balanced_accuracy=0.473760, macro_f1=0.479627
  k=7: balanced_accuracy=0.478712, macro_f1=0.483319
  k=9: balanced_accuracy=0.478712, macro_f1=0.481900
  k=11: balanced_accuracy=0.474854, macro_f1=0.473087
nearest: {'balanced_accuracy': 0.475877, 'macro_f1': 0.487425}
* [custom_embeder] cosine_similarity nearest nearest knn-temperature 0.2 k=1: balanced_accuracy=0.475877, macro_f1=0.487425 ← best
  k=3: balanced_accuracy=0.474441, macro_f1=0.484047
  k=5: balanced_accuracy=0.473760, macro_f1=0.479627
  k=7: balanced_accuracy=0.478712, macro_f1=0.483319
  k=9: balanced_accuracy=0.478712, macro_f1=0.481091
  k=11: balanced_accuracy=0.474854, macro_f1=0.473087
nearest: {'balanced_accuracy': 0.475877, 'macro_f1': 0.487425}
* hybrid classical linear_svc: {'balanced_accuracy': 0.326749, 'macro_f1': 0.333718}, linear_svc_calibrated: {'balanced_accuracy': 0.329814, 'macro_f1': 0.339358}, logreg: {'balanced_accuracy': 0.203034, 'macro_f1': 0.17321}, ridge_classifier: {'balanced_accuracy': 0.290112, 'macro_f1': 0.300091}, multinomial_nb_tfidf_only: {'balanced_accuracy': 0.130754, 'macro_f1': 0.118123}, complement_nb_tfidf_only: {'balanced_accuracy': 0.391068, 'macro_f1': 0.410826}, logreg_tfidf_only: {'balanced_accuracy': 0.255237, 'macro_f1': 0.277403}
* hybrid mlp noisy {'balanced_accuracy': 0.18798, 'macro_f1': 0.263201}
* hybrid mlp clean {'balanced_accuracy': 0.194534, 'macro_f1': 0.335835}
* [custom_embeder] hybrid classical linear_svc: {'balanced_accuracy': 0.467432, 'macro_f1': 0.467854}, linear_svc_calibrated: {'balanced_accuracy': 0.464949, 'macro_f1': 0.465898}, logreg: {'balanced_accuracy': 0.472738, 'macro_f1': 0.470336}, ridge_classifier: {'balanced_accuracy': 0.494248, 'macro_f1': 0.480432}, multinomial_nb_tfidf_only: {'balanced_accuracy': 0.130754, 'macro_f1': 0.118123}, complement_nb_tfidf_only: {'balanced_accuracy': 0.391068, 'macro_f1': 0.410826}, logreg_tfidf_only: {'balanced_accuracy': 0.255237, 'macro_f1': 0.277403}
* [custom_embeder] hybrid mlp noisy {'balanced_accuracy': 0.321263, 'macro_f1': 0.392325}
* [custom_embeder] hybrid mlp clean {'balanced_accuracy': 0.224263, 'macro_f1': 0.37946}

исходя из моих метрик выдели лучшие гиперпараметры, которые зададим как дефолтные - речь про knn-temperature и centroid-trim

3) и меня беспокоит warning в hybrid mlp /Users/v.papadyk/anaconda3/lib/python3.12/site-packages/sklearn/metrics/_classification.py:2924: UserWarning: y_pred contains classes not in y_true

4) на ruRoberta-large chunkmean train_augmented.csv получил метрики balanced_accuracy: 0.500587, f1_macro: 0.503661
я ожидаю значения выше, вот мой лог обучения по эпохам если поможет:
Epoch	Training Loss	Validation Loss	Balanced Accuracy	F1 Macro
1	28.131727	3.300491	0.031690	0.015798
2	22.232684	2.354844	0.250324	0.222599
3	12.376443	1.932763	0.381487	0.389741
4	7.809366	1.805618	0.393108	0.396293
5	5.298193	1.822188	0.390828	0.397640
6	3.966512	1.851042	0.461745	0.449475
7	3.325049	1.905259	0.393544	0.397905
8	2.926581	1.930589	0.452595	0.451821
9	2.667361	1.969331	0.418144	0.434723
10	2.527809	1.874999	0.470380	0.475185
11	2.385493	1.950628	0.473122	0.479439
12	2.356318	1.934033	0.491977	0.483480
13	2.343100	1.960858	0.494076	0.488936
14	2.282700	1.932090	0.489883	0.492312
15	2.263277	1.921544	0.501866	0.498737
16	2.245869	2.015441	0.488411	0.483470
17	2.235840	2.021085	0.496942	0.490029
18	2.229277	1.937971	0.500587	0.503661
19	2.231168	1.949299	0.499638	0.501972
20	2.212826	1.983791	0.491547	0.489645
21	2.204553	1.967865	0.484649	0.486948
 [351/351 00:09]

FINAL METRICS (best epoch in-memory weights)
balanced_accuracy: 0.500587
f1_macro:          0.503661
best_epoch:        18