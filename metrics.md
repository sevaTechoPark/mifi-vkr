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

Мои новые метрики дляtrain.csv:

--bert-weight 1.0:
* [custom_embeder] hybrid classical
linear_svc: {'balanced_accuracy': 0.417238, 'macro_f1': 0.433428}
logreg: {'balanced_accuracy': 0.389017, 'macro_f1': 0.412037}
ridge_classifier: {'balanced_accuracy': 0.419375, 'macro_f1': 0.432663}
multinomial_nb_tfidf_only: {'balanced_accuracy': 0.103561, 'macro_f1': 0.091129}
complement_nb_tfidf_only: {'balanced_accuracy': 0.321283, 'macro_f1': 0.348292}
logreg_tfidf_only: {'balanced_accuracy': 0.198756, 'macro_f1': 0.204418}
* [custom_embeder] hybrid mlp noisy {'balanced_accuracy': 0.43664, 'macro_f1': 0.492182}
* [custom_embeder] hybrid mlp clean {'balanced_accuracy': 0.390154, 'macro_f1': 0.439314}
* [default] hybrid classical
linear_svc: {'balanced_accuracy': 0.259641, 'macro_f1': 0.263827}
logreg: {'balanced_accuracy': 0.131756, 'macro_f1': 0.108775}
ridge_classifier: {'balanced_accuracy': 0.226946, 'macro_f1': 0.236813}
multinomial_nb_tfidf_only: {'balanced_accuracy': 0.103561, 'macro_f1': 0.091129}
complement_nb_tfidf_only: {'balanced_accuracy': 0.321283, 'macro_f1': 0.348292}
logreg_tfidf_only: {'balanced_accuracy': 0.198756, 'macro_f1': 0.204418}
* [default] hybrid mlp noisy {'balanced_accuracy': 0.274033, 'macro_f1': 0.328513}
* [default] hybrid mlp clean {'balanced_accuracy': 0.203026, 'macro_f1': 0.246099}
--bert-weight 2.0:
* [custom_embeder] hybrid classical
linear_svc: {'balanced_accuracy': 0.429295, 'macro_f1': 0.438792}
logreg: {'balanced_accuracy': 0.389017, 'macro_f1': 0.409166}
ridge_classifier: {'balanced_accuracy': 0.425715, 'macro_f1': 0.440789}
multinomial_nb_tfidf_only: {'balanced_accuracy': 0.103561, 'macro_f1': 0.091129}
complement_nb_tfidf_only: {'balanced_accuracy': 0.321283, 'macro_f1': 0.348292}
logreg_tfidf_only: {'balanced_accuracy': 0.198756, 'macro_f1': 0.204418}
* [custom_embeder] hybrid mlp noisy {'balanced_accuracy': 0.469919, 'macro_f1': 0.510649}
* [custom_embeder] hybrid mlp clean {'balanced_accuracy': 0.318495, 'macro_f1': 0.415975}
--bert-weight 3.0:
* [custom_embeder] hybrid classical
linear_svc: {'balanced_accuracy': 0.429295, 'macro_f1': 0.438707}
logreg: {'balanced_accuracy': 0.389017, 'macro_f1': 0.408866}
ridge_classifier: {'balanced_accuracy': 0.425715, 'macro_f1': 0.440789}
multinomial_nb_tfidf_only: {'balanced_accuracy': 0.103561, 'macro_f1': 0.091129}
complement_nb_tfidf_only: {'balanced_accuracy': 0.321283, 'macro_f1': 0.348292}
logreg_tfidf_only: {'balanced_accuracy': 0.198756, 'macro_f1': 0.204418}
* [custom_embeder] hybrid mlp noisy {'balanced_accuracy': 0.404688, 'macro_f1': 0.404226}
* [custom_embeder] hybrid mlp clean {'balanced_accuracy': 0.375578, 'macro_f1': 0.405228}
--bert-weight 4.0:
* [custom_embeder] hybrid classical
linear_svc: {'balanced_accuracy': 0.429295, 'macro_f1': 0.438707}
logreg: {'balanced_accuracy': 0.392104, 'macro_f1': 0.411287}
ridge_classifier: {'balanced_accuracy': 0.425715, 'macro_f1': 0.440789}
multinomial_nb_tfidf_only: {'balanced_accuracy': 0.103561, 'macro_f1': 0.091129}
complement_nb_tfidf_only: {'balanced_accuracy': 0.321283, 'macro_f1': 0.348292}
logreg_tfidf_only: {'balanced_accuracy': 0.198756, 'macro_f1': 0.204418}
* [custom_embeder] hybrid mlp noisy {'balanced_accuracy': 0.531178, 'macro_f1': 0.526613}
* [custom_embeder] hybrid mlp clean {'balanced_accuracy': 0.383746, 'macro_f1': 0.440861}
--bert-weight 5.0:
* [custom_embeder] hybrid classical
linear_svc: {'balanced_accuracy': 0.429295, 'macro_f1': 0.438707}
logreg: {'balanced_accuracy': 0.392104, 'macro_f1': 0.411287}
ridge_classifier: {'balanced_accuracy': 0.411826, 'macro_f1': 0.421572}
multinomial_nb_tfidf_only: {'balanced_accuracy': 0.103561, 'macro_f1': 0.091129}
complement_nb_tfidf_only: {'balanced_accuracy': 0.321283, 'macro_f1': 0.348292}
logreg_tfidf_only: {'balanced_accuracy': 0.198756, 'macro_f1': 0.204418}
* [custom_embeder] hybrid mlp noisy {'balanced_accuracy': 0.511481, 'macro_f1': 0.503916}
* [custom_embeder] hybrid mlp clean {'balanced_accuracy': 0.420657, 'macro_f1': 0.456348}
--bert-weight 6.0:
* [custom_embeder] hybrid classical
linear_svc: {'balanced_accuracy': 0.429295, 'macro_f1': 0.438707}
logreg: {'balanced_accuracy': 0.391486, 'macro_f1': 0.4098}
ridge_classifier: {'balanced_accuracy': 0.411826, 'macro_f1': 0.421572}
multinomial_nb_tfidf_only: {'balanced_accuracy': 0.103561, 'macro_f1': 0.091129}
complement_nb_tfidf_only: {'balanced_accuracy': 0.321283, 'macro_f1': 0.348292}
logreg_tfidf_only: {'balanced_accuracy': 0.198756, 'macro_f1': 0.204418}
* [custom_embeder] hybrid mlp noisy {'balanced_accuracy': 0.344328, 'macro_f1': 0.479895}
* [custom_embeder] hybrid mlp clean {'balanced_accuracy': 0.35538, 'macro_f1': 0.435689}
--bert-weight 7.0:
* [custom_embeder] hybrid classical
linear_svc: {'balanced_accuracy': 0.427906, 'macro_f1': 0.435856}
logreg: {'balanced_accuracy': 0.391486, 'macro_f1': 0.4098}
ridge_classifier: {'balanced_accuracy': 0.411826, 'macro_f1': 0.421572}
multinomial_nb_tfidf_only: {'balanced_accuracy': 0.103561, 'macro_f1': 0.091129}
complement_nb_tfidf_only: {'balanced_accuracy': 0.321283, 'macro_f1': 0.348292}
logreg_tfidf_only: {'balanced_accuracy': 0.198756, 'macro_f1': 0.204418}
* [custom_embeder] hybrid mlp noisy {'balanced_accuracy': 0.483271, 'macro_f1': 0.473512}
* [custom_embeder] hybrid mlp clean {'balanced_accuracy': 0.400315, 'macro_f1': 0.429637}
--bert-weight 10.0:
* [custom_embeder] hybrid classical
linear_svc: {'balanced_accuracy': 0.427906, 'macro_f1': 0.435856}
logreg: {'balanced_accuracy': 0.391486, 'macro_f1': 0.4098}
ridge_classifier: {'balanced_accuracy': 0.411826, 'macro_f1': 0.421572}
multinomial_nb_tfidf_only: {'balanced_accuracy': 0.103561, 'macro_f1': 0.091129}
complement_nb_tfidf_only: {'balanced_accuracy': 0.321283, 'macro_f1': 0.348292}
logreg_tfidf_only: {'balanced_accuracy': 0.198756, 'macro_f1': 0.204418}
* [custom_embeder] hybrid mlp noisy {'balanced_accuracy': 0.439726, 'macro_f1': 0.503697}
* [custom_embeder] hybrid mlp clean {'balanced_accuracy': 0.386329, 'macro_f1': 0.438586}

Без указания --bert-weight:
* [custom_embeder] hybrid classical
linear_svc: {'balanced_accuracy': 0.429295, 'macro_f1': 0.438707}
logreg: {'balanced_accuracy': 0.392104, 'macro_f1': 0.411287}
ridge_classifier: {'balanced_accuracy': 0.411826, 'macro_f1': 0.421572}
multinomial_nb_tfidf_only: {'balanced_accuracy': 0.103561, 'macro_f1': 0.091129}
complement_nb_tfidf_only: {'balanced_accuracy': 0.321283, 'macro_f1': 0.348292}
logreg_tfidf_only: {'balanced_accuracy': 0.198756, 'macro_f1': 0.204418}
* [custom_embeder] hybrid mlp noisy {'balanced_accuracy': 0.473958, 'macro_f1': 0.501036}
* [custom_embeder] hybrid mlp clean {'balanced_accuracy': 0.415077, 'macro_f1': 0.43839}
* [default] hybrid classical
linear_svc: {'balanced_accuracy': 0.259641, 'macro_f1': 0.263827}
logreg: {'balanced_accuracy': 0.131756, 'macro_f1': 0.108775}
ridge_classifier: {'balanced_accuracy': 0.226946, 'macro_f1': 0.236813}
multinomial_nb_tfidf_only: {'balanced_accuracy': 0.103561, 'macro_f1': 0.091129}
complement_nb_tfidf_only: {'balanced_accuracy': 0.321283, 'macro_f1': 0.348292}
logreg_tfidf_only: {'balanced_accuracy': 0.198756, 'macro_f1': 0.204418}
* [default] hybrid mlp noisy {'balanced_accuracy': 0.277828, 'macro_f1': 0.26883}
* [default] hybrid mlp clean {'balanced_accuracy': 0.281334, 'macro_f1': 0.328663}

Мои метрики для mlp меня устраивают, только поясни все же когда у hybrid mlp использовать --profile clean а когда noisy ? исходя из прошлых метрик и текущих я запутался.
Но меня категорически не устраивают hybrid classical
