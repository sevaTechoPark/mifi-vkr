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

Я применил все твои правки. Если будут новые учитывай что дифф от нового кода.

1) почему bert_classification не пишет _trainer_tmp на диск, хотя не импортирует tempfile ?

2) Новые метрики
train.csv
* cosine_similarity centroid: {'balanced_accuracy': 0.125010, 'macro_f1': 0.094988}
* cosine_similarity nearest: {'balanced_accuracy': 0.176384, 'macro_f1': 0.177300}
* hybrid MLP {'balanced_accuracy': 0.194008, 'macro_f1': 0.284068}
* (новая метрика)[custom_embeder] cosine_similarity centroid: {'balanced_accuracy': 0.564518, 'macro_f1': 0.498997}
* (новая метрика)[custom_embeder] cosine_similarity nearest: {'balanced_accuracy': 0.476479, 'macro_f1': 0.475520}
* (новая метрика)[custom_embeder] hybrid MLP {'balanced_accuracy': 0.282995, 'macro_f1': 0.373879}

train_augmented.csv
* cosine_similarity centroid: {'balanced_accuracy': 0.078869, 'macro_f1': 0.068340}
* cosine_similarity nearest: {'balanced_accuracy': 0.175316, 'macro_f1': 0.176586}
* hybrid MLP {'balanced_accuracy': 0.222742, 'macro_f1': 0.340881}
* [custom_embeder] cosine_similarity centroid: {'balanced_accuracy': 0.527558, 'macro_f1': 0.478878}
* [custom_embeder] cosine_similarity nearest: {'balanced_accuracy': 0.536058, 'macro_f1': 0.481524}
* [custom_embeder] hybrid MLP {'balanced_accuracy': 0.335715, 'macro_f1': 0.42528}

подозрительно что cosine_similarity знак в знак такой же и для custom_embeder и для голого берта для обоих методов. мы не делали улучшений для него он и так идеален?

Ошибка при рассчете classical hybrid(и для [custom_embeder] и для дефолтных весов):
(base) v.papadyk@v-papadyk mifi-vkr % python -m hybrid.main classical \
  --vecdir ~/papadyk-vkr/data/hybrid_vec  
  
--- hybrid (TF-IDF + BERT) | class_weight=None ---
linear_svc: {'balanced_accuracy': 0.436222, 'macro_f1': 0.442916}
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/v.papadyk/ml/mifi-vkr/hybrid/main.py", line 163, in <module>
    main()
  File "/Users/v.papadyk/ml/mifi-vkr/hybrid/main.py", line 149, in main
    run_classical(args.vecdir)
  File "/Users/v.papadyk/ml/mifi-vkr/hybrid/hybrid_classical_models.py", line 70, in run_classical
    _fit_eval(
  File "/Users/v.papadyk/ml/mifi-vkr/hybrid/hybrid_classical_models.py", line 29, in _fit_eval
    model.fit(X_train, y_train)
  File "/Users/v.papadyk/anaconda3/lib/python3.12/site-packages/sklearn/base.py", line 1336, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/v.papadyk/anaconda3/lib/python3.12/site-packages/sklearn/calibration.py", line 422, in fit
    raise ValueError(
ValueError: Requesting 3-fold cross-validation but provided less than 3 examples for at least one class.

hybrid MLP custom_embeder сильно просел, но hybrid MLP на дефолтных весах сильно вырос
