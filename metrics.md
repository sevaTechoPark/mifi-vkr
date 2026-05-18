## train.csv
* BASELINE METRICS: {'balanced_accuracy': 0.534, 'macro_f1': 0.543}
* cosine_similarity centroid balanced_accuracy: 0.125010, macro_f1: 0.094988
* cosine_similarity nearest balanced_accuracy: 0.176384, macro_f1: 0.177300
* hybrid MLP {'balanced_accuracy': 0.119549, 'macro_f1': 0.091449}
* hybrid classical linear_svc: {'balanced_accuracy': 0.341969, 'macro_f1': 0.314718}
* hybrid classical logreg: {'balanced_accuracy': 0.274202, 'macro_f1': 0.196601}
* [custom_embeder] cosine_similarity centroid
* [custom_embeder] cosine_similarity nearest
* [custom_embeder] hybrid MLP 
* [custom_embeder] hybrid classical 
* [custom_embeder] hybrid classical 
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

