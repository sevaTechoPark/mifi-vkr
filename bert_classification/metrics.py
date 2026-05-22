"""Метрики качества для HF Trainer.

Используются обе метрики: balanced_accuracy устойчив к дисбалансу классов,
а f1_macro учитывает и precision, и recall. В качестве `metric_for_best_model`
по умолчанию выбран f1_macro (см. TrainConfig).
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def compute_metrics(eval_pred):
    logits, labels_true = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "balanced_accuracy": balanced_accuracy_score(labels_true, preds),
        # zero_division=0 — чтобы при отсутствии предсказаний по классу
        # не получать ворнинги и NaN, а считать вклад этого класса нулевым.
        "f1_macro": f1_score(labels_true, preds, average="macro", zero_division=0),
    }
