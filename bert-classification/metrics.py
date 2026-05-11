import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def compute_metrics(eval_pred):
    logits, labels_true = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "balanced_accuracy": balanced_accuracy_score(labels_true, preds),
        "f1_macro": f1_score(labels_true, preds, average="macro", zero_division=0),
    }