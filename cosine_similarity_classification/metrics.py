from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report


def evaluate_predictions(y_true, y_pred):
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "classification_report": classification_report(y_true, y_pred, digits=4),
    }
    return metrics