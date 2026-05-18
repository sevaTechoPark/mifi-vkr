import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import balanced_accuracy_score, f1_score


def evaluate_predictions(y_true, y_pred):
    """
    Возвращает только агрегированные метрики, без per-class отчёта.
    zero_division=0 — глушит UndefinedMetricWarning для классов,
    которые модель ни разу не предсказала.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }