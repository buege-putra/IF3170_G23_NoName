import numpy as np
from numpy.typing import ArrayLike, NDArray


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> NDArray[np.int_]:
    """Return [[TN, FP], [FN, TP]] untuk label {0, 1}."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return np.array([[tn, fp], [fn, tp]])


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    (_, fp), (_, tp) = confusion_matrix(y_true, y_pred)
    return float(tp / (tp + fp)) if tp + fp else 0.0


def recall(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    (_, _), (fn, tp) = confusion_matrix(y_true, y_pred)
    return float(tp / (tp + fn)) if tp + fn else 0.0


def f1_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if p + r else 0.0


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 0.5 * (f1_score(y_true, y_pred) + f1_score(1 - y_true, 1 - y_pred))


def report(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }
