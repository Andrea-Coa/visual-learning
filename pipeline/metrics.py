"""Classification metrics computed with NumPy (no scikit-learn needed on Colab)."""
import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """cm[i, j] = number of observations of class i predicted as class j."""
    return np.bincount(y_true * n_classes + y_pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def classification_report(cm: np.ndarray, classes: tuple[str, ...]) -> dict:
    """Accuracy plus per-class and macro-averaged precision, recall and F1."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro": {"precision": float(precision.mean()), "recall": float(recall.mean()), "f1": float(f1.mean())},
        "per_class": {
            name: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for name, p, r, f, s in zip(classes, precision, recall, f1, support)
        },
        "confusion_matrix": cm.tolist(),
    }
