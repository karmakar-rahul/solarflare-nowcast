"""
src/metrics.py
--------------
Space-weather evaluation metrics for binary flare forecasting.

Standard accuracy is USELESS for rare-event forecasting:
  A model that always predicts "no flare" gets >98% accuracy on GOES data.

The space-weather community uses instead:
  • TSS  (True Skill Statistic)  — unaffected by class imbalance
  • HSS  (Heidke Skill Score)    — measures skill vs random chance
  • FAR  (False Alarm Ratio)     — fraction of warnings that were false
  • POD  (Probability of Detection) = recall

Benchmark thresholds (community rough guides):
  TSS > 0.5  →  genuinely useful operational system
  TSS > 0.3  →  better than chance, needs improvement
  TSS < 0.1  →  essentially random
"""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def compute_tss_hss(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute TSS, HSS and related binary classification metrics.

    Args:
        y_true : 1-D int array of true labels {0, 1}
        y_pred : 1-D int array of predicted labels {0, 1}

    Returns:
        dict with keys: tss, hss, pod, far, precision, recall, f1,
                        tp, tn, fp, fn, accuracy
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)

    # Confusion matrix entries
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # POD (Probability of Detection) = recall = sensitivity
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # POFD (Probability of False Detection) = fall-out = FPR
    pofd = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # FAR (False Alarm Ratio) — differs from false-positive rate!
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0

    # TSS = POD - POFD  (ranges -1 to +1; 0 = no skill; 1 = perfect)
    tss = pod - pofd

    # HSS = 2*(TP*TN - FP*FN) / ((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN))
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2.0 * (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    accuracy  = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "tss":       round(float(tss), 6),
        "hss":       round(float(hss), 6),
        "pod":       round(float(pod), 6),
        "far":       round(float(far), 6),
        "precision": round(float(precision), 6),
        "recall":    round(float(recall), 6),
        "f1":        round(float(f1), 6),
        "accuracy":  round(float(accuracy), 6),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def find_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    metric: str = "tss",
    thresholds: np.ndarray = None,
) -> tuple:
    """
    Grid-search the decision threshold that maximises a given metric.

    Args:
        probs      : 1-D float array of predicted probabilities [0, 1]
        labels     : 1-D int array of true labels {0, 1}
        metric     : metric key to maximise ('tss', 'hss', 'f1')
        thresholds : array of thresholds to try; defaults to np.linspace(0.1, 0.9, 81)

    Returns:
        (best_threshold, best_metrics_dict)
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)

    best_score = -np.inf
    best_thresh = 0.5
    best_metrics = {}

    for t in thresholds:
        preds = (probs >= t).astype(int)
        m = compute_tss_hss(labels, preds)
        if m[metric] > best_score:
            best_score  = m[metric]
            best_thresh = float(t)
            best_metrics = m

    print(f"[metrics] Best threshold by {metric.upper()}: "
          f"{best_thresh:.2f}  →  TSS={best_metrics['tss']:.4f}  "
          f"HSS={best_metrics['hss']:.4f}  "
          f"recall={best_metrics['recall']:.4f}  "
          f"precision={best_metrics['precision']:.4f}")

    return best_thresh, best_metrics


if __name__ == "__main__":
    # Demonstrate on toy data
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=1000)
    y_pred = rng.integers(0, 2, size=1000)
    m = compute_tss_hss(y_true, y_pred)
    print("Random classifier metrics:", m)