from __future__ import annotations
import torch


def edge_f1(
    confidence: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Precision, recall, F1, and raw counts for loop-closure inlier classification.

    Thresholds confidence scores at `threshold` to produce binary predictions,
    then computes classification metrics treating label = 1.0 as the positive
    (inlier) class.

    Why return counts, not just the three scalars?
        F1 is a nonlinear function of tp/fp/fn. Averaging per-batch F1 scores
        and computing F1 on accumulated counts-then-averaging diverge whenever
        batch size or class balance varies across batches — which it does across
        the outlier-rate/structure sweep. Callers (trainer.py, evaluate.py) must
        accumulate tp/fp/fn across all batches/graphs and compute F1 exactly once
        at the point of reporting, never average an average. See
        implementation_details.md §edge_f1.

    Args:
        confidence: Confidence scores in [0, 1], shape (E,).
                    Typically the loop-closure subset of EdgeGateGNN.forward()
                    output, but accepts any float tensor.
        labels:     Ground-truth inlier labels in {0.0, 1.0}, shape (E,).
        threshold:  Decision boundary (default 0.5).
                    Predicted inlier when confidence >= threshold.

    Returns:
        dict with six keys:
            "precision"  float  tp / (tp + fp); 1.0 if no positive predictions
            "recall"     float  tp / (tp + fn); 1.0 if no positive labels
            "f1"         float  harmonic mean of precision and recall; 0.0 if both are 0
            "tp"         int    true positives
            "fp"         int    false positives
            "fn"         int    false negatives
    """
    pred = confidence >= threshold       # (E,) bool — predicted inlier
    pos  = labels >= 0.5                 # (E,) bool — ground-truth inlier

    tp = int((pred &  pos).sum().item())
    fp = int((pred & ~pos).sum().item())
    fn = int((~pred &  pos).sum().item())

    # Guard against division-by-zero on degenerate inputs (no predictions / no GT inliers)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1        = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
    }
