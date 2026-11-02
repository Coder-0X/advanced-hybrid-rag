"""
Empirical Semantic Cache Threshold Calibration via Youden's J Statistic.

Calibrates the optimal cosine similarity threshold tau* over an annotated validation set
of query pairs, balancing True Positive Cache Hits against False Positive Cache Contamination.
Following ROC analysis: tau* = argmax_{tau} (TPR(tau) - FPR(tau)).
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class ROCPoint:
    threshold: float
    tpr: float
    fpr: float
    precision: float
    recall: float
    f1: float
    youden_j: float


class SemanticCacheThresholdCalibrator:
    """
    Evaluates ROC curve across cosine similarity thresholds to calibrate tau*.
    """

    @staticmethod
    def evaluate_threshold_sweep(
        cosine_similarities: np.ndarray,
        ground_truth_labels: np.ndarray,  # 1 for semantic match, 0 for distinct intent
        thresholds: Optional[List[float]] = None
    ) -> List[ROCPoint]:
        """
        Sweep thresholds from 0.70 to 0.99 and calculate ROC / Youden's J points.
        """
        if thresholds is None:
            thresholds = [round(t, 2) for t in np.arange(0.75, 1.00, 0.01)]

        points: List[ROCPoint] = []
        pos_total = int(np.sum(ground_truth_labels == 1))
        neg_total = int(np.sum(ground_truth_labels == 0))

        for tau in thresholds:
            preds = (cosine_similarities >= tau).astype(int)

            tp = int(np.sum((preds == 1) & (ground_truth_labels == 1)))
            fp = int(np.sum((preds == 1) & (ground_truth_labels == 0)))
            fn = int(np.sum((preds == 0) & (ground_truth_labels == 1)))
            tn = int(np.sum((preds == 0) & (ground_truth_labels == 0)))

            tpr = tp / max(1, pos_total)
            fpr = fp / max(1, neg_total)
            prec = tp / max(1, tp + fp)
            rec = tpr
            f1 = (2.0 * prec * rec) / max(1e-5, prec + rec)
            youden_j = tpr - fpr

            points.append(
                ROCPoint(
                    threshold=float(tau),
                    tpr=float(tpr),
                    fpr=float(fpr),
                    precision=float(prec),
                    recall=float(rec),
                    f1=float(f1),
                    youden_j=float(youden_j)
                )
            )

        return points

    @classmethod
    def find_optimal_threshold(
        cls,
        cosine_similarities: np.ndarray,
        ground_truth_labels: np.ndarray
    ) -> Tuple[float, ROCPoint, List[ROCPoint]]:
        """
        Finds tau* maximizing Youden's J statistic.
        """
        points = cls.evaluate_threshold_sweep(cosine_similarities, ground_truth_labels)
        best_point = max(points, key=lambda pt: pt.youden_j)
        return best_point.threshold, best_point, points
