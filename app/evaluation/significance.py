"""
Statistical Hypothesis Testing for Information Retrieval and RAG Benchmarking.
Implements:
  1. Two-sided Paired Wilcoxon Signed-Rank Test (Non-parametric for skewed, bounded metrics)
  2. Fisher's Paired Randomization Permutation Test (B = 10,000 permutations)
  3. Holm-Bonferroni Family-Wise Error Rate (FWER) step-down correction for multiple comparisons
Aligned with SIGIR and TREC empirical evaluation standards (Smucker et al., 2007; Carterette, 2012).
"""
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy import stats

def paired_wilcoxon_test(
    scores_baseline: List[float],
    scores_treatment: List[float]
) -> Dict[str, Any]:
    """
    Computes Two-Sided Paired Wilcoxon Signed-Rank Test.
    Suitable for bounded ranking metrics (NDCG, MAP, MRR) where normality assumptions fail.
    """
    a = np.array(scores_baseline, dtype=np.float64)
    b = np.array(scores_treatment, dtype=np.float64)
    diff = b - a

    # Filter out zero differences
    nonzero_diff = diff[diff != 0.0]
    if len(nonzero_diff) == 0:
        return {
            "test": "wilcoxon",
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_at_05": False,
            "mean_difference": 0.0,
            "n_effective": 0
        }

    try:
        res = stats.wilcoxon(b, a, alternative="two-sided", zero_method="wilcox")
        stat = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception:
        stat, p_val = 0.0, 1.0

    return {
        "test": "wilcoxon",
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "significant_at_05": bool(p_val < 0.05),
        "mean_difference": round(float(np.mean(diff)), 4),
        "n_effective": int(len(nonzero_diff))
    }

def fisher_randomization_test(
    scores_baseline: List[float],
    scores_treatment: List[float],
    n_permutations: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Fisher's Paired Randomization Test (Monte Carlo Permutation Test with B permutations).
    Tests the null hypothesis that there is no treatment effect by randomly flipping signs
    of paired differences: d_i = treatment_i - baseline_i.
    """
    a = np.array(scores_baseline, dtype=np.float64)
    b = np.array(scores_treatment, dtype=np.float64)
    n = len(a)
    if n == 0 or len(b) != n:
        raise ValueError("Baseline and treatment arrays must be non-empty and of equal length.")

    diff = b - a
    obs_mean_diff = float(np.mean(diff))
    obs_t = abs(obs_mean_diff)

    rng = np.random.default_rng(seed)
    # Generate B random sign vectors in {-1, 1}
    # Matrix of shape (n_permutations, n)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    permuted_means = np.abs(np.mean(signs * diff, axis=1))

    # Two-sided permutation p-value with standard +1 correction to prevent p=0
    count_extreme = np.sum(permuted_means >= obs_t)
    p_value = (count_extreme + 1.0) / (n_permutations + 1.0)

    return {
        "test": "fisher_randomization",
        "permutations": n_permutations,
        "observed_mean_delta": round(obs_mean_diff, 4),
        "p_value": round(float(p_value), 6),
        "significant_at_05": bool(p_value < 0.05)
    }

def holm_bonferroni_correction(
    raw_p_values: Dict[str, float],
    alpha: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    """
    Holm-Bonferroni step-down correction for multiple hypothesis testing.
    Controls Family-Wise Error Rate (FWER) across metric batteries without the
    excessive conservatism of standard Bonferroni.
    
    Formula:
      Sort p-values: p_(1) <= p_(2) <= ... <= p_(m)
      Adjusted p-value: p_adj_(i) = min(1.0, max_{j <= i} ((m - j + 1) * p_(j)))
      Rejection threshold: p_(i) <= alpha / (m - i + 1)
    """
    m = len(raw_p_values)
    if m == 0:
        return {}

    sorted_items = sorted(raw_p_values.items(), key=lambda kv: kv[1])
    adjusted_results: Dict[str, Dict[str, Any]] = {}
    
    running_max = 0.0
    for rank, (metric_name, p_val) in enumerate(sorted_items, start=1):
        multiplier = m - rank + 1
        adj_p = min(1.0, max(running_max, multiplier * p_val))
        running_max = adj_p
        
        threshold = alpha / multiplier
        is_sig = bool(p_val <= threshold)
        
        adjusted_results[metric_name] = {
            "raw_p_value": round(p_val, 6),
            "adjusted_p_value": round(adj_p, 6),
            "stepdown_rank": rank,
            "threshold_alpha": round(threshold, 6),
            "statistically_significant": is_sig
        }

    return adjusted_results
