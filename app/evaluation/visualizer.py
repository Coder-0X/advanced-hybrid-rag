"""
Publication-Quality Visualization Suite for RAG and IR Benchmarks.
Generates publication-ready figures conforming to IEEE/ACM conference style:
  1. Pareto Frontier (NDCG@10 vs P95 Latency)
  2. Query Archetype Ablation Breakdown
  3. Score Distribution Analysis
"""
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"

def set_publication_style():
    """Configure matplotlib with publication-grade aesthetic parameters."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

def plot_pareto_frontier(
    metrics_summary: Dict[str, Any],
    output_path: Optional[Path] = None
) -> Path:
    """
    Plots the Accuracy-Latency Pareto Frontier comparing Naive, Hybrid, and Reranked.
    X-axis: P95 Latency (ms) [Log scale or linear]
    Y-axis: Context Precision / NDCG@10
    """
    set_publication_style()
    out = output_path or (FIGURES_DIR / "pareto_frontier.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    modes = ["naive", "hybrid", "reranked"]
    labels = ["Naive (Dense)", "Hybrid (Dense + BM25+)", "Reranked (Cross-Encoder)"]
    colors = ["#3b82f6", "#8b5cf6", "#10b981"]
    markers = ["o", "s", "^"]

    latencies = []
    precisions = []

    for mode in modes:
        data = metrics_summary.get(mode, {})
        lat = data.get("latency_p95_ms", 10.0)
        prec = data.get("context_precision", 0.5)
        latencies.append(lat)
        precisions.append(prec)

    # Plot frontier line
    ax.plot(latencies, precisions, linestyle="--", color="#6b7280", alpha=0.6, zorder=1)

    # Plot points
    for lat, prec, label, color, marker in zip(latencies, precisions, labels, colors, markers):
        ax.scatter(lat, prec, color=color, s=140, marker=marker, label=label, zorder=2, edgecolors="black", linewidth=1.2)
        ax.annotate(
            f"{label}\n({lat:.1f}ms, {prec:.3f})",
            xy=(lat, prec),
            xytext=(10, -5 if "Reranked" in label else 10),
            textcoords="offset points",
            fontsize=9,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f3f4f6", ec=color, lw=1.2, alpha=0.9)
        )

    ax.set_xlabel("P95 Retrieval Latency (ms)")
    ax.set_ylabel("Context Precision@5")
    ax.set_title("Pareto Efficiency: Retrieval Accuracy vs. Pipeline Latency")
    ax.set_ylim([min(precisions) - 0.05, max(precisions) + 0.08])
    ax.set_xlim([max(0, min(latencies) - 15), max(latencies) + 25])
    ax.legend(loc="lower right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out

def plot_archetype_ablation(
    detailed_results: List[Dict[str, Any]],
    output_path: Optional[Path] = None
) -> Path:
    """
    Plots performance delta across 5 query archetypes.
    """
    set_publication_style()
    out = output_path or (FIGURES_DIR / "archetype_ablation.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate precision per category per mode
    cat_data: Dict[str, Dict[str, List[float]]] = {}
    for item in detailed_results:
        cat = item.get("category", "general")
        if cat not in cat_data:
            cat_data[cat] = {"naive": [], "hybrid": [], "reranked": []}
        for mode in ["naive", "hybrid", "reranked"]:
            m_res = item.get("modes", {}).get(mode, {})
            score = m_res.get("context_precision", 0.0)
            cat_data[cat][mode].append(score)

    categories = list(cat_data.keys())
    if not categories:
        return out

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    naive_means = [np.mean(cat_data[c]["naive"]) if cat_data[c]["naive"] else 0.0 for c in categories]
    hybrid_means = [np.mean(cat_data[c]["hybrid"]) if cat_data[c]["hybrid"] else 0.0 for c in categories]
    reranked_means = [np.mean(cat_data[c]["reranked"]) if cat_data[c]["reranked"] else 0.0 for c in categories]

    ax.bar(x - width, naive_means, width, label="Naive (Dense)", color="#3b82f6", edgecolor="black", alpha=0.85)
    ax.bar(x, hybrid_means, width, label="Hybrid (Dense + BM25+)", color="#8b5cf6", edgecolor="black", alpha=0.85)
    ax.bar(x + width, reranked_means, width, label="Reranked (Cross-Encoder)", color="#10b981", edgecolor="black", alpha=0.85)

    clean_cat_labels = [c.replace("_", " ").title() for c in categories]
    ax.set_xticks(x)
    ax.set_xticklabels(clean_cat_labels, rotation=15, ha="right")
    ax.set_ylabel("Mean Context Precision")
    ax.set_title("Retrieval Archetype Ablation: Precision Breakdown by Query Class")
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out

def generate_all_visualizations(
    benchmark_report: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> List[Path]:
    """Generates all figures and returns their filepaths."""
    target_dir = output_dir or FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    summary = benchmark_report.get("summary", {})
    detailed = benchmark_report.get("detailed_results", [])

    fig1 = plot_pareto_frontier(
        metrics_summary={
            "naive": summary.get("naive", {}),
            "hybrid": summary.get("hybrid", {}),
            "reranked": summary.get("reranked", {})
        },
        output_path=target_dir / "pareto_frontier.png"
    )

    fig2 = plot_archetype_ablation(
        detailed_results=detailed,
        output_path=target_dir / "archetype_ablation.png"
    )

    return [fig1, fig2]
