"""
Standalone CLI Benchmark Runner
Executes quantitative evaluation comparing Naive RAG vs Hybrid RAG vs Reranked RAG
across 105 systematically crafted multi-domain queries.
"""

import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

# Force UTF-8 stdout for Windows consoles
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.pipeline import HybridRAGEngine
from app.evaluation.benchmark_runner import AutomatedBenchmarkRunner


def main():
    print("=" * 90)
    print(" ADVANCED HYBRID RAG ENGINE: QUANTITATIVE BENCHMARK SUITE")
    print("=" * 90)
    print("Experimental Setup:")
    print("  Corpus          : 15 Multi-Domain Technical & Financial Documents (Layout-Aware Ingestion)")
    print("  Test Set        : 105 Ground-Truth Queries (Lexical, Semantic, Tabular, Adversarial, Multihop)")
    print("  Evaluated Modes :")
    print("    1. Naive RAG    : Dense Vector Search (Qdrant + all-MiniLM-L6-v2)")
    print("    2. Hybrid RAG   : Dense + Sparse BM25+ Inverted Index via Cormack RRF (k=60)")
    print("    3. Reranked RAG : Hybrid Candidate Pool (Top-15) -> Cross-Encoder (ms-marco-MiniLM-L-6-v2)")
    print("-" * 90)

    engine = HybridRAGEngine()
    runner = AutomatedBenchmarkRunner(engine=engine)

    print("\nStarting evaluation harness across 105 queries (315 retrieval passes)...")
    report = runner.run_benchmark(generate_plots=True)

    print("\n" + "=" * 90)
    print(" OVERALL BENCHMARK RESULTS MATRIX")
    print("=" * 90)
    table = runner.format_markdown_table(report.summary)
    print(table)
    print("=" * 90)

    # Category Breakdown Analysis
    cat_stats = defaultdict(lambda: {"naive_mrr": [], "hybrid_mrr": [], "reranked_mrr": [], "naive_hit": [], "hybrid_hit": [], "reranked_hit": [], "reranked_ndcg": []})
    for item in report.detailed_results:
        cat = item.get("category", "general")
        modes = item["modes"]
        cat_stats[cat]["naive_mrr"].append(modes["naive"]["mrr"])
        cat_stats[cat]["hybrid_mrr"].append(modes["hybrid"]["mrr"])
        cat_stats[cat]["reranked_mrr"].append(modes["reranked"]["mrr"])
        cat_stats[cat]["naive_hit"].append(modes["naive"]["hit_rate"])
        cat_stats[cat]["hybrid_hit"].append(modes["hybrid"]["hit_rate"])
        cat_stats[cat]["reranked_hit"].append(modes["reranked"]["hit_rate"])
        cat_stats[cat]["reranked_ndcg"].append(modes["reranked"].get("ndcg_10", 0.0))

    print("\n" + "=" * 90)
    print(" ARCHETYPE BREAKDOWN: ABLATION ACROSS QUERY CATEGORIES (MRR@5, NDCG@10 & Hit Rate)")
    print("=" * 90)
    print(f"| {'Query Category':<25} | {'Count':<5} | {'Naive MRR':<10} | {'Hybrid MRR':<10} | {'Reranked MRR':<12} | {'NDCG@10 (Rerank)':<16} | {'Hit Rate (Rerank)':<17} |")
    print(f"| {'-'*25} | {'-'*5} | {'-'*10} | {'-'*10} | {'-'*12} | {'-'*16} | {'-'*17} |")

    for cat, data in sorted(cat_stats.items()):
        cnt = len(data["naive_mrr"])
        n_mrr = sum(data["naive_mrr"]) / cnt
        h_mrr = sum(data["hybrid_mrr"]) / cnt
        r_mrr = sum(data["reranked_mrr"]) / cnt
        r_ndcg = sum(data["reranked_ndcg"]) / cnt
        r_hit = sum(data["reranked_hit"]) / cnt
        print(f"| {cat:<25} | {cnt:<5} | {n_mrr:<10.3f} | {h_mrr:<10.3f} | {r_mrr:<12.3f} | {r_ndcg:<16.3f} | {r_hit:<17.1%} |")
    print("=" * 90)

    # Statistical Significance Summary
    if report.statistical_significance:
        print("\n" + "=" * 90)
        print(" STATISTICAL SIGNIFICANCE TESTING (RERANKED VS NAIVE BASELINE)")
        print("=" * 90)
        print(f"| {'Metric Tested':<22} | {'Wilcoxon Stat':<14} | {'Wilcoxon p-val':<15} | {'Fisher Perm p-val':<18} | {'Significant (α=0.05)':<20} |")
        print(f"| {'-'*22} | {'-'*14} | {'-'*15} | {'-'*18} | {'-'*20} |")
        for m in ["context_precision", "mrr", "ndcg_10", "map_k"]:
            key = f"{m}_reranked_vs_naive"
            res = report.statistical_significance.get(key, {})
            w = res.get("wilcoxon", {})
            f = res.get("fisher_permutation", {})
            sig = "YES (p < 0.05)" if w.get("significant_at_05") else "NO"
            print(f"| {m:<22} | {w.get('statistic', 0.0):<14.2f} | {w.get('p_value', 1.0):<15.6f} | {f.get('p_value', 1.0):<18.6f} | {sig:<20} |")
        print("=" * 90)

    s = report.summary
    print("\n[KEY EMPIRICAL FINDINGS]")
    print(f"* Context Precision  : Naive={s.naive.context_precision:.3f} -> Hybrid={s.hybrid.context_precision:.3f} -> Reranked={s.reranked.context_precision:.3f} (Gain: {s.delta_reranked_vs_naive['context_precision']:+.3f})")
    print(f"* Graded NDCG@10     : Naive={s.naive.ndcg_10:.3f} -> Hybrid={s.hybrid.ndcg_10:.3f} -> Reranked={s.reranked.ndcg_10:.3f} (Gain: {s.delta_reranked_vs_naive['ndcg_10']:+.3f})")
    print(f"* Cranfield MAP@10   : Naive={s.naive.map_k:.3f} -> Hybrid={s.hybrid.map_k:.3f} -> Reranked={s.reranked.map_k:.3f} (Gain: {s.delta_reranked_vs_naive['map_k']:+.3f})")
    print(f"* MRR (Recip. Rank)  : Naive={s.naive.mrr:.3f} -> Hybrid={s.hybrid.mrr:.3f} -> Reranked={s.reranked.mrr:.3f} (Gain: {s.delta_reranked_vs_naive['mrr']:+.3f})")
    print(f"* ROUGE-L F1         : Naive={s.naive.rouge_l:.3f} -> Hybrid={s.hybrid.rouge_l:.3f} -> Reranked={s.reranked.rouge_l:.3f} (Gain: {s.delta_reranked_vs_naive['rouge_l']:+.3f})")
    print(f"* Token F1           : Naive={s.naive.token_f1:.3f} -> Hybrid={s.hybrid.token_f1:.3f} -> Reranked={s.reranked.token_f1:.3f} (Gain: {s.delta_reranked_vs_naive['token_f1']:+.3f})")
    print(f"* Median Latency (P50): Naive={s.naive.latency_p50_ms:.1f}ms | Hybrid={s.hybrid.latency_p50_ms:.1f}ms | Reranked={s.reranked.latency_p50_ms:.1f}ms")
    print(f"\n[Artifact Output] Full quantitative report persisted to: data/benchmarks/{report.run_id}.json")
    print(f"[Publication Figures] Saved to: reports/figures/pareto_frontier.png, reports/figures/archetype_ablation.png")


if __name__ == "__main__":
    main()
