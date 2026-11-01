import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.models.schemas import (
    RetrievalMode,
    QueryRequest,
    BenchmarkDatasetItem,
    MetricSummary,
    PipelineBenchmarkComparison,
    BenchmarkReport
)
from app.core.pipeline import HybridRAGEngine
from app.evaluation.metrics import RAGMetrics
from app.evaluation.dataset import DEFAULT_BENCHMARK_DATASET, DEFAULT_CORPUS
from app.config import settings

logger = logging.getLogger(__name__)

class AutomatedBenchmarkRunner:
    """
    Automated Quantitative Benchmarking Suite comparing:
    - Naive RAG (Dense vector search)
    - Hybrid RAG (Dense + BM25 merged via Reciprocal Rank Fusion)
    - Reranked Hybrid RAG (Hybrid Candidates -> Cross-Encoder Reranking)

    Evaluates:
    - Faithfulness (Context Grounding)
    - Context Precision@K
    - Context Recall
    - Hit Rate@K
    - Mean Reciprocal Rank (MRR)
    - Answer Relevancy
    - Latency (Mean, P50, P95)
    """

    def __init__(self, engine: Optional[HybridRAGEngine] = None):
        self.engine = engine or HybridRAGEngine()

    def ensure_corpus_indexed(self):
        """Ensure test corpus is indexed prior to benchmarking."""
        if self.engine.vector_store.count() == 0:
            logger.info("Indexing default evaluation corpus...")
            self.engine.ingest_documents(DEFAULT_CORPUS)

    def run_benchmark(
        self,
        dataset: Optional[List[BenchmarkDatasetItem]] = None,
        top_k: int = 5
    ) -> BenchmarkReport:
        """
        Execute benchmark across all three retrieval modes on the dataset.
        """
        self.ensure_corpus_indexed()
        eval_items = dataset or DEFAULT_BENCHMARK_DATASET
        run_id = f"bench_{int(time.time())}"
        timestamp = datetime.utcnow().isoformat() + "Z"

        modes = [RetrievalMode.NAIVE, RetrievalMode.HYBRID, RetrievalMode.RERANKED]
        mode_metrics: Dict[RetrievalMode, Dict[str, List[float]]] = {
            m: {
                "faithfulness": [],
                "context_precision": [],
                "context_recall": [],
                "hit_rate": [],
                "mrr": [],
                "answer_relevancy": [],
                "latencies_ms": []
            }
            for m in modes
        }

        detailed_results: List[Dict[str, Any]] = []

        total_queries = len(eval_items)
        for idx, item in enumerate(eval_items, start=1):
            if idx % 15 == 0 or idx == 1 or idx == total_queries:
                print(f"  [Evaluating query {idx:3d}/{total_queries}] ({item.category or 'general'}): {item.query[:55]}...")

            item_entry: Dict[str, Any] = {
                "id": item.id,
                "category": item.category or "general",
                "query": item.query,
                "ground_truth_answer": item.ground_truth_answer,
                "modes": {}
            }

            for mode in modes:
                req = QueryRequest(
                    query=item.query,
                    mode=mode,
                    top_k=top_k,
                    generate_answer=True
                )
                resp = self.engine.query(req)

                retrieved_texts = [c.text for c in resp.retrieved_chunks]
                answer = resp.answer or ""

                # Compute quantitative metrics
                faithfulness = RAGMetrics.calculate_faithfulness(answer, retrieved_texts)
                precision = RAGMetrics.calculate_context_precision(retrieved_texts, item.ground_truth_context_fragments)
                recall = RAGMetrics.calculate_context_recall(retrieved_texts, item.ground_truth_context_fragments)
                hit_rate = RAGMetrics.calculate_hit_rate(retrieved_texts, item.ground_truth_context_fragments)
                mrr = RAGMetrics.calculate_mrr(retrieved_texts, item.ground_truth_context_fragments)
                relevancy = RAGMetrics.calculate_answer_relevancy(item.query, answer)
                latency = resp.latencies.total_pipeline_ms

                # Record
                mode_metrics[mode]["faithfulness"].append(faithfulness)
                mode_metrics[mode]["context_precision"].append(precision)
                mode_metrics[mode]["context_recall"].append(recall)
                mode_metrics[mode]["hit_rate"].append(hit_rate)
                mode_metrics[mode]["mrr"].append(mrr)
                mode_metrics[mode]["answer_relevancy"].append(relevancy)
                mode_metrics[mode]["latencies_ms"].append(latency)

                item_entry["modes"][mode.value] = {
                    "answer": answer,
                    "retrieved_chunk_ids": [c.chunk_id for c in resp.retrieved_chunks],
                    "faithfulness": faithfulness,
                    "context_precision": precision,
                    "context_recall": recall,
                    "hit_rate": hit_rate,
                    "mrr": mrr,
                    "answer_relevancy": relevancy,
                    "latency_ms": latency
                }

            detailed_results.append(item_entry)

        # Summarize metrics per mode
        summaries: Dict[str, MetricSummary] = {}
        for mode in modes:
            m_data = mode_metrics[mode]
            lat_stats = RAGMetrics.calculate_latency_stats(m_data["latencies_ms"])
            summaries[mode.value] = MetricSummary(
                faithfulness=round(sum(m_data["faithfulness"]) / len(m_data["faithfulness"]), 4),
                context_precision=round(sum(m_data["context_precision"]) / len(m_data["context_precision"]), 4),
                context_recall=round(sum(m_data["context_recall"]) / len(m_data["context_recall"]), 4),
                hit_rate=round(sum(m_data["hit_rate"]) / len(m_data["hit_rate"]), 4),
                mrr=round(sum(m_data["mrr"]) / len(m_data["mrr"]), 4),
                answer_relevancy=round(sum(m_data["answer_relevancy"]) / len(m_data["answer_relevancy"]), 4),
                latency_mean_ms=lat_stats["mean"],
                latency_p50_ms=lat_stats["p50"],
                latency_p95_ms=lat_stats["p95"]
            )

        # Calculate deltas relative to Naive baseline
        naive_s = summaries["naive"]
        hybrid_s = summaries["hybrid"]
        reranked_s = summaries["reranked"]

        delta_hybrid = {
            "faithfulness": round(hybrid_s.faithfulness - naive_s.faithfulness, 4),
            "context_precision": round(hybrid_s.context_precision - naive_s.context_precision, 4),
            "context_recall": round(hybrid_s.context_recall - naive_s.context_recall, 4),
            "hit_rate": round(hybrid_s.hit_rate - naive_s.hit_rate, 4),
            "mrr": round(hybrid_s.mrr - naive_s.mrr, 4),
            "answer_relevancy": round(hybrid_s.answer_relevancy - naive_s.answer_relevancy, 4),
            "latency_p50_ms": round(hybrid_s.latency_p50_ms - naive_s.latency_p50_ms, 2)
        }

        delta_reranked = {
            "faithfulness": round(reranked_s.faithfulness - naive_s.faithfulness, 4),
            "context_precision": round(reranked_s.context_precision - naive_s.context_precision, 4),
            "context_recall": round(reranked_s.context_recall - naive_s.context_recall, 4),
            "hit_rate": round(reranked_s.hit_rate - naive_s.hit_rate, 4),
            "mrr": round(reranked_s.mrr - naive_s.mrr, 4),
            "answer_relevancy": round(reranked_s.answer_relevancy - naive_s.answer_relevancy, 4),
            "latency_p50_ms": round(reranked_s.latency_p50_ms - naive_s.latency_p50_ms, 2)
        }

        comparison = PipelineBenchmarkComparison(
            naive=naive_s,
            hybrid=hybrid_s,
            reranked=reranked_s,
            delta_hybrid_vs_naive=delta_hybrid,
            delta_reranked_vs_naive=delta_reranked
        )

        report = BenchmarkReport(
            run_id=run_id,
            timestamp=timestamp,
            sample_count=len(eval_items),
            summary=comparison,
            detailed_results=detailed_results
        )

        # Persist report
        self._save_report(report)
        return report

    def _save_report(self, report: BenchmarkReport):
        """Save report to JSON files."""
        run_file = settings.BENCHMARK_RESULTS_DIR / f"{report.run_id}.json"
        latest_file = settings.BENCHMARK_RESULTS_DIR / "latest_benchmark.json"

        report_data = report.model_dump()
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

    @staticmethod
    def format_markdown_table(comparison: PipelineBenchmarkComparison) -> str:
        """Render a formatted markdown comparison table."""
        n = comparison.naive
        h = comparison.hybrid
        r = comparison.reranked
        dh = comparison.delta_hybrid_vs_naive
        dr = comparison.delta_reranked_vs_naive

        def fmt_delta(val: float, is_latency: bool = False) -> str:
            sign = "+" if val > 0 else ""
            if is_latency:
                return f"{sign}{val:.1f}ms"
            return f"{sign}{val:.3f}"

        table = (
            "| Metric | Naive RAG (Dense) | Hybrid RAG (Dense + BM25) | Reranked RAG (Cross-Encoder) | Hybrid Delta | Reranked Delta |\n"
            "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
            f"| **Faithfulness** | {n.faithfulness:.3f} | {h.faithfulness:.3f} | **{r.faithfulness:.3f}** | {fmt_delta(dh['faithfulness'])} | {fmt_delta(dr['faithfulness'])} |\n"
            f"| **Context Precision** | {n.context_precision:.3f} | {h.context_precision:.3f} | **{r.context_precision:.3f}** | {fmt_delta(dh['context_precision'])} | {fmt_delta(dr['context_precision'])} |\n"
            f"| **Context Recall** | {n.context_recall:.3f} | {h.context_recall:.3f} | **{r.context_recall:.3f}** | {fmt_delta(dh['context_recall'])} | {fmt_delta(dr['context_recall'])} |\n"
            f"| **Hit Rate@K** | {n.hit_rate:.3f} | {h.hit_rate:.3f} | **{r.hit_rate:.3f}** | {fmt_delta(dh['hit_rate'])} | {fmt_delta(dr['hit_rate'])} |\n"
            f"| **MRR (Mean Recip. Rank)** | {n.mrr:.3f} | {h.mrr:.3f} | **{r.mrr:.3f}** | {fmt_delta(dh['mrr'])} | {fmt_delta(dr['mrr'])} |\n"
            f"| **Answer Relevancy** | {n.answer_relevancy:.3f} | {h.answer_relevancy:.3f} | **{r.answer_relevancy:.3f}** | {fmt_delta(dh['answer_relevancy'])} | {fmt_delta(dr['answer_relevancy'])} |\n"
            f"| **Latency P50** | {n.latency_p50_ms:.1f} ms | {h.latency_p50_ms:.1f} ms | {r.latency_p50_ms:.1f} ms | {fmt_delta(dh['latency_p50_ms'], True)} | {fmt_delta(dr['latency_p50_ms'], True)} |\n"
            f"| **Latency P95** | {n.latency_p95_ms:.1f} ms | {h.latency_p95_ms:.1f} ms | {r.latency_p95_ms:.1f} ms | - | - |\n"
            f"| **Latency Mean** | {n.latency_mean_ms:.1f} ms | {h.latency_mean_ms:.1f} ms | {r.latency_mean_ms:.1f} ms | - | - |"
        )
        return table
