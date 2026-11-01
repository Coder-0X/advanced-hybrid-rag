import pytest
from app.evaluation.metrics import RAGMetrics
from app.evaluation.benchmark_runner import AutomatedBenchmarkRunner
from app.models.schemas import BenchmarkDatasetItem

def test_metrics_faithfulness():
    context = ["Kubernetes HPA scales pod replicas based on CPU target utilization."]
    grounded_answer = "Kubernetes HPA scales replicas based on CPU utilization."
    hallucinated_answer = "Kubernetes was invented on Mars in 1842 by Victorian explorers."

    score_good = RAGMetrics.calculate_faithfulness(grounded_answer, context)
    score_bad = RAGMetrics.calculate_faithfulness(hallucinated_answer, context)

    assert score_good > 0.8
    assert score_bad < 0.4

def test_metrics_context_precision():
    retrieved = [
        "Kubernetes HPA scales pod replicas automatically based on CPU metrics.", # Relevant
        "Vanilla cake recipe with butter and sugar.",                            # Irrelevant
        "General computing history."                                             # Irrelevant
    ]
    ground_truth = ["Kubernetes HPA scales pod replicas automatically based on CPU metrics."]

    precision = RAGMetrics.calculate_context_precision(retrieved, ground_truth)
    assert precision == 1.0  # Rank 1 is relevant -> Precision@1 = 1.0

def test_metrics_mrr():
    retrieved = [
        "Irrelevant chunk 1",
        "Target relevant chunk",
        "Irrelevant chunk 2"
    ]
    ground_truth = ["Target relevant chunk"]

    mrr = RAGMetrics.calculate_mrr(retrieved, ground_truth)
    assert mrr == 0.5  # Rank 2 -> 1/2 = 0.5

def test_benchmark_runner_integration():
    runner = AutomatedBenchmarkRunner()
    test_dataset = [
        BenchmarkDatasetItem(
            id="test_q1",
            query="How does Kubernetes HPA calculate desired replicas?",
            ground_truth_answer="desiredReplicas = ceil[currentReplicas * (currentMetricValue / desiredMetricValue)]",
            ground_truth_context_fragments=["Horizontal Pod Autoscaler (HPA) to automatically scale"]
        )
    ]
    report = runner.run_benchmark(dataset=test_dataset, top_k=3)
    assert report.sample_count == 1
    assert report.summary.naive.faithfulness >= 0.0
    assert report.summary.hybrid.hit_rate >= 0.0
    assert report.summary.reranked.mrr >= 0.0
