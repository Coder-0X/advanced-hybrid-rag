import pytest
from app.evaluation.metrics import RAGMetrics
from app.evaluation.benchmark_runner import AutomatedBenchmarkRunner
from app.evaluation.significance import paired_wilcoxon_test, fisher_randomization_test, holm_bonferroni_correction
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

def test_metrics_ndcg_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    qrels = {"doc1": 2, "doc2": 1, "doc3": 0}
    # Rank 1: rel=2, Rank 2: rel=1 -> optimal ranking -> NDCG = 1.0
    ndcg = RAGMetrics.calculate_ndcg_at_k(retrieved, qrels, k=3)
    assert ndcg == 1.0

    # Passage duplicate chunks of the same doc must be deduplicated
    retrieved_with_dups = ["doc1", "doc1", "doc2", "doc3"]
    ndcg_dups = RAGMetrics.calculate_ndcg_at_k(retrieved_with_dups, qrels, k=3)
    assert ndcg_dups == 1.0

    # Suboptimal ranking
    subopt_retrieved = ["doc3", "doc2", "doc1"]
    subopt_ndcg = RAGMetrics.calculate_ndcg_at_k(subopt_retrieved, qrels, k=3)
    assert 0.0 < subopt_ndcg < 1.0

def test_metrics_map_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    qrels = {"doc1": 1, "doc2": 0, "doc3": 1}
    # Rank 1: rel (P@1 = 1/1 = 1.0)
    # Rank 2: non-rel
    # Rank 3: rel (P@3 = 2/3 = 0.6667)
    # Total rel = 2. AP = (1.0 + 0.6667) / 2 = 0.8333
    map_score = RAGMetrics.calculate_map_at_k(retrieved, qrels, k=3)
    assert abs(map_score - 0.8333) < 0.01

def test_metrics_rouge_l():
    cand = "Kubernetes HPA scales pod replicas based on CPU target metrics."
    ref = "Kubernetes HPA scales pod replicas based on CPU metrics."
    f1 = RAGMetrics.calculate_rouge_l(cand, ref)
    assert f1 > 0.85

    disjoint = "Completely different text about solar panels."
    assert RAGMetrics.calculate_rouge_l(disjoint, ref) < 0.2

def test_statistical_significance_tests():
    # Treatment clearly better than baseline
    baseline = [0.2, 0.3, 0.4, 0.1, 0.3, 0.2, 0.4, 0.3, 0.2, 0.3]
    treatment = [0.8, 0.9, 0.7, 0.85, 0.9, 0.8, 0.95, 0.85, 0.75, 0.9]

    w_res = paired_wilcoxon_test(baseline, treatment)
    assert w_res["significant_at_05"] is True
    assert w_res["p_value"] < 0.01

    f_res = fisher_randomization_test(baseline, treatment, n_permutations=1000)
    assert f_res["significant_at_05"] is True
    assert f_res["p_value"] < 0.01

    p_dict = {"metric1": 0.001, "metric2": 0.03, "metric3": 0.06}
    adj = holm_bonferroni_correction(p_dict, alpha=0.05)
    assert "metric1" in adj
    assert adj["metric1"]["statistically_significant"] is True

def test_benchmark_runner_integration():
    runner = AutomatedBenchmarkRunner()
    test_dataset = [
        BenchmarkDatasetItem(
            id="test_q1",
            query="How does Kubernetes HPA calculate desired replicas?",
            ground_truth_answer="desiredReplicas = ceil[currentReplicas * (currentMetricValue / desiredMetricValue)]",
            ground_truth_context_fragments=["Horizontal Pod Autoscaler (HPA) to automatically scale"],
            target_document_id="doc_k8s_autoscaling"
        )
    ]
    report = runner.run_benchmark(dataset=test_dataset, top_k=3, generate_plots=False, save_report=False)
    assert report.sample_count == 1
    assert report.summary.naive.faithfulness >= 0.0
    assert report.summary.hybrid.hit_rate >= 0.0
    assert report.summary.reranked.mrr >= 0.0
