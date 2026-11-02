"""
Unit tests for Neural Distillation, Hard Negative Mining, and ONNX INT8 Quantization.
"""
import pytest
import torch
import numpy as np

from app.training.hard_negatives import HardNegativeMiner, Triplet
from app.training.loss import MarginMSELoss, InfoNCELoss, PairwiseHingeLoss
from app.training.dataset import TripletDataset, dynamic_triplet_collate_fn
from app.core.onnx_reranker import ONNXReranker


def test_hard_negative_miner():
    """Verify hard negative miner filters out positive candidates and samples false positives."""
    qrels = {
        "q1": {"doc_pos": 2, "doc_rel": 1, "doc_neg": 0}
    }
    miner = HardNegativeMiner(qrels=qrels)
    corpus_lookup = {
        "doc_pos": "Horizontal Pod Autoscaler formula",
        "doc_rel": "Autoscaling metrics and controller",
        "doc_neg": "PostgreSQL MVCC vacuuming mechanism"
    }

    triplets = miner.mine_triplets_for_query(
        query_id="q1",
        query_text="How does HPA scale pods?",
        retrieved_bm25_ids=["doc_neg", "doc_pos"],
        retrieved_dense_ids=["doc_neg", "doc_rel"],
        corpus_lookup=corpus_lookup,
        num_negatives=2
    )

    assert len(triplets) > 0
    for t in triplets:
        assert t.query_id == "q1"
        assert t.pos_doc_id in ["doc_pos", "doc_rel"]
        assert t.neg_doc_id == "doc_neg"
        assert t.negative_text == corpus_lookup["doc_neg"]


def test_margin_mse_loss():
    """Verify MarginMSE distillation loss computation."""
    loss_fn = MarginMSELoss()

    student_pos = torch.tensor([2.5, 3.0], requires_grad=True)
    student_neg = torch.tensor([1.0, 0.5], requires_grad=True)
    teacher_pos = torch.tensor([5.0, 6.0])
    teacher_neg = torch.tensor([2.0, 1.0])

    loss = loss_fn(student_pos, student_neg, teacher_pos, teacher_neg)
    assert loss.item() > 0.0

    # Test backpropagation
    loss.backward()
    assert student_pos.grad is not None
    assert student_neg.grad is not None


def test_info_nce_loss():
    """Verify InfoNCE contrastive ranking loss."""
    loss_fn = InfoNCELoss(temperature=0.1)

    pos_scores = torch.tensor([[3.0], [2.0]])
    neg_scores = torch.tensor([[1.0, 0.0], [0.5, -0.5]])

    loss = loss_fn(pos_scores, neg_scores)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_dynamic_triplet_collate():
    """Verify dynamic padding collate function pads to batch max length."""
    batch = [
        {
            "pos_input_ids": [101, 2054, 102],
            "pos_attention_mask": [1, 1, 1],
            "neg_input_ids": [101, 102],
            "neg_attention_mask": [1, 1],
            "teacher_margin": 1.5,
            "teacher_pos_score": 4.0,
            "teacher_neg_score": 2.5
        },
        {
            "pos_input_ids": [101, 2054, 2003, 1037, 102],
            "pos_attention_mask": [1, 1, 1, 1, 1],
            "neg_input_ids": [101, 2054, 102],
            "neg_attention_mask": [1, 1, 1],
            "teacher_margin": 2.0,
            "teacher_pos_score": 5.0,
            "teacher_neg_score": 3.0
        }
    ]

    collated = dynamic_triplet_collate_fn(batch, pad_token_id=0)
    assert collated["pos_input_ids"].shape == (2, 5)
    assert collated["neg_input_ids"].shape == (2, 3)
    assert collated["teacher_margin"].shape == (2,)


def test_onnx_reranker_inference():
    """Verify ONNXReranker session loads and accurately ranks relevant passages."""
    reranker = ONNXReranker(use_int8=True)
    query = "What is the Cormack constant in Reciprocal Rank Fusion?"
    passages = [
        ("doc_rel", "Reciprocal Rank Fusion (RRF) uses a canonical Cormack smoothing constant of k = 60."),
        ("doc_irr", "Kubernetes ingress controllers route external HTTP traffic to cluster clusterIP services.")
    ]

    scores = reranker.predict([(query, p[1]) for p in passages])
    assert len(scores) == 2
    assert 0.0 <= scores[0] <= 1.0
    assert 0.0 <= scores[1] <= 1.0
    # Relevant passage must receive a higher score than irrelevant passage
    assert scores[0] > scores[1]
