"""
PyTorch Triplet Dataset and Dynamic Collation Engine for Reranker Training.

Provides tokenization, sequence batching, and dynamic padding for Cross-Encoder
fine-tuning and knowledge distillation.
"""
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from app.training.hard_negatives import Triplet, TeacherScoredTriplet


class TripletDataset(Dataset):
    """
    PyTorch Dataset wrapping mined (query, positive, negative) passage triplets.
    Supports both margin targets (for MarginMSE distillation) and un-annotated triplets.
    """

    def __init__(
        self,
        triplets: List[Any],  # Triplet or TeacherScoredTriplet
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 384
    ):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.triplets[idx]

        query = item.query
        pos_text = item.positive_text
        neg_text = item.negative_text

        # Tokenize pair 1: (query, positive)
        pos_encoded = self.tokenizer(
            query,
            pos_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors=None
        )

        # Tokenize pair 2: (query, negative)
        neg_encoded = self.tokenizer(
            query,
            neg_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors=None
        )

        result = {
            "pos_input_ids": pos_encoded["input_ids"],
            "pos_attention_mask": pos_encoded["attention_mask"],
            "neg_input_ids": neg_encoded["input_ids"],
            "neg_attention_mask": neg_encoded["attention_mask"],
        }

        if hasattr(item, "teacher_margin"):
            result["teacher_margin"] = float(item.teacher_margin)
            result["teacher_pos_score"] = float(item.teacher_pos_score)
            result["teacher_neg_score"] = float(item.teacher_neg_score)

        return result


def dynamic_triplet_collate_fn(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Collate function with dynamic sequence padding to batch max length.
    Prevents allocating wasted FLOPs on fixed max_length padding across batches.
    """
    # 1. Pad positives
    pos_max_len = max(len(item["pos_input_ids"]) for item in batch)
    pos_input_ids = []
    pos_attention_mask = []

    for item in batch:
        cur_len = len(item["pos_input_ids"])
        pad_len = pos_max_len - cur_len
        pos_input_ids.append(item["pos_input_ids"] + [pad_token_id] * pad_len)
        pos_attention_mask.append(item["pos_attention_mask"] + [0] * pad_len)

    # 2. Pad negatives
    neg_max_len = max(len(item["neg_input_ids"]) for item in batch)
    neg_input_ids = []
    neg_attention_mask = []

    for item in batch:
        cur_len = len(item["neg_input_ids"])
        pad_len = neg_max_len - cur_len
        neg_input_ids.append(item["neg_input_ids"] + [pad_token_id] * pad_len)
        neg_attention_mask.append(item["neg_attention_mask"] + [0] * pad_len)

    collated = {
        "pos_input_ids": torch.tensor(pos_input_ids, dtype=torch.long),
        "pos_attention_mask": torch.tensor(pos_attention_mask, dtype=torch.long),
        "neg_input_ids": torch.tensor(neg_input_ids, dtype=torch.long),
        "neg_attention_mask": torch.tensor(neg_attention_mask, dtype=torch.long),
    }

    if "teacher_margin" in batch[0]:
        collated["teacher_margin"] = torch.tensor(
            [item["teacher_margin"] for item in batch],
            dtype=torch.float32
        )
        collated["teacher_pos_score"] = torch.tensor(
            [item["teacher_pos_score"] for item in batch],
            dtype=torch.float32
        )
        collated["teacher_neg_score"] = torch.tensor(
            [item["teacher_neg_score"] for item in batch],
            dtype=torch.float32
        )

    return collated
