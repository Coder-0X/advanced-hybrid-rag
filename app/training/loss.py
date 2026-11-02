"""
Ranking and Knowledge Distillation Loss Functions for Neural Rerankers.

Implements:
1. MarginMSELoss (Hofstätter et al., 2020):
   Distills continuous score margins from a large Cross-Encoder teacher into a student:
   L_MarginMSE = (1 / |B|) * sum( ((S(q, d+) - S(q, d-)) - (T(q, d+) - T(q, d-)))^2 )
2. InfoNCELoss:
   Temperature-scaled contrastive ranking loss over positive and mined negative candidates.
3. PairwiseHingeLoss:
   Margin ranking loss with configurable minimum score delta.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginMSELoss(nn.Module):
    """
    Teacher-Student Knowledge Distillation via Margin Mean Squared Error.
    Optimizes the student model to replicate the margin between positive and negative
    passage scores predicted by a powerful teacher Cross-Encoder.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        student_pos_scores: torch.Tensor,
        student_neg_scores: torch.Tensor,
        teacher_pos_scores: torch.Tensor,
        teacher_neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MarginMSE loss over student and teacher predictions.

        Args:
            student_pos_scores: Student model logits for (query, positive) pairs [B] or [B, 1].
            student_neg_scores: Student model logits for (query, negative) pairs [B] or [B, 1].
            teacher_pos_scores: Teacher model logits for (query, positive) pairs [B] or [B, 1].
            teacher_neg_scores: Teacher model logits for (query, negative) pairs [B] or [B, 1].

        Returns:
            Scalar loss tensor.
        """
        student_pos = student_pos_scores.view(-1)
        student_neg = student_neg_scores.view(-1)
        teacher_pos = teacher_pos_scores.view(-1)
        teacher_neg = teacher_neg_scores.view(-1)

        student_margin = student_pos - student_neg
        teacher_margin = teacher_pos - teacher_neg

        loss = F.mse_loss(student_margin, teacher_margin, reduction=self.reduction)
        return loss

    def forward_with_target_margin(
        self,
        student_pos_scores: torch.Tensor,
        student_neg_scores: torch.Tensor,
        target_margin: torch.Tensor
    ) -> torch.Tensor:
        """
        Convenience computation when teacher margin (T_pos - T_neg) is precomputed.
        """
        student_margin = student_pos_scores.view(-1) - student_neg_scores.view(-1)
        target = target_margin.view(-1)
        return F.mse_loss(student_margin, target, reduction=self.reduction)


class InfoNCELoss(nn.Module):
    """
    Temperature-scaled categorical cross-entropy over positive vs negative candidate pools.
    Forces positive document logit to exceed all negative document logits by a large gap.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pos_scores: Positive candidate logits [B] or [B, 1].
            neg_scores: Negative candidate logits [B, N_neg] or [B].

        Returns:
            Scalar InfoNCE loss.
        """
        pos = pos_scores.view(-1, 1) / self.temperature
        if neg_scores.dim() == 1:
            neg = neg_scores.view(-1, 1) / self.temperature
        else:
            neg = neg_scores / self.temperature

        # Concatenate positive at index 0, negatives along dimension 1
        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


class PairwiseHingeLoss(nn.Module):
    """
    Hinge loss: max(0, margin - (S(q, d+) - S(q, d-))).
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> torch.Tensor:
        diff = pos_scores.view(-1) - neg_scores.view(-1)
        return torch.clamp(self.margin - diff, min=0.0).mean()
