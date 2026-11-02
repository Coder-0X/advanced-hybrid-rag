"""
PyTorch Distillation and Fine-Tuning Pipeline for Domain-Adapted Neural Rerankers.

Executes knowledge distillation from a teacher Cross-Encoder using MarginMSELoss
over hard-negative passage triplets mined from first-stage retrieval.
"""
import os
import time
import logging
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from app.training.hard_negatives import TeacherScoredTriplet, Triplet
from app.training.dataset import TripletDataset, dynamic_triplet_collate_fn
from app.training.loss import MarginMSELoss, InfoNCELoss

logger = logging.getLogger(__name__)


class RerankerTrainer:
    """
    Distillation training loop optimizing student Cross-Encoder weights.
    """

    def __init__(
        self,
        student_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: Optional[str] = None
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Initializing RerankerTrainer on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(student_model_name, num_labels=1)
        self.model.to(self.device)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.loss_fn = MarginMSELoss()

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any
    ) -> float:
        """Run single training epoch with gradient accumulation and clipping."""
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            pos_input_ids = batch["pos_input_ids"].to(self.device)
            pos_attention_mask = batch["pos_attention_mask"].to(self.device)
            neg_input_ids = batch["neg_input_ids"].to(self.device)
            neg_attention_mask = batch["neg_attention_mask"].to(self.device)

            # Student forward passes
            pos_logits = self.model(pos_input_ids, attention_mask=pos_attention_mask).logits.view(-1)
            neg_logits = self.model(neg_input_ids, attention_mask=neg_attention_mask).logits.view(-1)

            if "teacher_margin" in batch:
                target_margin = batch["teacher_margin"].to(self.device)
                loss = self.loss_fn.forward_with_target_margin(pos_logits, neg_logits, target_margin)
            else:
                teacher_pos = batch["teacher_pos_score"].to(self.device)
                teacher_neg = batch["teacher_neg_score"].to(self.device)
                loss = self.loss_fn(pos_logits, neg_logits, teacher_pos, teacher_neg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / max(1, len(dataloader))

    def fit(
        self,
        train_triplets: List[TeacherScoredTriplet],
        epochs: int = 3,
        batch_size: int = 16,
        warmup_ratio: float = 0.1,
        output_dir: str = "models/distilled_reranker"
    ) -> Dict[str, Any]:
        """
        Execute full training run.

        Args:
            train_triplets: Scored triplets with teacher margins.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            warmup_ratio: Linear warmup proportion.
            output_dir: Destination path for saved model and tokenizer.

        Returns:
            Dictionary with training history and metrics.
        """
        dataset = TripletDataset(
            triplets=train_triplets,
            tokenizer=self.tokenizer,
            max_length=384
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: dynamic_triplet_collate_fn(b, pad_token_id=self.tokenizer.pad_token_id or 0)
        )

        total_steps = len(dataloader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)

        # Separate weight decay for bias / LayerNorm
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        logger.info(f"Starting distillation training for {epochs} epochs ({total_steps} steps).")
        t0 = time.time()
        history = []

        for epoch in range(epochs):
            epoch_loss = self.train_epoch(dataloader, optimizer, scheduler)
            history.append({"epoch": epoch + 1, "loss": epoch_loss})
            logger.info(f"Epoch {epoch + 1}/{epochs} - MarginMSE Loss: {epoch_loss:.5f}")

        duration = time.time() - t0

        # Save student artifacts
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Student model saved to {output_dir} in {duration:.1f}s.")

        return {
            "epochs": epochs,
            "total_steps": total_steps,
            "training_duration_sec": duration,
            "final_loss": history[-1]["loss"] if history else 0.0,
            "history": history,
            "output_dir": output_dir
        }
