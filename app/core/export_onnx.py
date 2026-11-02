"""
ONNX Graph Export and INT8 Dynamic Quantization Pipeline.

Converts HuggingFace PyTorch Cross-Encoder models into optimized ONNX computation graphs
with dynamic sequence and batch axes, followed by dynamic INT8 quantization targeting
x86_64 AVX-512 / VNNI SIMD instruction sets.
"""
import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

logger = logging.getLogger(__name__)


class ONNXExporter:
    """
    Exports Cross-Encoder models to ONNX and performs INT8 dynamic quantization.
    """

    def __init__(self, model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def export_and_quantize(
        self,
        output_dir: str = "models/onnx_reranker",
        opset_version: int = 14
    ) -> Dict[str, Any]:
        """
        Executes end-to-end export: PyTorch FP32 -> ONNX FP32 -> ONNX INT8.

        Args:
            output_dir: Target directory for exported model files.
            opset_version: ONNX operator set version (14 recommended for Transformers).

        Returns:
            Dictionary with paths, model file sizes, and validation metrics.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        fp32_onnx_path = out_path / "model_fp32.onnx"
        int8_onnx_path = out_path / "model_int8.onnx"

        # 1. Load PyTorch model in evaluation mode
        logger.info(f"Loading PyTorch model from {self.model_name_or_path} for ONNX export...")
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        model.eval()

        # Save tokenizer alongside ONNX models for self-contained deployment
        self.tokenizer.save_pretrained(out_path)

        # 2. Prepare dummy calibration input for graph tracing
        dummy_query = "What is the computational complexity of Cormack Reciprocal Rank Fusion?"
        dummy_passage = (
            "Reciprocal Rank Fusion (RRF) evaluates ranked lists across multiple retrieval systems "
            "with O(N log N) sorting overhead, using smoothing constant k = 60."
        )
        dummy_inputs = self.tokenizer(
            dummy_query,
            dummy_passage,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        )

        input_ids = dummy_inputs["input_ids"]
        attention_mask = dummy_inputs["attention_mask"]

        # Run reference PyTorch forward pass
        with torch.no_grad():
            torch_out = model(input_ids, attention_mask=attention_mask).logits.detach().cpu().numpy()

        # 3. Export to ONNX with dynamic batch and sequence axes
        logger.info(f"Tracing PyTorch computational graph to ONNX FP32: {fp32_onnx_path}...")
        try:
            torch.onnx.export(
                model,
                (input_ids, attention_mask),
                str(fp32_onnx_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size"}
                },
                opset_version=opset_version,
                do_constant_folding=True,
                dynamo=False
            )
        except TypeError:
            # Fallback for torch versions that do not accept dynamo parameter
            torch.onnx.export(
                model,
                (input_ids, attention_mask),
                str(fp32_onnx_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size"}
                },
                opset_version=opset_version,
                do_constant_folding=True
            )

        # 4. Verify ONNX FP32 model with ONNX Runtime
        sess_fp32 = ort.InferenceSession(str(fp32_onnx_path), providers=["CPUExecutionProvider"])
        onnx_inputs = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy()
        }
        onnx_fp32_out = sess_fp32.run(["logits"], onnx_inputs)[0]
        max_fp32_diff = float(np.max(np.abs(torch_out - onnx_fp32_out)))
        logger.info(f"ONNX FP32 verification complete. Max absolute logit difference: {max_fp32_diff:.6e}")

        # 5. Execute Dynamic INT8 Quantization
        logger.info(f"Executing Dynamic INT8 Quantization: {int8_onnx_path}...")
        quantize_dynamic(
            model_input=str(fp32_onnx_path),
            model_output=str(int8_onnx_path),
            weight_type=QuantType.QInt8
        )

        # 6. Verify ONNX INT8 model
        sess_int8 = ort.InferenceSession(str(int8_onnx_path), providers=["CPUExecutionProvider"])
        onnx_int8_out = sess_int8.run(["logits"], onnx_inputs)[0]
        max_int8_diff = float(np.max(np.abs(torch_out - onnx_int8_out)))
        logger.info(f"ONNX INT8 verification complete. Max quantization delta: {max_int8_diff:.4f}")

        # Measure file sizes
        fp32_size_mb = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
        int8_size_mb = os.path.getsize(int8_onnx_path) / (1024 * 1024)
        compression_ratio = fp32_size_mb / max(1e-5, int8_size_mb)

        logger.info(
            f"Quantization Successful: FP32 ({fp32_size_mb:.2f} MB) -> "
            f"INT8 ({int8_size_mb:.2f} MB) | Compression Ratio: {compression_ratio:.2f}x"
        )

        return {
            "fp32_onnx_path": str(fp32_onnx_path),
            "int8_onnx_path": str(int8_onnx_path),
            "fp32_size_mb": fp32_size_mb,
            "int8_size_mb": int8_size_mb,
            "compression_ratio": compression_ratio,
            "max_fp32_diff": max_fp32_diff,
            "max_int8_diff": max_int8_diff,
            "tokenizer_dir": str(out_path)
        }


def export_cli():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    exporter = ONNXExporter()
    res = exporter.export_and_quantize()
    print("Export Summary:", res)


if __name__ == "__main__":
    export_cli()
