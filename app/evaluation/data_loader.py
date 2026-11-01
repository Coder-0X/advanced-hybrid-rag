"""
Standard Cranfield / BEIR dataset loader.
Parses corpus.jsonl, queries.jsonl, and TREC-formatted qrels.tsv.
Eliminates synthetic Python-embedded string constants in favor of standard IR benchmarks.
"""
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.models.schemas import BenchmarkDatasetItem, DocumentInput

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

class DatasetLoader:
    """
    Loads research-standard Cranfield IR datasets.
    Directory structure:
      data/
        raw/
          corpus/
            corpus.jsonl
        testset/
          queries.jsonl
          qrels.tsv
    """
    def __init__(self, data_dir: Optional[str or Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.corpus_path = self.data_dir / "raw" / "corpus" / "corpus.jsonl"
        self.queries_path = self.data_dir / "testset" / "queries.jsonl"
        self.qrels_path = self.data_dir / "testset" / "qrels.tsv"

    def load_corpus(self) -> List[DocumentInput]:
        """Load corpus documents from corpus.jsonl."""
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        documents: List[DocumentInput] = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    doc = DocumentInput(
                        id=data.get("_id") or data.get("id"),
                        title=data.get("title"),
                        text=data.get("text", ""),
                        metadata=data.get("metadata", {})
                    )
                    documents.append(doc)
                except Exception as e:
                    raise ValueError(f"Failed to parse corpus line {line_num}: {e}")
        return documents

    def load_queries(self) -> List[BenchmarkDatasetItem]:
        """Load benchmark queries and ground-truth contexts from queries.jsonl."""
        if not self.queries_path.exists():
            raise FileNotFoundError(f"Queries file not found: {self.queries_path}")

        queries: List[BenchmarkDatasetItem] = []
        with open(self.queries_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    item = BenchmarkDatasetItem(
                        id=data.get("_id") or data.get("id"),
                        query=data.get("text") or data.get("query"),
                        ground_truth_answer=data.get("ground_truth_answer", ""),
                        ground_truth_context_fragments=data.get("ground_truth_context_fragments", []),
                        target_document_id=data.get("target_document_id"),
                        category=data.get("category", "general")
                    )
                    queries.append(item)
                except Exception as e:
                    raise ValueError(f"Failed to parse query line {line_num}: {e}")
        return queries

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        """
        Load TREC-format qrels file.
        Format: <query_id> <iteration/Q0> <doc_id> <relevance_grade>
        Returns: {query_id: {doc_id: grade}}
        """
        if not self.qrels_path.exists():
            raise FileNotFoundError(f"Qrels file not found: {self.qrels_path}")

        qrels: Dict[str, Dict[str, int]] = {}
        with open(self.qrels_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for line_num, row in enumerate(reader, start=1):
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 4:
                    # Support space-separated fallback
                    parts = row[0].split()
                    if len(parts) >= 4:
                        qid, _, doc_id, grade = parts[0], parts[1], parts[2], int(parts[3])
                    else:
                        continue
                else:
                    qid, _, doc_id, grade = row[0].strip(), row[1].strip(), row[2].strip(), int(row[3].strip())
                
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = grade
        return qrels

def load_evaluation_dataset() -> Tuple[List[DocumentInput], List[BenchmarkDatasetItem], Dict[str, Dict[str, int]]]:
    """Helper to load all evaluation assets in one call."""
    loader = DatasetLoader()
    return loader.load_corpus(), loader.load_queries(), loader.load_qrels()
