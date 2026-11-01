"""
Curated Evaluation Corpus and Benchmark Dataset Module.
Loads standard Cranfield / BEIR formatted corpus and query evaluation sets
from data/raw/corpus/ and data/testset/.
"""
from typing import Dict, List
from app.evaluation.data_loader import DatasetLoader
from app.models.schemas import BenchmarkDatasetItem, DocumentInput

_loader = DatasetLoader()

try:
    DEFAULT_CORPUS: List[DocumentInput] = _loader.load_corpus()
    DEFAULT_BENCHMARK_DATASET: List[BenchmarkDatasetItem] = _loader.load_queries()
    DEFAULT_QRELS: Dict[str, Dict[str, int]] = _loader.load_qrels()
except Exception as _e:
    # Fallback to empty if files not yet created during custom builds
    DEFAULT_CORPUS = []
    DEFAULT_BENCHMARK_DATASET = []
    DEFAULT_QRELS = {}
