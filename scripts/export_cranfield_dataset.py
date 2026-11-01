"""
Export hardcoded dataset from app.evaluation.dataset into standard Cranfield/BEIR format.
Generates:
  - data/raw/corpus/corpus.jsonl
  - data/testset/queries.jsonl
  - data/testset/qrels.tsv (TREC format: query_id Q0 doc_id relevance)
"""
import json
import os
import sys

# Ensure repository root is in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from app.evaluation.dataset import DEFAULT_CORPUS, DEFAULT_BENCHMARK_DATASET

def main():
    corpus_dir = os.path.join(repo_root, "data", "raw", "corpus")
    testset_dir = os.path.join(repo_root, "data", "testset")
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(testset_dir, exist_ok=True)

    corpus_file = os.path.join(corpus_dir, "corpus.jsonl")
    queries_file = os.path.join(testset_dir, "queries.jsonl")
    qrels_file = os.path.join(testset_dir, "qrels.tsv")

    print(f"Exporting {len(DEFAULT_CORPUS)} corpus documents to {corpus_file}...")
    with open(corpus_file, "w", encoding="utf-8") as f:
        for doc in DEFAULT_CORPUS:
            record = {
                "_id": doc.id,
                "title": doc.title or "",
                "text": doc.text,
                "metadata": doc.metadata or {}
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Exporting {len(DEFAULT_BENCHMARK_DATASET)} queries to {queries_file}...")
    with open(queries_file, "w", encoding="utf-8") as f:
        for item in DEFAULT_BENCHMARK_DATASET:
            record = {
                "_id": item.id,
                "text": item.query,
                "ground_truth_answer": item.ground_truth_answer,
                "ground_truth_context_fragments": item.ground_truth_context_fragments,
                "target_document_id": item.target_document_id,
                "category": item.category or "general"
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Exporting TREC qrels to {qrels_file}...")
    # Build category map for documents
    doc_categories = {doc.id: doc.metadata.get("category", "") for doc in DEFAULT_CORPUS}
    
    with open(qrels_file, "w", encoding="utf-8") as f:
        for item in DEFAULT_BENCHMARK_DATASET:
            target_id = item.target_document_id
            target_cat = doc_categories.get(target_id, "")
            
            # Primary target document has grade 2 (definitely relevant)
            if target_id:
                f.write(f"{item.id}\t0\t{target_id}\t2\n")
            
            # Related documents in the same domain category have grade 1 (partially relevant)
            # Adversarial distractors have grade 0 (explicitly non-relevant negative examples)
            for doc in DEFAULT_CORPUS:
                if doc.id == target_id:
                    continue
                doc_cat = doc.metadata.get("category", "")
                if item.category == "adversarial_distractor":
                    # For adversarial queries, other domain docs are negative
                    continue
                elif doc_cat == target_cat and target_cat:
                    # Same domain category doc -> graded 1 (partially relevant background)
                    f.write(f"{item.id}\t0\t{doc.id}\t1\n")

    print("Cranfield dataset export complete.")

if __name__ == "__main__":
    main()
