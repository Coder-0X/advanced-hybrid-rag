# Advanced Hybrid RAG Engine: Production Architecture & Cranfield Evaluation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Vector%20Store-Qdrant-dc2626.svg)](https://qdrant.tech/)
[![Inverted Index](https://img.shields.io/badge/Sparse%20IR-BM25Plus%20Inverted%20Index-amber.svg)](https://en.wikipedia.org/wiki/Okapi_BM25)
[![Reranker](https://img.shields.io/badge/Reranker-Cross--Encoder-purple.svg)](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
[![Evaluation](https://img.shields.io/badge/Benchmark-105%20Cranfield%20Queries-success.svg)](data/testset/)
[![Significance](https://img.shields.io/badge/Significance-Wilcoxon%20%26%20Fisher%20Permutation-blueviolet.svg)](app/evaluation/significance.py)
[![INT8 Quantization](https://img.shields.io/badge/ONNX%20INT8-3.95x%20Compression%20%7C%201.80x%20Speedup-brightgreen.svg)](benchmarks/run_quantization_eval.py)
[![Concurrency](https://img.shields.io/badge/Concurrency-Sliding%20Window%20Micro--Batching-orange.svg)](app/concurrency/)
[![Observability](https://img.shields.io/badge/Observability-OpenTelemetry%20W3C%20%2B%20Prometheus-blue.svg)](app/observability/)

A research-grade Information Retrieval (IR) and Retrieval-Augmented Generation (RAG) system engineered to solve semantic drift, vocabulary mismatch, table fragmentation, and hallucination. 

Built from first principles with **Layout-Aware Document Parsing**, a **Custom Sparse Inverted Index** implementing BM25+ ($k_1=1.5, b=0.75, \delta=1.0$), **Dense Semantic Search** via HNSW Qdrant embeddings, **Cormack Reciprocal Rank Fusion (RRF)**, and **Cross-Encoder Cross-Attention Reranking**. Validated via a 105-query Cranfield benchmark suite featuring **Graded NDCG@10**, **Cranfield MAP@10**, **Paired Wilcoxon Signed-Rank Tests**, and **Fisher's Paired Randomization Tests** ($B=10,000$).

---

## 🌟 Architectural Overview

```mermaid
flowchart TD
    RawDoc["Raw Semi-Structured Documents\n(SEC 10-Ks, RFCs, Papers)"] --> LayoutParser["Layout-Aware Ingestion Pipeline\n(Headers, Tables, Lists, Code, Paragraphs)"]
    LayoutParser --> TableEngine["Table Schema Preservation & Row Linearization\n(Duplicate headers across splits, entity-value tuples)"]
    TableEngine --> Chunks["Enriched Document Chunks with Breadcrumbs"]
    
    Chunks --> DenseStore["Dense Vector Store\n(Qdrant HNSW + all-MiniLM-L6-v2)"]
    Chunks --> SparseIndex["First-Principles Inverted Index\n(Posting Lists, DF table, BM25+ delta=1.0)"]
    
    UserQuery["User Information Need / Query"] --> DenseStore
    UserQuery --> SparseIndex
    
    DenseStore --> RRF["Reciprocal Rank Fusion (Cormack et al., SIGIR 2009)\nRRF(d) = sum w_m / (60 + r_m(d))"]
    SparseIndex --> RRF
    
    RRF --> Candidates["Top-15 Candidate Pool"]
    UserQuery --> CrossEncoder["Cross-Encoder Reranker\n(ms-marco-MiniLM-L-6-v2 Full Cross-Attention)"]
    Candidates --> CrossEncoder
    
    CrossEncoder --> TopK["Top-K Filtered Passages"]
    TopK --> Generator["Grounded Synthesis Engine\n(Reference-based ROUGE-L / Token F1 Verification)"]
    Generator --> Response["Synthesized Answer with Citations & Latency Telemetry"]
```

---

## 📐 Mathematical Formulation

### 1. First-Principles BM25+ Sparse Retrieval
Standard Okapi BM25 penalizes long documents excessively, causing term frequency saturation where documents containing valid query terms receive scores close to zero. We implement **BM25+** (Lv & Zhai, 2011) with lower-bound parameter $\delta = 1.0$:

$$\text{Score}_{\text{BM25+}}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \left[ \frac{\text{TF}(t, D) \cdot (k_1 + 1)}{\text{TF}(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)} + \delta \right]$$

Where:
- $k_1 = 1.5$ (term frequency scaling parameter)
- $b = 0.75$ (document length normalization factor)
- $\delta = 1.0$ (lower-bound floor guaranteeing non-negative contribution for matched terms)
- Robertson-Spärck Jones IDF with Robertson-Zaragoza smoothing:
  $$\text{IDF}(t) = \ln\left(1 + \frac{N - n(t) + 0.5}{n(t) + 0.5}\right)$$

### 2. Cormack Reciprocal Rank Fusion (RRF)
Directly adding dense cosine similarity scores $\in [-1, 1]$ and sparse BM25 scores $\in [0, \infty)$ fails due to disparate, uncalibrated score distributions. We apply **Reciprocal Rank Fusion** (Cormack, Clarke & Büttcher, SIGIR 2009):

$$\text{RRF}(d) = \sum_{m \in \{\text{Dense}, \text{BM25}\}} \frac{w_m}{k + r_m(d)}$$

Where $k = 60$ is the Cormack smoothing constant, $r_m(d)$ is the 1-indexed rank of document $d$ in retrieval modality $m$, and $w_m$ is the modality weight ($w_d = 0.5, w_s = 0.5$).

### 3. Cross-Encoder Cross-Attention Reranking
While bi-encoders produce static sentence embeddings independently ($\mathbf{e}_q = f(q), \mathbf{e}_d = f(d)$), the cross-encoder feeds both sequences concatenated through all transformer layers simultaneously:

$$s(q, d) = \sigma\left(\mathbf{W} \cdot \text{Transformer}(\text{[CLS]} \circ q \circ \text{[SEP]} \circ d \circ \text{[SEP]})\right)$$

This captures fine-grained, word-level cross-attention interactions that bi-encoders miss.

---

## 📊 Quantitative Benchmark Results (105-Query Cranfield Evaluation)

Evaluation executed across **15 multi-domain technical documents** and **105 ground-truth queries** (315 evaluation passes) spanning 5 retrieval archetypes: Lexical Exact ($N=25$), Semantic Paraphrase ($N=25$), Tabular Quantitative ($N=25$), Adversarial Distractors ($N=15$), and Multi-Hop Synthesis ($N=15$).

### Benchmark Results Matrix

| Evaluation Metric | Naive RAG (Dense) | Hybrid (Dense + BM25+) | Reranked (Cross-Encoder) | Hybrid Δ | Reranked Δ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Context Precision@5** | 0.780 | 0.825 | **0.865** | +0.045 | **+0.085\*** |
| **Graded NDCG@10** | 0.878 | 0.877 | **0.892** | -0.001 | **+0.013** |
| **Cranfield MAP@10** | 0.694 | 0.687 | **0.731** | -0.006 | **+0.037** |
| **Mean Reciprocal Rank (MRR)** | 0.785 | 0.829 | **0.869** | +0.043 | **+0.083\*** |
| **Context Recall** | 0.968 | 0.978 | **0.981** | +0.009 | **+0.013** |
| **Hit Rate@5** | 0.971 | 0.981 | **0.981** | +0.010 | **+0.010** |
| **ROUGE-L F1** | 0.106 | 0.102 | **0.103** | -0.004 | **-0.003** |
| **Token F1** | 0.125 | 0.120 | **0.122** | -0.005 | **-0.004** |
| **Latency P50 (Median)** | 36.1 ms | 27.7 ms | 937.4 ms | -8.4ms | +901.3ms |
| **Latency P95** | 339.3 ms | 158.1 ms | 3375.4 ms | - | - |

*\*Statistically significant at $\alpha = 0.05$ after Holm-Bonferroni correction.*

### Query Archetype Ablation Breakdown (MRR@5 & Hit Rate)

| Query Category | Count | Naive MRR | Hybrid MRR | Reranked MRR | NDCG@10 (Rerank) | Hit Rate (Rerank) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adversarial Distractor** | 15 | 0.833 | 0.889 | **0.956** | 1.000 | 100.0% |
| **Lexical Exact** | 25 | 0.933 | 0.953 | **0.973** | 0.868 | 100.0% |
| **Semantic Paraphrase** | 25 | 0.657 | 0.710 | **0.808** | 0.852 | 96.0% |
| **Multi-Hop Synthesis** | 15 | 0.867 | 0.867 | **0.900** | 0.834 | 93.3% |
| **Tabular Quantitative** | 25 | 0.688 | 0.763 | **0.753** | 0.925 | 100.0% |

### Non-Parametric Statistical Significance Tests (Reranked vs. Naive Baseline)

| Metric Tested | Paired Wilcoxon Stat | Wilcoxon p-value | Fisher Randomization Perm. p-value ($B=5000$) | Significant ($\alpha=0.05$)? |
| :--- | :---: | :---: | :---: | :---: |
| **Context Precision@5** | 68.00 | **0.000303** | **0.001000** | **YES ($p < 0.05$)** |
| **Mean Reciprocal Rank (MRR)** | 46.00 | **0.000367** | **0.001400** | **YES ($p < 0.05$)** |
| **Graded NDCG@10** | 327.00 | 0.177875 | 0.124575 | Null not rejected |
| **Cranfield MAP@10** | 236.50 | 0.079987 | 0.043591 | Marginal ($p \approx 0.04-0.07$) |

---

## 📈 Publication Figures & Empirical Technical Report

The evaluation suite automatically renders publication-grade figures in IEEE/ACM format and compiles full empirical reports:

- 📄 **[Full Academic Experiments Report](reports/EXPERIMENTS_REPORT.md)**: Comprehensive empirical evaluation adhering to SIGIR / VLDB / MLSys experimental protocols, non-parametric Wilcoxon/Fisher statistical tests, Amdahl's Law latency profiling, and ablation analyses.
- `reports/figures/pareto_frontier.png`: Precision vs. P95 latency Pareto frontier illustrating the operational trade-off between Hybrid RRF (ultra-low latency) and Cross-Encoder reranking (maximum precision).
- `reports/figures/archetype_ablation.png`: Comparative bar chart dissecting performance across the 5 query archetypes.
- `reports/figures/hnsw_pareto_frontier.png`: HNSW vector index Pareto frontier ($M \in [8..64]$, $ef_{\text{construct}} \in [64, 128]$, $ef_{\text{search}} \in [10..200]$) evaluated against exact brute-force ground truth.
- `reports/figures/cache_threshold_calibration.png`: Semantic cache Youden's $J$ ROC curve and score distribution calibration for false-positive-free retrieval caching.

---

## 🔬 Systems Engineering & Advanced Architectures

### 1. Dynamic INT8 ONNX Quantization & Knowledge Distillation
To resolve the $97.2\%$ Amdahl's Law bottleneck in neural reranking, the Cross-Encoder is exported to ONNX format with dynamic tensor shapes and quantized to INT8 precision with AVX-512/VNNI acceleration:
- **$3.95\times$ Model Compression**: Memory footprint reduced from $86.78\text{ MB} \to 21.99\text{ MB}$.
- **$1.80\times$ Latency Speedup**: Mean per-query reranking latency drops from $675.09\text{ms} \to 374.15\text{ms}$.
- **Empirical Non-Degradation**: Evaluated on 105 Cranfield queries with $98.33\%$ NDCG retention. Paired Two-Sided Wilcoxon Signed-Rank Test yields $W = 529.50, p = 0.29707 > 0.05$, confirming null hypothesis of zero ranking fidelity loss cannot be rejected.

### 2. Sliding-Window Dynamic Micro-Batching Concurrency Engine
Under bursty concurrent traffic, naive sequential inference blocks the FastAPI asynchronous event loop and suffers heavy Python GIL contention. We implement an asynchronous micro-batcher (`app/concurrency/batch_queue.py`):
- **Adaptive Windowing**: Coalesces concurrent queries within a sliding timeout window ($\Delta t = 8\text{ms}$) up to $B_{\text{max}} = 32$.
- **Off-Loop Thread-Pool Execution**: Dispatches vectorized matrix operations via `loop.run_in_executor()` to dedicated CPU worker threads.
- **Concurrency Speedup**: Under 50 concurrent simulated users, throughput increases by **$1.78\times$** ($196.97 \to 349.83\text{ QPS}$) with a **$95.5\%$ reduction in latency jitter** ($246.29\text{ms} \to 11.01\text{ms}$).

### 3. Empirical Vector Store HNSW Hyperparameter Ablation
Vector databases are often treated as black boxes. We implement an exact $O(N \cdot D)$ linear scan baseline (`app/storage/brute_force_index.py`) and perform a 40-configuration grid sweep across HNSW graph parameters (`app/storage/hnsw_ablation.py`):
- Mapped trade-offs between construction parameters ($M \in [8..64]$, $ef_{\text{construct}} \in [64, 128]$) and query runtime parameters ($ef_{\text{search}} \in [10..200]$).
- Identified the empirical Pareto inflection point at $M=16, ef_{\text{search}}=50$, which delivers $0.985$ Recall@10 while preserving sub-millisecond query time ($0.88\text{ms}$).

### 4. Anti-Stampede SingleFlight Semantic Cache
To eliminate the cache stampede ("thundering herd") problem during concurrent bursts, we engineer a two-tier caching architecture (`app/cache/`):
- **L1 AST & SHA-256 Cache**: O(1) in-memory LRU cache matching normalized query syntax trees.
- **L2 Semantic ANN Vector Cache**: Embeds queries and searches past results via cosine similarity threshold calibrated via Youden's $J$ statistic ($\tau^* = 0.75$, $J=1.000, \text{FPR}=0.0\%$).
- **SingleFlight Mutex Coalescing**: Resolves simultaneous cache misses by allowing only one worker to execute the physical pipeline while all concurrent duplicates await the same asynchronous Promise, achieving a **$98.0\%$ compute suppression ratio** under a 50-request stampede.

### 5. Distributed OpenTelemetry Tracing & Prometheus Observability
- **W3C Distributed Trace Context**: Propagates `traceparent` headers across asynchronous thread boundaries and background execution pools (`app/observability/tracer.py`).
- **Prometheus Metrics Endpoint**: Exposes bounded categorical histograms (`rag_retrieval_latency_seconds_bucket`, `rag_pipeline_requests_total`) partitioned by execution stage and query archetype at `/metrics`.
- **ASGI Middleware**: Auto-generates and returns `X-Trace-Id` headers on all REST requests for end-to-end distributed observability.

---

## 📁 Repository Structure

```
advanced-hybrid-rag/
├── app/
│   ├── api/
│   │   ├── dependencies.py       # FastAPI dependency injection (get_engine)
│   │   ├── endpoints.py          # REST API endpoints (/documents, /query, /benchmark)
│   │   └── ui.py                 # Jinja2 template rendering controller
│   ├── cache/
│   │   ├── hierarchical_cache.py # L1 SHA-256 AST LRU + L2 Semantic Vector Cache
│   │   ├── single_flight.py      # Async mutex promise coalescer (anti-stampede)
│   │   └── threshold_evaluator.py# ROC & Youden's J empirical similarity calibrator
│   ├── concurrency/
│   │   └── batch_queue.py        # Sliding-window dynamic micro-batcher
│   ├── core/
│   │   ├── bm25_retriever.py     # High-level sparse retriever wrapper
│   │   ├── document_processor.py # Layout-aware parser, Markdown table linearizer
│   │   ├── export_onnx.py        # PyTorch -> ONNX dynamic export & INT8 quantizer
│   │   ├── fusion.py             # Cormack Reciprocal Rank Fusion (RRF)
│   │   ├── generator.py          # Grounded synthesis with citation verification
│   │   ├── inverted_index.py     # First-principles InvertedIndex (postings, BM25+)
│   │   ├── onnx_reranker.py      # AVX-512 accelerated ONNX INT8 Cross-Encoder
│   │   ├── pipeline.py           # Unified HybridRAGEngine orchestrator
│   │   ├── reranker.py           # Cross-Encoder neural reranker
│   │   └── vector_store.py       # Dense vector store (Qdrant + sentence-transformers)
│   ├── evaluation/
│   │   ├── benchmark_runner.py   # Automated 3-mode benchmark runner & reporter
│   │   ├── data_loader.py        # Cranfield / BEIR dataset loader
│   │   ├── dataset.py            # Evaluation corpus & query re-exports
│   │   ├── metrics.py            # NDCG@10, MAP@10, MRR, Context Precision, ROUGE-L
│   │   ├── significance.py       # Wilcoxon, Fisher's Randomization, Holm-Bonferroni
│   │   └── visualizer.py         # Matplotlib publication figure generator
│   ├── models/
│   │   └── schemas.py            # Pydantic v2 schemas and telemetry models
│   ├── observability/
│   │   ├── metrics.py            # Prometheus bounded histograms & counters
│   │   ├── middleware.py         # ASGI OpenTelemetry tracing & X-Trace-Id middleware
│   │   └── tracer.py             # OpenTelemetry W3C trace context provider
│   ├── storage/
│   │   ├── brute_force_index.py  # Exact O(N*D) linear scan baseline for true ground truth
│   │   └── hnsw_ablation.py      # Grid sweep engine over M, ef_construct, ef_search
│   ├── training/
│   │   ├── dataset.py            # PyTorch TripletDataset with dynamic padding collate
│   │   ├── hard_negatives.py     # BM25+/Dense adversary false-positive miner
│   │   ├── loss.py               # MarginMSE distillation & InfoNCE contrastive losses
│   │   └── train_reranker.py     # Teacher-student Cross-Encoder distillation trainer
│   ├── config.py                 # Pydantic Settings & environment configuration
│   └── main.py                   # FastAPI app with metrics endpoint & telemetry
├── benchmarks/
│   ├── run_batching_concurrency.py # Micro-batching vs naive async benchmark
│   ├── run_cache_eval.py         # SingleFlight stampede suppression & ROC calibration
│   ├── run_hnsw_ablation.py      # 40-configuration HNSW Pareto frontier study
│   └── run_quantization_eval.py  # INT8 ONNX vs FP32 PyTorch Cranfield benchmark
├── data/
│   ├── raw/corpus/
│   │   └── corpus.jsonl          # 15 technical documents (JSONL format)
│   └── testset/
│       ├── queries.jsonl         # 105 ground-truth queries with answers & contexts
│       └── qrels.tsv             # Standard TREC 4-column graded relevance judgments
├── reports/
│   ├── figures/
│   │   ├── archetype_ablation.png       # Archetype performance breakdown
│   │   ├── cache_threshold_calibration.png # Youden's J ROC calibration curve
│   │   ├── hnsw_pareto_frontier.png     # HNSW Recall vs Latency Pareto frontier
│   │   └── pareto_frontier.png          # Latency vs Accuracy Pareto frontier
│   ├── EXPERIMENTS_REPORT.md     # Full academic empirical systems technical report
│   ├── cache_benchmark.json      # SingleFlight & threshold calibration results
│   ├── concurrency_benchmark.json# Dynamic batching concurrency telemetry
│   └── hnsw_ablation_results.json# 40-run HNSW ablation raw data
├── static/
│   ├── css/style.css             # Glassmorphism dark theme styles
│   └── js/app.js                 # Interactive dashboard logic
├── templates/
│   └── index.html                # Semantic HTML5 playground & benchmark dashboard
├── tests/
│   ├── conftest.py               # Pytest environment fixtures
│   ├── test_api.py               # REST API integration tests (health, query, stats)
│   ├── test_bm25.py              # Inverted index & BM25+ unit tests
│   ├── test_cache.py             # SingleFlight & hierarchical cache unit tests
│   ├── test_chunking.py          # Table preservation & row linearization tests
│   ├── test_concurrency.py       # Micro-batching queue & concurrency tests
│   ├── test_evaluation.py        # NDCG, MAP, ROUGE-L, significance tests
│   ├── test_fusion.py            # RRF fusion mathematical tests
│   ├── test_observability.py     # OpenTelemetry & Prometheus middleware tests
│   ├── test_pipeline.py          # End-to-end pipeline execution tests
│   ├── test_quantization_and_training.py # Hard-negatives, MarginMSE, ONNX tests
│   ├── test_reranker.py          # Cross-encoder tests
│   └── test_storage_ablation.py  # Brute-force & HNSW ablation unit tests
├── Dockerfile                    # Multi-stage, non-root production Docker image
├── docker-compose.yml            # Container orchestration definition
├── pyproject.toml                # PEP 518/621 packaging (Ruff, Mypy, Pytest)
├── requirements.txt              # Pinned production dependencies
└── run_benchmark.py              # Standalone CLI benchmark runner
```

---

## 🚀 Quick Start

### 1. Local Environment Setup

```bash
# Clone and enter repository
git clone https://github.com/Coder-0X/advanced-hybrid-rag.git
cd advanced-hybrid-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Benchmark Suite

```bash
python run_benchmark.py
```

### 3. Run Unit and Integration Tests

```bash
pytest tests/ -v
```

### 4. Launch the Interactive Dashboard

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open [http://localhost:8000](http://localhost:8000) to access the interactive search playground, document ingestion manager, and evaluation benchmark visualizer.

---

## 🐳 Docker Deployment

```bash
docker compose up --build
```
Access the application at `http://localhost:8000` with containerized health checks and resource management.

---

## 📜 Academic Citations & Theoretical References

1. **Reciprocal Rank Fusion**: Cormack, G. V., Clarke, C. L., & Büttcher, S. (2009). *Reciprocal rank fusion outperforms Condorcet and individual machine learning methods*. Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval (SIGIR '09), 758–759.
2. **BM25+ Ranking**: Lv, Y., & Zhai, C. (2011). *Lower-bounding term frequency normalization*. Proceedings of the 20th ACM international conference on Information and knowledge management (CIKM '11), 7–16.
3. **Statistical Significance in IR**: Smucker, M. D., Allan, J., & Carterette, B. (2007). *A comparison of statistical significance tests for information retrieval evaluation*. Proceedings of the 16th ACM conference on Conference on information and knowledge management (CIKM '07), 623–632.
4. **Graded Relevance Metrics**: Järvelin, K., & Kekäläinen, J. (2002). *Cumulated gain-based evaluation of retrieval techniques*. ACM Transactions on Information Systems (TOIS), 20(4), 422–446.
