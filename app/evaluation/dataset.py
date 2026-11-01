"""
Curated Evaluation Corpus and Benchmark Dataset.
Contains 15 comprehensive technical documents and 105 systematically crafted queries
spanning Lexical Exact, Semantic Paraphrase, Tabular Quantitative, Adversarial Distractor,
and Multi-Hop Synthesis retrieval archetypes.
"""
from typing import List
from app.models.schemas import BenchmarkDatasetItem, DocumentInput

DEFAULT_CORPUS: List[DocumentInput] = [
    DocumentInput(
        id="doc_sec_10k_balance_sheet",
        title="SEC Form 10-K: Consolidated Balance Sheets and Liquidity Ratios",
        text="""# Item 8: Financial Statements - Consolidated Balance Sheets
The following consolidated balance sheets detail the financial condition of the Corporation as of fiscal year-end 2024 and 2025 (in millions of USD):

| Balance Sheet Item | FY 2024 (USD M) | FY 2025 (USD M) | Classification |
|---|---|---|---|
| Cash and cash equivalents | 29965 | 34812 | Current Asset |
| Marketable securities | 31580 | 38420 | Current Asset |
| Accounts receivable net | 29500 | 32150 | Current Asset |
| Inventories | 6520 | 7110 | Current Asset |
| Total current assets | 97565 | 112492 | Current Asset |
| Property plant and equipment net | 43715 | 51200 | Non-Current Asset |
| Total assets | 141280 | 163692 | Total Asset |
| Accounts payable | 54120 | 59800 | Current Liability |
| Accrued expenses | 28450 | 31220 | Current Liability |
| Total current liabilities | 82570 | 91020 | Current Liability |
| Long-term debt | 95280 | 106630 | Non-Current Liability |
| Total stockholders equity | -36570 | -33958 | Equity Deficit |

## Liquidity Ratio Analysis
The corporate current ratio increased from 1.182 in fiscal 2024 to 1.236 in fiscal 2025, driven by expansion in marketable securities.
The quick ratio (cash, marketable securities, and accounts receivable divided by total current liabilities) was 1.103 in 2024 and expanded to 1.158 in 2025.
Working capital totaled 21472 million USD at the close of fiscal 2025, compared to 14995 million USD in 2024.""",
        metadata={"category": "finance_regulatory"}
    ),
    DocumentInput(
        id="doc_sec_10k_leases_commitments",
        title="SEC Form 10-K Note 14: Operating Leases and Cloud Commitments",
        text="""# Note 14: Operating Leases and Infrastructure Commitments
The company holds substantial operating leases for data center colocation facilities, office space, and manufacturing sites.

| Fiscal Maturity Year | Undiscounted Lease Payments (USD M) | Cloud Commitments (USD M) |
|---|---|---|
| 2026 | 11240 | 3450 |
| 2027 | 9810 | 3800 |
| 2028 | 7450 | 3200 |
| 2029 | 5120 | 2500 |
| Thereafter | 14600 | 1550 |
| Total Undiscounted Payments | 48220 | 14500 |

## Discounting and Present Value
Total undiscounted operating lease payments equal 48220 million USD. After deducting imputed interest of 6140 million USD calculated using a weighted-average discount rate of 3.85%, the present value of operating lease liabilities recorded on the balance sheet is 42080 million USD.

## Cloud Infrastructure Agreements
Under multi-year take-or-pay agreements with cloud service providers (AWS and Google Cloud Platform), the company has committed to spend a cumulative minimum of 14500 million USD through fiscal 2030. If annual minimum compute and bandwidth thresholds are not achieved, an 18% contractual underutilization penalty applies to unconsumed capacity.""",
        metadata={"category": "finance_regulatory"}
    ),
    DocumentInput(
        id="doc_sec_10k_segment_revenue",
        title="SEC Form 10-K Note 19: Segment Operations and Geographic Margins",
        text="""# Note 19: Segment Reporting and Regional Net Sales
The Corporation reports operating results across four geographic segments: Americas, Europe, Greater China, and Rest of Asia Pacific.

| Geographic Segment | FY 2024 Net Sales (USD M) | FY 2025 Net Sales (USD M) | Operating Margin |
|---|---|---|---|
| Americas | 150200 | 162500 | 37.4% |
| Europe | 88400 | 94800 | 34.1% |
| Greater China | 68200 | 72600 | 41.2% |
| Rest of Asia Pacific | 28900 | 31200 | 32.8% |
| Total Consolidated Net Sales | 335700 | 361100 | 36.8% |

## Revenue by Product Category
Consolidated net sales in fiscal 2025 were composed of Hardware Products totaling 265800 million USD (73.6% of net sales) and Services revenue totaling 95300 million USD (26.4% of net sales).
Gross margin for Services was 73.8% compared to 36.5% for Hardware Products, reflecting the high operating leverage of digital ecosystem subscriptions and developer fee structures.""",
        metadata={"category": "finance_regulatory"}
    ),
    DocumentInput(
        id="doc_k8s_autoscaling",
        title="Kubernetes Pod Autoscaling Mechanics: HPA and VPA Architecture",
        text="""# Kubernetes Pod Autoscaling Mechanics
Kubernetes implements the Horizontal Pod Autoscaler (HPA) to dynamically adjust pod replica counts based on observed CPU utilization, memory consumption, or custom Prometheus metrics.

## HPA Scaling Formula
The desired replica count is deterministically computed according to the following mathematical formula:
desiredReplicas = ceil[currentReplicas * (currentMetricValue / desiredMetricValue)]

## Stabilization and Flapping Prevention
HPA queries the cluster Metrics Server periodically every 15 seconds.
To prevent rapid flapping or thrashing during transient load drops, HPA enforces a configurable scale-down stabilization window with a default duration of 300 seconds.
Scale-up operations are governed by rate-limiting constraints allowing a maximum expansion of 100% or 4 pods per 15-second evaluation interval.

## Vertical Pod Autoscaler (VPA) Modes
While HPA adjusts horizontal replicas, Vertical Pod Autoscaler (VPA) modifies container CPU and memory resource requests. VPA operates in four distinct execution modes:
- Off: Computes recommended container requests without applying changes.
- Initial: Assigns recommended resource requests only at pod creation time.
- Recreate: Evicts and restarts active pods if current allocations deviate significantly from recommendations.
- Auto: Automatically updates resource requests during runtime.""",
        metadata={"category": "distributed_systems"}
    ),
    DocumentInput(
        id="doc_k8s_networking_cni",
        title="Kubernetes CNI, Kube-Proxy Modes, and Service Mesh Routing",
        text="""# Kubernetes Networking Architecture: CNI, Kube-Proxy, and CoreDNS
Container Network Interface (CNI) plugins provide cross-node pod-to-pod networking where every pod receives a unique, routable cluster IP address.

## Kube-Proxy Implementation Modes
Kube-proxy maintains internal cluster networking rules across worker nodes. It operates in two primary modes:
1. iptables Mode: Sequential rule traversal. Packet filtering scales at O(n) latency as cluster service counts increase, leading to high CPU consumption when clusters exceed 5000 services.
2. IPVS Mode: Leverages Linux kernel IP Virtual Server using ipset hash tables, providing O(1) packet lookup latency regardless of service count.

## Internal Service Discovery via CoreDNS
Internal service discovery maps hostnames to ClusterIP addresses. Standard fully-qualified domain names (FQDN) follow the canonical format:
service-name.namespace.svc.cluster.local

## Next-Generation eBPF Networking
Modern plugins such as Cilium replace kube-proxy entirely by attaching extended Berkeley Packet Filter (eBPF) programs to Linux kernel socket operations (sockops), bypassing the kernel TCP/IP network stack for node-local inter-pod communication.""",
        metadata={"category": "distributed_systems"}
    ),
    DocumentInput(
        id="doc_raft_consensus",
        title="Raft Distributed Consensus Protocol: Leader Election and Safety",
        text="""# Raft Distributed Consensus Protocol
Raft is an understandability-centric distributed consensus algorithm that guarantees state machine safety across asynchronous networks with fail-stop failures.

## Leader Election and Timers
A Raft cluster comprises nodes in one of three states: Leader, Follower, or Candidate.
Elections are initiated when a Follower's election timer expires without receiving heartbeats.
To avoid split-vote deadlocks, Raft uses randomized election timeouts uniformly distributed between 150ms and 300ms.
The leader maintains authority by broadcasting periodic heartbeats via AppendEntries RPCs at an interval of HeartbeatTimeout = 50ms.

## RPC Specification and Quorums
Raft relies on two fundamental Remote Procedure Calls:
- RequestVote RPC: Invoked by Candidate nodes to gather majority votes.
- AppendEntries RPC: Invoked by the Leader for log replication and heartbeat transmission.
Log entries are committed only after successful replication across a strict majority quorum: QuorumSize = floor(N / 2) + 1.

## Raft Invariants and Election Safety
- Election Safety: At most one leader can be elected in a given term.
- Leader Append-Only: A leader never overwrites or truncates its own log entries.
- Log Matching Property: If two logs contain an entry with the same index and term, they are identical up to that index.""",
        metadata={"category": "distributed_systems"}
    ),
    DocumentInput(
        id="doc_bm25_information_retrieval",
        title="Okapi BM25 and BM25Plus Lexical Ranking Formulations",
        text="""# Okapi BM25 and BM25Plus Lexical Ranking
Okapi BM25 is a non-linear term-matching ranking function derived from probabilistic relevance modeling.

## Mathematical Formulation
For a document D and query Q containing terms q_i, the BM25 relevance score is calculated as:
Score(D, Q) = sum(IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * (|D| / avgdl))))

## Hyperparameter Dynamics
- k1 (Term Frequency Saturation): Governs the rate at which term frequency saturates. Typical values range between 1.2 and 2.0. As k1 approaches zero, term frequency contribution vanishes, reducing the function to pure IDF matching.
- b (Document Length Normalization): Controls the penalty applied to long documents. When b equals 1.0, document length is fully normalized against avgdl; when b equals 0.0, length normalization is completely disabled.

## The BM25Plus Extension
Standard BM25 suffers when document length |D| is exceptionally large: long documents with a single term occurrence can receive scores arbitrarily close to zero.
BM25Plus remedies this by adding a lower-bound smoothing constant delta = 1.0:
BM25Plus_Score = BM25_Score + delta
This guarantees that any document containing a matching query token receives a strictly non-zero score regardless of total document length.""",
        metadata={"category": "information_retrieval"}
    ),
    DocumentInput(
        id="doc_qdrant_hnsw_indexing",
        title="Hierarchical Navigable Small World (HNSW) Graphs in Qdrant Vector Search",
        text="""# Hierarchical Navigable Small World (HNSW) Indexing
Qdrant uses HNSW multi-layer graph indexing to perform sub-millisecond Approximate Nearest Neighbor (ANN) search over dense vector embeddings.

## Multi-Layer Graph Architecture
HNSW organizes vectors into a hierarchy of graph layers:
- Top Layers: Sparse graphs containing long-distance links that enable logarithmic O(log N) geometric exploration across the vector space.
- Layer 0: Dense bottom layer containing all indexed vector points connected by short-range local links for fine-grained distance evaluation.

## Indexing and Search Hyperparameters
1. m (Maximum Edges per Node): Number of bidirectional links established per vertex, typically tuned between 16 and 64. Higher m improves recall on high-dimensional vectors at the cost of higher RAM consumption.
2. ef_construct (Construction Beam Width): The size of the dynamic candidate list evaluated during graph construction, typically set between 100 and 512. Higher ef_construct yields superior graph connectivity.
3. ef_search (Query-Time Beam Width): The size of the priority queue evaluated during query time. Increasing ef_search improves recall without rebuilding the index.

## Distance Metric Formulations
Qdrant natively evaluates distance using Cosine distance (1 - cos_sim), Squared Euclidean L2 distance, and Dot Product for unit-normalized vectors.""",
        metadata={"category": "information_retrieval"}
    ),
    DocumentInput(
        id="doc_rrf_fusion_algorithm",
        title="Cormack Reciprocal Rank Fusion (RRF) for Hybrid IR",
        text="""# Reciprocal Rank Fusion (RRF) for Hybrid Information Retrieval
Reciprocal Rank Fusion is an unsupervised score-aggregation algorithm formulated by Cormack, Clarke, and Buettcher in 2009.

## The Calibration Problem in Hybrid Search
Linear combination of retrieval scores fails because dense cosine similarity scores [0.0, 1.0] and sparse BM25 scores [0.0, inf) exhibit incompatible probability distributions and scale variances.

## Mathematical Formulation
RRF overcomes distribution divergence by ignoring raw scores entirely and operating solely on monotonic ordinal ranks:
RRF_Score(d) = sum_{m in {dense, sparse}} (w_m / (k + rank_m(d)))

## Hyperparameters and Properties
- k = 60: The canonical Cormack smoothing constant. Setting k = 60 ensures that top-1 rankings do not excessively dominate positions 2 through 5, while preventing low-ranked tail results from distorting aggregate scores.
- w_dense, w_sparse: Relative channel weights, conventionally set to 0.5 and 0.5.
- Monotonic Stability: If document A ranks ahead of document B across all retrieval systems, RRF mathematically guarantees that A will rank ahead of B in the fused output.""",
        metadata={"category": "information_retrieval"}
    ),
    DocumentInput(
        id="doc_cross_encoders_reranking",
        title="Cross-Encoder Joint Attention Architectures vs Bi-Encoder Encoders",
        text="""# Cross-Encoder Rerankers vs Bi-Encoder Retrieval
In modern two-stage Information Retrieval architectures, bi-encoders and cross-encoders fulfill complementary roles.

## Bi-Encoder Information Bottleneck
Bi-encoders (e.g., all-MiniLM-L6-v2, BGE-small) encode queries and document passages independently into separate fixed-size dense vectors u and v.
Similarity is computed via dot product: sim(q, d) = u . v.
Because query tokens cannot attend to document tokens during transformer forward passes, bi-encoders suffer from semantic compression bottlenecks and cannot detect subtle lexical contradictions.

## Cross-Encoder Bidirectional Self-Attention
Cross-encoders concatenate query q and candidate passage p into a unified transformer sequence:
Input = [CLS] query [SEP] passage [SEP]
Every query token evaluates full bidirectional cross-attention with every passage token through 6 to 24 transformer layers according to:
Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V

## The Two-Stage Hybrid Topology
1. First Stage: Rapid retrieval of 50 to 100 candidate chunks via dense vector search and sparse BM25 combined with RRF (latency: 15-25ms).
2. Second Stage: Cross-Encoder (e.g., ms-marco-MiniLM-L-6-v2) computes joint cross-attention logits over the candidates, filtering semantic distractors and false positives (latency: 30-50ms).""",
        metadata={"category": "deep_learning"}
    ),
    DocumentInput(
        id="doc_database_lsm_btree",
        title="Database Storage Engines: LSM-Trees vs B+Trees and Write Amplification",
        text="""# Database Storage Engines: Log-Structured Merge-Trees vs B+Trees
Storage engines dictate the trade-offs between write amplification, read latency, and disk fragmentation.

## LSM-Tree Architecture
Log-Structured Merge-trees (LSM-trees), utilized in RocksDB and Cassandra, optimize for sequential write throughput:
1. MemTable: In-memory write buffer implemented via a concurrent skiplist.
2. Write-Ahead Log (WAL): Sequential append-only disk log providing durability across crashes.
3. SSTables (Sorted String Tables): Immutable on-disk files organized into hierarchical levels L0 through Lk.

## Compaction Strategies and Amplification
- Leveled Compaction: Each level has a fixed capacity (e.g. 10x growth factor). SSTables in L1+ contain non-overlapping key ranges, minimizing Read Amplification at the cost of higher Write Amplification Factor (WAF typically 10 to 30).
- Size-Tiered Compaction: Flushes files of similar sizes together, reducing write amplification for write-intensive workloads at the expense of higher temporary disk space overhead.

## B+Tree Comparison
B+Trees, utilized in PostgreSQL and MySQL InnoDB, maintain balanced tree structures with in-place page updates. B+Trees provide predictable O(log N) read latency and point lookups but suffer from random I/O write amplification when dirty pages are flushed to storage.""",
        metadata={"category": "database_internals"}
    ),
    DocumentInput(
        id="doc_database_acid_isolation",
        title="ACID Transaction Isolation Anomalies and PostgreSQL Snapshot Isolation",
        text="""# Transaction Isolation Levels and Concurrency Anomalies
The ANSI SQL-92 standard defines isolation levels to prevent concurrency phenomena across multi-threaded transactions.

## Concurrency Anomalies Taxonomy
1. Dirty Read (G1): Transaction T2 reads uncommitted modifications made by T1.
2. Non-Repeatable Read (G2a): Transaction T1 re-reads a row and discovers modified values committed by T2.
3. Phantom Read (A3): Transaction T1 executes a range query and discovers newly inserted rows committed by T2 that satisfy the search predicate.
4. Write Skew (A5B): Occurs when two concurrent transactions read overlapping datasets, verify invariant constraints independently, and perform non-overlapping writes that collectively violate the global constraint.

## PostgreSQL MVCC and Snapshot Isolation
PostgreSQL implements Multi-Version Concurrency Control (MVCC) where rows are never modified in place:
- Each tuple header contains xmin (transaction ID that inserted the tuple) and xmax (transaction ID that deleted or updated the tuple).
- Repeatable Read isolation prevents phantom reads using a single transaction snapshot taken at transaction start.
- To eliminate Write Skew without table-level locking, PostgreSQL uses Serializable Snapshot Isolation (SSI), which dynamically tracks read-write conflicts (rw-antidependencies) in an in-memory lock graph to detect dependency cycles.""",
        metadata={"category": "database_internals"}
    ),
    DocumentInput(
        id="doc_llm_flashattention",
        title="FlashAttention: Fast and Memory-Efficient Exact Attention via Tiling",
        text="""# FlashAttention: Fast and Memory-Efficient Exact Attention
FlashAttention (Dao et al., 2022) resolves the fundamental memory-bandwidth bottleneck of self-attention on modern GPU hardware.

## The Memory Hierarchy Bottleneck
Standard scaled dot-product attention computes:
S = Q * K^T,   P = softmax(S),   O = P * V
On modern GPUs (e.g., NVIDIA A100/H100), computing and storing the intermediate attention matrix S and P of size N x N in High-Bandwidth Memory (HBM, ~1.5 to 3.0 TB/s) is memory-bound. The GPU compute cores (Tensor Cores) stall while waiting for HBM memory transfers.

## Tiling and Online Softmax
FlashAttention avoids materializing the N x N attention matrix in HBM by loading blocks of Q, K, and V into fast on-chip SRAM (~19 TB/s bandwidth):
1. Tiling: Divides the Q, K, and V matrices into blocks that fit entirely within SRAM cache.
2. Online Softmax: Utilizes the Milakov & Gimelshein online softmax trick to incrementally compute softmax scaling factors without needing the global row maximum in advance.
3. Recomputation: In the backward pass, FlashAttention does not read stored attention matrices from HBM; instead, it recomputes attention on-the-fly in fast SRAM, yielding 2x to 4x wall-clock speedups.""",
        metadata={"category": "deep_learning"}
    ),
    DocumentInput(
        id="doc_llm_speculative_decoding",
        title="Speculative Decoding and KV-Cache Memory Bandwidth Optimization",
        text="""# Speculative Decoding and KV-Cache Architecture
Autoregressive token generation in Large Language Models is strictly memory-bandwidth bound rather than compute-bound.

## Arithmetic Intensity Bottleneck
During autoregressive generation, generating a single token requires transferring all model weights (e.g., 70 billion parameters = 140 GB in FP16) from GPU HBM into compute registers.
Because each forward pass performs only a single matrix-vector multiplication, arithmetic intensity is less than 1 FLOP per byte, severely underutilizing GPU Tensor Cores.

## Speculative Decoding Mechanics
Speculative Decoding (Leviathan et al., Chen et al.) pairs a small, fast draft model with a large target model:
1. Draft Generation: The draft model autoregressively generates K speculative tokens (e.g., K = 4 to 8) at high speed.
2. Parallel Verification: The target model evaluates all K tokens simultaneously in a single parallel forward pass.
3. Modified Rejection Sampling: Tokens are accepted or rejected based on the probability ratio P_target(x) / P_draft(x), mathematically guaranteeing that the sampled distribution matches the target model exactly.

## PagedAttention and Memory Fragmentation
PagedAttention (vLLM) resolves Key-Value (KV) cache memory waste by allocating KV tensors into non-contiguous physical memory blocks, reducing memory fragmentation from over 60% down to under 4%.""",
        metadata={"category": "deep_learning"}
    ),
    DocumentInput(
        id="doc_llm_qlora_quantization",
        title="QLoRA: Efficient Finetuning of Quantized LLMs via 4-bit NormalFloat",
        text="""# QLoRA: Efficient Finetuning of Quantized LLMs
QLoRA (Dettmers et al., 2023) reduces the memory footprint of LLM fine-tuning to enable training a 65B parameter model on a single 48GB GPU without performance degradation.

## 4-bit NormalFloat (NF4)
Standard integer or uniform float quantization is suboptimal for neural network weights, which follow zero-mean normal distributions.
NF4 constructs an information-theoretically optimal quantile distribution where each bin has an equal number of expected data points from a standard normal distribution N(0, 1), minimizing information loss.

## Double Quantization (DQ)
Quantization requires storing block-wise scale constants c_fp32 (e.g., one 32-bit constant per 64 parameters).
Double Quantization quantizes these scale constants themselves into 8-bit FP8 numbers with second-level 32-bit scales, reducing memory overhead from 0.5 bits per parameter to 0.127 bits per parameter (a net saving of 0.373 bits per parameter).

## Paged Optimizers
QLoRA utilizes CUDA Unified Memory to allocate optimizer states (e.g. AdamW first and second moments) into CPU RAM when GPU memory usage spikes during sequence length bursts, eliminating Out-of-Memory (OOM) errors.""",
        metadata={"category": "deep_learning"}
    ),
]

DEFAULT_BENCHMARK_DATASET: List[BenchmarkDatasetItem] = [
    BenchmarkDatasetItem(
        id="lex_01_hpa_formula",
        query="What is the deterministic formula used by Horizontal Pod Autoscaler to compute desired replicas?",
        category="lexical_exact",
        ground_truth_answer="desiredReplicas = ceil[currentReplicas * (currentMetricValue / desiredMetricValue)]",
        ground_truth_context_fragments=['desiredReplicas = ceil[currentReplicas * (currentMetricValue / desiredMetricValue)]'],
        target_document_id="doc_k8s_autoscaling"
    ),
    BenchmarkDatasetItem(
        id="lex_02_mvcc_tuple_headers",
        query="What header fields in PostgreSQL MVCC tuple headers record the inserting and deleting transaction IDs?",
        category="lexical_exact",
        ground_truth_answer="xmin records the transaction ID that inserted the tuple, and xmax records the transaction ID that deleted or updated the tuple.",
        ground_truth_context_fragments=['xmin (transaction ID that inserted the tuple) and xmax (transaction ID that deleted or updated the tuple)'],
        target_document_id="doc_database_acid_isolation"
    ),
    BenchmarkDatasetItem(
        id="lex_03_rrf_cormack_constant",
        query="What is the canonical Cormack smoothing constant k used in Reciprocal Rank Fusion?",
        category="lexical_exact",
        ground_truth_answer="The canonical Cormack smoothing constant is k = 60.",
        ground_truth_context_fragments=['k = 60: The canonical Cormack smoothing constant'],
        target_document_id="doc_rrf_fusion_algorithm"
    ),
    BenchmarkDatasetItem(
        id="lex_04_bm25plus_delta",
        query="What lower-bound smoothing constant delta does BM25Plus introduce to prevent zero scores on long documents?",
        category="lexical_exact",
        ground_truth_answer="BM25Plus introduces a lower-bound smoothing constant delta = 1.0.",
        ground_truth_context_fragments=['adding a lower-bound smoothing constant delta = 1.0', 'BM25Plus_Score = BM25_Score + delta'],
        target_document_id="doc_bm25_information_retrieval"
    ),
    BenchmarkDatasetItem(
        id="lex_05_hnsw_hyperparameters",
        query="What are the three primary hyperparameters in Qdrant HNSW indexing and what do they control?",
        category="lexical_exact",
        ground_truth_answer="m (maximum edges per node, 16-64), ef_construct (construction beam width, 100-512), and ef_search (query-time beam width).",
        ground_truth_context_fragments=['m (Maximum Edges per Node): Number of bidirectional links established per vertex', 'ef_construct (Construction Beam Width)', 'ef_search (Query-Time Beam Width)'],
        target_document_id="doc_qdrant_hnsw_indexing"
    ),
    BenchmarkDatasetItem(
        id="lex_06_raft_heartbeat_timeout",
        query="What is the heartbeat interval HeartbeatTimeout used by the Raft leader to maintain authority?",
        category="lexical_exact",
        ground_truth_answer="The heartbeat interval is HeartbeatTimeout = 50ms broadcast via AppendEntries RPCs.",
        ground_truth_context_fragments=['AppendEntries RPCs at an interval of HeartbeatTimeout = 50ms'],
        target_document_id="doc_raft_consensus"
    ),
    BenchmarkDatasetItem(
        id="lex_07_write_skew_anomaly",
        query="What is the definition of the Write Skew (A5B) concurrency anomaly in transaction isolation?",
        category="lexical_exact",
        ground_truth_answer="Write Skew occurs when two concurrent transactions read overlapping datasets, verify invariant constraints independently, and perform non-overlapping writes that collectively violate the global constraint.",
        ground_truth_context_fragments=['Write Skew (A5B): Occurs when two concurrent transactions read overlapping datasets, verify invariant constraints independently, and perform non-overlapping writes'],
        target_document_id="doc_database_acid_isolation"
    ),
    BenchmarkDatasetItem(
        id="lex_08_nf4_quantization",
        query="What is the 4-bit NormalFloat (NF4) data type in QLoRA designed to quantize?",
        category="lexical_exact",
        ground_truth_answer="NF4 constructs an information-theoretically optimal quantile distribution for zero-mean normal distribution N(0, 1) weights.",
        ground_truth_context_fragments=['NF4 constructs an information-theoretically optimal quantile distribution where each bin has an equal number of expected data points from a standard normal distribution N(0, 1)'],
        target_document_id="doc_llm_qlora_quantization"
    ),
    BenchmarkDatasetItem(
        id="lex_09_double_quantization_savings",
        query="How many bits per parameter does Double Quantization save in QLoRA by quantizing scale constants?",
        category="lexical_exact",
        ground_truth_answer="Double Quantization saves 0.373 bits per parameter by reducing scale constant overhead from 0.5 to 0.127 bits per parameter.",
        ground_truth_context_fragments=['reducing memory overhead from 0.5 bits per parameter to 0.127 bits per parameter (a net saving of 0.373 bits per parameter)'],
        target_document_id="doc_llm_qlora_quantization"
    ),
    BenchmarkDatasetItem(
        id="lex_10_paged_attention_waste",
        query="By how much does PagedAttention reduce KV-cache memory fragmentation?",
        category="lexical_exact",
        ground_truth_answer="PagedAttention reduces memory fragmentation from over 60% down to under 4%.",
        ground_truth_context_fragments=['reducing memory fragmentation from over 60% down to under 4%'],
        target_document_id="doc_llm_speculative_decoding"
    ),
    BenchmarkDatasetItem(
        id="lex_11_cloud_take_or_pay_penalty",
        query="What contractual underutilization penalty applies under the company's take-or-pay cloud infrastructure agreements?",
        category="lexical_exact",
        ground_truth_answer="An 18% contractual underutilization penalty applies to unconsumed compute and bandwidth capacity.",
        ground_truth_context_fragments=['an 18% contractual underutilization penalty applies to unconsumed capacity'],
        target_document_id="doc_sec_10k_leases_commitments"
    ),
    BenchmarkDatasetItem(
        id="lex_12_coredns_fqdn_format",
        query="What is the canonical fully-qualified domain name (FQDN) format used by CoreDNS in Kubernetes?",
        category="lexical_exact",
        ground_truth_answer="service-name.namespace.svc.cluster.local",
        ground_truth_context_fragments=['service-name.namespace.svc.cluster.local'],
        target_document_id="doc_k8s_networking_cni"
    ),
    BenchmarkDatasetItem(
        id="lex_13_cilium_ebpf_sockops",
        query="How does Cilium bypass the Linux kernel TCP/IP network stack for inter-pod communication?",
        category="lexical_exact",
        ground_truth_answer="By attaching extended Berkeley Packet Filter (eBPF) programs to Linux kernel socket operations (sockops).",
        ground_truth_context_fragments=['attaching extended Berkeley Packet Filter (eBPF) programs to Linux kernel socket operations (sockops), bypassing the kernel TCP/IP network stack'],
        target_document_id="doc_k8s_networking_cni"
    ),
    BenchmarkDatasetItem(
        id="lex_14_siread_lock_graph",
        query="How does PostgreSQL's Serializable Snapshot Isolation detect write skew without table-level locks?",
        category="lexical_exact",
        ground_truth_answer="By dynamically tracking read-write conflicts (rw-antidependencies) in an in-memory lock graph to detect dependency cycles.",
        ground_truth_context_fragments=['Serializable Snapshot Isolation (SSI), which dynamically tracks read-write conflicts (rw-antidependencies) in an in-memory lock graph to detect dependency cycles'],
        target_document_id="doc_database_acid_isolation"
    ),
    BenchmarkDatasetItem(
        id="lex_15_online_softmax_trick",
        query="What algorithmic technique allows FlashAttention to incrementally compute softmax scaling factors in SRAM?",
        category="lexical_exact",
        ground_truth_answer="The Milakov & Gimelshein online softmax trick.",
        ground_truth_context_fragments=['Utilizes the Milakov & Gimelshein online softmax trick to incrementally compute softmax scaling factors'],
        target_document_id="doc_llm_flashattention"
    ),
    BenchmarkDatasetItem(
        id="lex_16_memtable_skiplist",
        query="What data structure is typically used to implement the in-memory MemTable in an LSM-tree?",
        category="lexical_exact",
        ground_truth_answer="A concurrent skiplist.",
        ground_truth_context_fragments=['MemTable: In-memory write buffer implemented via a concurrent skiplist'],
        target_document_id="doc_database_lsm_btree"
    ),
    BenchmarkDatasetItem(
        id="lex_17_hpa_stabilization_window",
        query="What is the default duration of the scale-down stabilization window in Kubernetes HPA?",
        category="lexical_exact",
        ground_truth_answer="The default duration is 300 seconds.",
        ground_truth_context_fragments=['default duration of 300 seconds'],
        target_document_id="doc_k8s_autoscaling"
    ),
    BenchmarkDatasetItem(
        id="lex_18_raft_quorum_formula",
        query="What is the mathematical quorum size required to commit log entries in a Raft cluster of N nodes?",
        category="lexical_exact",
        ground_truth_answer="QuorumSize = floor(N / 2) + 1",
        ground_truth_context_fragments=['QuorumSize = floor(N / 2) + 1'],
        target_document_id="doc_raft_consensus"
    ),
    BenchmarkDatasetItem(
        id="lex_19_vpa_four_modes",
        query="What are the four execution modes of Vertical Pod Autoscaler (VPA)?",
        category="lexical_exact",
        ground_truth_answer="Off, Initial, Recreate, and Auto.",
        ground_truth_context_fragments=['four distinct execution modes:', 'Off: Computes recommended container requests without applying changes', 'Initial: Assigns recommended resource requests only at pod creation time', 'Recreate: Evicts and restarts active pods', 'Auto: Automatically updates resource requests during runtime'],
        target_document_id="doc_k8s_autoscaling"
    ),
    BenchmarkDatasetItem(
        id="lex_20_ipvs_vs_iptables_complexity",
        query="What is the algorithmic packet lookup complexity of IPVS mode compared to iptables mode in kube-proxy?",
        category="lexical_exact",
        ground_truth_answer="IPVS mode operates at O(1) packet lookup latency using ipset hash tables, while iptables scales at O(n) linear latency.",
        ground_truth_context_fragments=['iptables Mode: Sequential rule traversal. Packet filtering scales at O(n) latency', 'IPVS Mode: Leverages Linux kernel IP Virtual Server using ipset hash tables, providing O(1) packet lookup latency'],
        target_document_id="doc_k8s_networking_cni"
    ),
    BenchmarkDatasetItem(
        id="lex_21_leveled_compaction_waf",
        query="What is the typical Write Amplification Factor (WAF) in Leveled Compaction for LSM-trees?",
        category="lexical_exact",
        ground_truth_answer="WAF is typically between 10 and 30.",
        ground_truth_context_fragments=['Write Amplification Factor (WAF typically 10 to 30)'],
        target_document_id="doc_database_lsm_btree"
    ),
    BenchmarkDatasetItem(
        id="lex_22_cross_encoder_model_name",
        query="Which transformer model architecture is cited for second-stage candidate reranking?",
        category="lexical_exact",
        ground_truth_answer="ms-marco-MiniLM-L-6-v2",
        ground_truth_context_fragments=['ms-marco-MiniLM-L-6-v2'],
        target_document_id="doc_cross_encoders_reranking"
    ),
    BenchmarkDatasetItem(
        id="lex_23_imputed_interest_leases",
        query="How much imputed interest was deducted from undiscounted operating lease payments to arrive at present value?",
        category="lexical_exact",
        ground_truth_answer="Imputed interest of 6140 million USD calculated at a 3.85% discount rate.",
        ground_truth_context_fragments=['deducting imputed interest of 6140 million USD calculated using a weighted-average discount rate of 3.85%'],
        target_document_id="doc_sec_10k_leases_commitments"
    ),
    BenchmarkDatasetItem(
        id="lex_24_services_gross_margin",
        query="What was the Services gross margin in fiscal 2025 compared to Hardware Products?",
        category="lexical_exact",
        ground_truth_answer="Services gross margin was 73.8% compared to 36.5% for Hardware Products.",
        ground_truth_context_fragments=['Gross margin for Services was 73.8% compared to 36.5% for Hardware Products'],
        target_document_id="doc_sec_10k_segment_revenue"
    ),
    BenchmarkDatasetItem(
        id="lex_25_cuda_unified_memory_qlora",
        query="What GPU memory technology does QLoRA utilize for paged optimizers to eliminate out-of-memory errors?",
        category="lexical_exact",
        ground_truth_answer="CUDA Unified Memory to page optimizer states into CPU RAM.",
        ground_truth_context_fragments=['QLoRA utilizes CUDA Unified Memory to allocate optimizer states (e.g. AdamW first and second moments) into CPU RAM'],
        target_document_id="doc_llm_qlora_quantization"
    ),
    BenchmarkDatasetItem(
        id="sem_26_flapping_mitigation",
        query="How does the workload scheduler prevent sharp oscillations when decreasing server instances?",
        category="semantic_paraphrase",
        ground_truth_answer="HPA enforces a configurable scale-down stabilization window with a default duration of 300 seconds to prevent rapid flapping.",
        ground_truth_context_fragments=['prevent rapid flapping or thrashing during transient load drops, HPA enforces a configurable scale-down stabilization window with a default duration of 300 seconds'],
        target_document_id="doc_k8s_autoscaling"
    ),
    BenchmarkDatasetItem(
        id="sem_27_score_calibration_failure",
        query="Why does simple mathematical addition fail when merging dense vector similarities with inverted index scores?",
        category="semantic_paraphrase",
        ground_truth_answer="Linear combination fails because dense cosine scores and sparse BM25 scores exhibit incompatible probability distributions and scale variances.",
        ground_truth_context_fragments=['Linear combination of retrieval scores fails because dense cosine similarity scores [0.0, 1.0] and sparse BM25 scores [0.0, inf) exhibit incompatible probability distributions'],
        target_document_id="doc_rrf_fusion_algorithm"
    ),
    BenchmarkDatasetItem(
        id="sem_28_logarithmic_vector_search",
        query="What structural characteristic allows approximate nearest neighbor graphs to navigate vector spaces in sub-linear time?",
        category="semantic_paraphrase",
        ground_truth_answer="Top layers feature sparse graphs with long-distance links enabling logarithmic O(log N) geometric exploration.",
        ground_truth_context_fragments=['Top Layers: Sparse graphs containing long-distance links that enable logarithmic O(log N) geometric exploration across the vector space'],
        target_document_id="doc_qdrant_hnsw_indexing"
    ),
    BenchmarkDatasetItem(
        id="sem_29_cross_transaction_invariant_violation",
        query="What concurrency problem arises when concurrent processes read related records and write distinct modifications that together violate constraints?",
        category="semantic_paraphrase",
        ground_truth_answer="Write Skew (A5B), where transactions read overlapping data, verify constraints independently, and perform non-overlapping writes violating the constraint.",
        ground_truth_context_fragments=['Write Skew (A5B): Occurs when two concurrent transactions read overlapping datasets, verify invariant constraints independently, and perform non-overlapping writes that collectively violate the global constraint'],
        target_document_id="doc_database_acid_isolation"
    ),
    BenchmarkDatasetItem(
        id="sem_30_decoding_underutilization",
        query="Why does sequentially producing words one after another underutilize processor computational capacity?",
        category="semantic_paraphrase",
        ground_truth_answer="Autoregressive generation is strictly memory-bandwidth bound with arithmetic intensity under 1 FLOP per byte as all model weights must be moved from HBM for every token.",
        ground_truth_context_fragments=['generating a single token requires transferring all model weights', 'arithmetic intensity is less than 1 FLOP per byte, severely underutilizing GPU Tensor Cores'],
        target_document_id="doc_llm_speculative_decoding"
    ),
    BenchmarkDatasetItem(
        id="sem_31_bypassing_tcp_ip_stack",
        query="How do modern cluster network drivers eliminate kernel network stack latency for container communication on the same host?",
        category="semantic_paraphrase",
        ground_truth_answer="By hooking eBPF programs into kernel socket operations to bypass the standard TCP/IP stack.",
        ground_truth_context_fragments=['attaching extended Berkeley Packet Filter (eBPF) programs to Linux kernel socket operations (sockops), bypassing the kernel TCP/IP network stack'],
        target_document_id="doc_k8s_networking_cni"
    ),
    BenchmarkDatasetItem(
        id="sem_32_joint_cross_attention_superiority",
        query="Why can full transformer self-attention over combined inputs resolve nuances that isolated vector representations miss?",
        category="semantic_paraphrase",
        ground_truth_answer="Every query token can evaluate full bidirectional cross-attention with every passage token, eliminating the semantic compression bottleneck of bi-encoders.",
        ground_truth_context_fragments=['Every query token evaluates full bidirectional cross-attention with every passage token through 6 to 24 transformer layers', 'filtering semantic distractors and false positives'],
        target_document_id="doc_cross_encoders_reranking"
    ),
    BenchmarkDatasetItem(
        id="sem_33_preventing_election_deadlocks",
        query="How does the consensus state machine prevent election deadlocks when multiple candidate peers demand leadership simultaneously?",
        category="semantic_paraphrase",
        ground_truth_answer="By assigning randomized election timeouts uniformly distributed between 150ms and 300ms to desynchronize election attempts.",
        ground_truth_context_fragments=['To avoid split-vote deadlocks, Raft uses randomized election timeouts uniformly distributed between 150ms and 300ms'],
        target_document_id="doc_raft_consensus"
    ),
    BenchmarkDatasetItem(
        id="sem_34_preventing_long_document_underflow",
        query="What enhancement prevents lexical search from assigning near-zero relevance to lengthy articles that mention an uncommon keyword once?",
        category="semantic_paraphrase",
        ground_truth_answer="BM25Plus adds a lower-bound smoothing constant delta = 1.0 guaranteeing that documents with matching tokens receive a non-zero score.",
        ground_truth_context_fragments=['BM25Plus remedies this by adding a lower-bound smoothing constant delta = 1.0', 'guarantees that any document containing a matching query token receives a strictly non-zero score regardless of total document length'],
        target_document_id="doc_bm25_information_retrieval"
    ),
    BenchmarkDatasetItem(
        id="sem_35_preventing_gpu_stall_via_sram",
        query="How does FlashAttention avoid GPU core stalls during attention matrix evaluation?",
        category="semantic_paraphrase",
        ground_truth_answer="By loading blocks of Q, K, and V into fast on-chip SRAM (~19 TB/s) using tiling and recomputing attention on-the-fly during backpropagation.",
        ground_truth_context_fragments=['FlashAttention avoids materializing the N x N attention matrix in HBM by loading blocks of Q, K, and V into fast on-chip SRAM', 'Tiling: Divides the Q, K, and V matrices into blocks that fit entirely within SRAM cache'],
        target_document_id="doc_llm_flashattention"
    ),
    BenchmarkDatasetItem(
        id="sem_36_kv_cache_virtual_paging",
        query="How can memory allocation for generative model key-value tensors be optimized to avoid contiguous block waste?",
        category="semantic_paraphrase",
        ground_truth_answer="PagedAttention allocates KV tensors into non-contiguous physical memory blocks, reducing fragmentation to under 4%.",
        ground_truth_context_fragments=['PagedAttention (vLLM) resolves Key-Value (KV) cache memory waste by allocating KV tensors into non-contiguous physical memory blocks'],
        target_document_id="doc_llm_speculative_decoding"
    ),
    BenchmarkDatasetItem(
        id="sem_37_reducing_read_amplification_lsm",
        query="Why does maintaining non-overlapping key boundaries in disk levels accelerate retrieval in log-structured storage?",
        category="semantic_paraphrase",
        ground_truth_answer="Non-overlapping key ranges in L1+ ensure that each key resides in at most one SSTable per level, minimizing Read Amplification.",
        ground_truth_context_fragments=['SSTables in L1+ contain non-overlapping key ranges, minimizing Read Amplification'],
        target_document_id="doc_database_lsm_btree"
    ),
    BenchmarkDatasetItem(
        id="sem_38_non_overwriting_tuple_storage",
        query="How does the relational database engine preserve transaction isolation without mutating existing data rows in place?",
        category="semantic_paraphrase",
        ground_truth_answer="Via Multi-Version Concurrency Control (MVCC) where row versions are tagged with xmin and xmax tuple headers.",
        ground_truth_context_fragments=['PostgreSQL implements Multi-Version Concurrency Control (MVCC) where rows are never modified in place', 'Each tuple header contains xmin'],
        target_document_id="doc_database_acid_isolation"
    ),
    BenchmarkDatasetItem(
        id="sem_39_quantized_parameter_finetuning",
        query="How can massive neural networks be adapted on modest hardware without loss of model quality?",
        category="semantic_paraphrase",
        ground_truth_answer="QLoRA combines 4-bit NormalFloat quantization, Double Quantization, and Paged Optimizers to fine-tune 65B models on a single 48GB GPU.",
        ground_truth_context_fragments=['QLoRA (Dettmers et al., 2023) reduces the memory footprint of LLM fine-tuning to enable training a 65B parameter model on a single 48GB GPU without performance degradation'],
        target_document_id="doc_llm_qlora_quantization"
    ),
    BenchmarkDatasetItem(
        id="sem_40_underutilization_cloud_penalties",
        query="What financial obligation occurs if actual server utilization falls short of guaranteed cloud provider contracts?",
        category="semantic_paraphrase",
        ground_truth_answer="An 18% contractual underutilization penalty applies to unconsumed capacity under multi-year take-or-pay agreements.",
        ground_truth_context_fragments=['If annual minimum compute and bandwidth thresholds are not achieved, an 18% contractual underutilization penalty applies to unconsumed capacity'],
        target_document_id="doc_sec_10k_leases_commitments"
    ),
    BenchmarkDatasetItem(
        id="sem_41_iptables_linear_scaling_issue",
        query="Why does service routing performance degrade severely in large clusters when using packet filter rule chains?",
        category="semantic_paraphrase",
        ground_truth_answer="Sequential rule evaluation scales at O(n) latency, causing high CPU consumption when services exceed 5000.",
        ground_truth_context_fragments=['Sequential rule traversal. Packet filtering scales at O(n) latency as cluster service counts increase, leading to high CPU consumption when clusters exceed 5000 services'],
        target_document_id="doc_k8s_networking_cni"
    ),
    BenchmarkDatasetItem(
        id="sem_42_liquid_asset_solvency_ratio",
        query="How is the company's immediate ability to discharge debts evaluated excluding unsold inventory?",
        category="semantic_paraphrase",
        ground_truth_answer="Via the quick ratio (cash, marketable securities, and receivables divided by current liabilities), which stood at 1.158 in 2025.",
        ground_truth_context_fragments=['The quick ratio (cash, marketable securities, and accounts receivable divided by total current liabilities) was 1.103 in 2024 and expanded to 1.158 in 2025'],
        target_document_id="doc_sec_10k_balance_sheet"
    ),
    BenchmarkDatasetItem(
        id="sem_43_rejection_sampling_distribution_match",
        query="How does the speculative generation verification step guarantee identical statistical outputs to standard LLM generation?",
        category="semantic_paraphrase",
        ground_truth_answer="By using modified rejection sampling based on the acceptance ratio P_target(x) / P_draft(x), mathematically ensuring exact distribution equivalence.",
        ground_truth_context_fragments=['Modified Rejection Sampling: Tokens are accepted or rejected based on the probability ratio P_target(x) / P_draft(x), mathematically guaranteeing that the sampled distribution matches the target model exactly'],
        target_document_id="doc_llm_speculative_decoding"
    ),
    BenchmarkDatasetItem(
        id="sem_44_bm25_length_normalization_tradeoff",
        query="Which parameter in probabilistic document scoring tunes the penalty for lengthy documents?",
        category="semantic_paraphrase",
        ground_truth_answer="Parameter b, where b = 1.0 fully normalizes against average length and b = 0.0 disables length penalty.",
        ground_truth_context_fragments=['b (Document Length Normalization): Controls the penalty applied to long documents. When b equals 1.0, document length is fully normalized against avgdl'],
        target_document_id="doc_bm25_information_retrieval"
    ),
    BenchmarkDatasetItem(
        id="sem_45_wal_crash_recovery",
        query="What disk log enables database engines to restore modifications held in volatile memory after power loss?",
        category="semantic_paraphrase",
        ground_truth_answer="The Write-Ahead Log (WAL), a sequential append-only disk log providing crash durability.",
        ground_truth_context_fragments=['Write-Ahead Log (WAL): Sequential append-only disk log providing durability across crashes'],
        target_document_id="doc_database_lsm_btree"
    ),
    BenchmarkDatasetItem(
        id="sem_46_services_margin_advantage",
        query="Why does software and subscription income yield substantially better profit margins than manufacturing hardware goods?",
        category="semantic_paraphrase",
        ground_truth_answer="Services achieved a 73.8% gross margin compared to 36.5% for hardware due to the high operating leverage of digital subscriptions and fees.",
        ground_truth_context_fragments=['Gross margin for Services was 73.8% compared to 36.5% for Hardware Products, reflecting the high operating leverage of digital ecosystem subscriptions'],
        target_document_id="doc_sec_10k_segment_revenue"
    ),
    BenchmarkDatasetItem(
        id="sem_47_hnsw_edge_connectivity_tradeoff",
        query="What trade-off is involved when configuring maximum link density per node in hierarchical proximity graphs?",
        category="semantic_paraphrase",
        ground_truth_answer="Higher m improves search recall on high-dimensional vectors but increases index RAM consumption.",
        ground_truth_context_fragments=['m (Maximum Edges per Node): Number of bidirectional links established per vertex, typically tuned between 16 and 64. Higher m improves recall on high-dimensional vectors at the cost of higher RAM consumption'],
        target_document_id="doc_qdrant_hnsw_indexing"
    ),
    BenchmarkDatasetItem(
        id="sem_48_double_quantization_concept",
        query="How does compressing the compression constants themselves yield footprint reductions in low-precision neural networks?",
        category="semantic_paraphrase",
        ground_truth_answer="Double Quantization quantizes 32-bit scale constants into 8-bit FP8 numbers with secondary scales, saving 0.373 bits per parameter.",
        ground_truth_context_fragments=['Double Quantization quantizes these scale constants themselves into 8-bit FP8 numbers with second-level 32-bit scales'],
        target_document_id="doc_llm_qlora_quantization"
    ),
    BenchmarkDatasetItem(
        id="sem_49_vpa_resource_modification",
        query="How does the workload scaler change container CPU and RAM sizing without altering the replica count?",
        category="semantic_paraphrase",
        ground_truth_answer="Vertical Pod Autoscaler (VPA) modifies container CPU and memory resource requests across Off, Initial, Recreate, and Auto modes.",
        ground_truth_context_fragments=['Vertical Pod Autoscaler (VPA) modifies container CPU and memory resource requests'],
        target_document_id="doc_k8s_autoscaling"
    ),
    BenchmarkDatasetItem(
        id="sem_50_raft_log_safety_guarantee",
        query="How does the consensus protocol ensure that consistent log histories are maintained across all distributed cluster nodes?",
        category="semantic_paraphrase",
        ground_truth_answer="Via the Log Matching Property: if two logs contain an entry with the same index and term, they are identical up to that index.",
        ground_truth_context_fragments=['Log Matching Property: If two logs contain an entry with the same index and term, they are identical up to that index'],
        target_document_id="doc_raft_consensus"
    ),
    BenchmarkDatasetItem(
        id="tab_51_total_current_liabilities_2025",
        query="What were the total current liabilities of the Corporation in fiscal year 2025?",
        category="tabular_quantitative",
        ground_truth_answer="91020 million USD.",
        ground_truth_context_fragments=['| Total current liabilities | 82570 | 91020 | Current Liability |'],
        target_document_id="doc_sec_10k_balance_sheet"
    ),
    BenchmarkDatasetItem(
        id="tab_52_cash_equivalents_2024",
        query="What was the cash and cash equivalents balance at the end of fiscal year 2024?",
        category="tabular_quantitative",
        ground_truth_answer="29965 million USD.",
        ground_truth_context_fragments=['| Cash and cash equivalents | 29965 | 34812 | Current Asset |'],
        target_document_id="doc_sec_10k_balance_sheet"
    ),
    BenchmarkDatasetItem(
        id="tab_53_quick_ratio_figures",
        query="What were the quick ratios for fiscal 2024 and 2025 reported in the liquidity analysis?",
        category="tabular_quantitative",
        ground_truth_answer="1.103 in 2024 and 1.158 in 2025.",
        ground_truth_context_fragments=['quick ratio (cash, marketable securities, and accounts receivable divided by total current liabilities) was 1.103 in 2024 and expanded to 1.158 in 2025'],
        target_document_id="doc_sec_10k_balance_sheet"
    ),
    BenchmarkDatasetItem(
        id="tab_54_total_current_assets_2025",
        query="What was total current assets in fiscal 2025 compared to 2024?",
        category="tabular_quantitative",
        ground_truth_answer="112492 million USD in 2025 compared to 97565 million USD in 2024.",
        ground_truth_context_fragments=['| Total current assets | 97565 | 112492 | Current Asset |'],
        target_document_id="doc_sec_10k_balance_sheet"
    ),
    BenchmarkDatasetItem(
        id="tab_55_stockholders_equity_deficit_2025",
        query="What was the stockholders' equity deficit in fiscal 2025?",
        category="tabular_quantitative",
        ground_truth_answer="-33958 million USD.",
        ground_truth_context_fragments=['| Total stockholders equity | -36570 | -33958 | Equity Deficit |'],
        target_document_id="doc_sec_10k_balance_sheet"
    ),
    BenchmarkDatasetItem(
        id="tab_56_americas_sales_2025",
        query="What was the net sales figure for the Americas geographic segment in fiscal 2025?",
        category="tabular_quantitative",
        ground_truth_answer="162500 million USD.",
        ground_truth_context_fragments=['| Americas | 150200 | 162500 | 37.4% |'],
        target_document_id="doc_sec_10k_segment_revenue"
    ),
    BenchmarkDatasetItem(
        id="tab_57_china_operating_margin",
        query="What was the operating margin for Greater China reported in Note 19?",
        category="tabular_quantitative",
        ground_truth_answer="41.2%",
        ground_truth_context_fragments=['| Greater China | 68200 | 72600 | 41.2% |'],
        target_document_id="doc_sec_10k_segment_revenue"
    ),
    BenchmarkDatasetItem(
        id="tab_58_consolidated_net_sales_2025",
        query="What were total consolidated net sales in fiscal 2025?",
        category="tabular_quantitative",
        ground_truth_answer="361100 million USD.",
        ground_truth_context_fragments=['| Total Consolidated Net Sales | 335700 | 361100 | 36.8% |'],
        target_document_id="doc_sec_10k_segment_revenue"
    ),
    BenchmarkDatasetItem(
        id="tab_59_europe_sales_2024",
        query="What was the net sales for Europe in fiscal year 2024?",
        category="tabular_quantitative",
        ground_truth_answer="88400 million USD.",
        ground_truth_context_fragments=['| Europe | 88400 | 94800 | 34.1% |'],
        target_document_id="doc_sec_10k_segment_revenue"
    ),
    BenchmarkDatasetItem(
        id="tab_60_lease_payments_2026",
        query="How much undiscounted lease payment is due in fiscal maturity year 2026?",
        category="tabular_quantitative",
        ground_truth_answer="11240 million USD.",
        ground_truth_context_fragments=['| 2026 | 11240 | 3450 |'],
        target_document_id="doc_sec_10k_leases_commitments"
    ),
    BenchmarkDatasetItem(
        id="tab_61_cloud_commitments_2027",
        query="What are the scheduled cloud commitments for fiscal year 2027?",
        category="tabular_quantitative",
        ground_truth_answer="3800 million USD.",
        ground_truth_context_fragments=['| 2027 | 9810 | 3800 |'],
        target_document_id="doc_sec_10k_leases_commitments"
    ),
    BenchmarkDatasetItem(
        id="tab_62_total_cloud_commitments",
        query="What is the total cumulative cloud commitments amount through fiscal 2030?",
        category="tabular_quantitative",
        ground_truth_answer="14500 million USD.",
        ground_truth_context_fragments=['| Total Undiscounted Payments | 48220 | 14500 |', 'minimum of 14500 million USD through fiscal 2030'],
        target_document_id="doc_sec_10k_leases_commitments"
    ),
    BenchmarkDatasetItem(
        id="tab_63_ppe_net_2025",
        query="What was Property plant and equipment net at the end of fiscal 2025?",
        category="tabular_quantitative",
        ground_truth_answer="51200 million USD.",
        ground_truth_context_fragments=['| Property plant and equipment net | 43715 | 51200 | Non-Current Asset |'],
        target_document_id="doc_sec_10k_balance_sheet"
    ),
    BenchmarkDatasetItem(
        id="tab_64_working_capital_2025",
        query="What was the working capital balance at the close of fiscal 2025?",
        category="tabular_quantitative",
        ground_truth_answer="21472 million USD.",
        ground_truth_context_fragments=['Working capital totaled 21472 million USD at the close of fiscal 2025'],
        target_document_id="doc_sec_10k_balance_sheet"
    ),
    BenchmarkDatasetItem(
        id="tab_65_pv_operating_leases",
        query="What was the present value of operating lease liabilities recorded on the balance sheet?",
        category="tabular_quantitative",
        ground_truth_answer="42080 million USD.",
        ground_truth_context_fragments=['present value of operating lease liabilities recorded on the balance sheet is 42080 million USD'],
        target_document_id="doc_sec_10k_leases_commitments"
    ),
    BenchmarkDatasetItem(
        id="tab_66_services_gross_margin_value",
        query="What was the gross margin for Services reported in fiscal 2025?",
        category="tabular_quantitative",
        ground_truth_answer="73.8%",
        ground_truth_context_fragments=['Gross margin for Services was 73.8% compared to 36.5% for Hardware Products'],
        target_document_id="doc_sec_10k_segment_revenue"
    ),
    BenchmarkDatasetItem(
        id="tab_67_accounts_payable_comparison",
        query="What were accounts payable in fiscal 2024 and 2025?",
        category="tabular_quantitative",
        ground_truth_answer="54120 million USD in 2024 and 59800 million USD in 2025.",
        ground_truth_context_fragments=['| Accounts payable | 54120 | 59800 | Current Liability |'],
        target_document_id="doc_sec_10k_balance_sheet"
    ),
    BenchmarkDatasetItem(
        id="tab_68_lease_payments_2029",
        query="What undiscounted lease payment is due in fiscal maturity year 2029?",
        category="tabular_quantitative",
        ground_truth_answer="5120 million USD.",
        ground_truth_context_fragments=['| 2029 | 5120 | 2500 |'],
        target_document_id="doc_sec_10k_leases_commitments"
    ),
    BenchmarkDatasetItem(
        id="tab_69_current_ratio_2024",
        query="What was the corporate current ratio in fiscal year 2024?",
        category="tabular_quantitative",
        ground_truth_answer="1.182",
        ground_truth_context_fragments=['current ratio increased from 1.182 in fiscal 2024 to 1.236 in fiscal 2025'],
        target_document_id="doc_sec_10k_balance_sheet"
    ),
    BenchmarkDatasetItem(
        id="tab_70_hardware_revenue_2025",
        query="How much net sales revenue was generated from Hardware Products in fiscal 2025?",
        category="tabular_quantitative",
        ground_truth_answer="265800 million USD.",
        ground_truth_context_fragments=['Hardware Products totaling 265800 million USD (73.6% of net sales)'],
        target_document_id="doc_sec_10k_segment_revenue"
    ),
    BenchmarkDatasetItem(
        id="tab_71_asia_pacific_margin",
        query="What was the operating margin for Rest of Asia Pacific in fiscal 2025?",
        category="tabular_quantitative",
        ground_truth_answer="32.8%",
        ground_truth_context_fragments=['| Rest of Asia Pacific | 28900 | 31200 | 32.8% |'],
        target_document_id="doc_sec_10k_segment_revenue"
    ),
    BenchmarkDatasetItem(
        id="tab_72_accrued_expenses_2025",
        query="What was the accrued expenses figure at fiscal year-end 2025?",
        category="tabular_quantitative",
        ground_truth_answer="31220 million USD.",
        ground_truth_context_fragments=['| Accrued expenses | 28450 | 31220 | Current Liability |'],
        target_document_id="doc_sec_10k_balance_sheet"
    ),
    BenchmarkDatasetItem(
        id="tab_73_leases_thereafter",
        query="How much undiscounted lease obligation is due thereafter past fiscal 2029?",
        category="tabular_quantitative",
        ground_truth_answer="14600 million USD.",
        ground_truth_context_fragments=['| Thereafter | 14600 | 1550 |'],
        target_document_id="doc_sec_10k_leases_commitments"
    ),
    BenchmarkDatasetItem(
        id="tab_74_total_undiscounted_leases",
        query="What was the total undiscounted operating lease payments before imputed interest?",
        category="tabular_quantitative",
        ground_truth_answer="48220 million USD.",
        ground_truth_context_fragments=['Total undiscounted operating lease payments equal 48220 million USD'],
        target_document_id="doc_sec_10k_leases_commitments"
    ),
    BenchmarkDatasetItem(
        id="tab_75_services_revenue_percentage",
        query="What percentage of consolidated net sales was represented by Services in fiscal 2025?",
        category="tabular_quantitative",
        ground_truth_answer="26.4%",
        ground_truth_context_fragments=['Services revenue totaling 95300 million USD (26.4% of net sales)'],
        target_document_id="doc_sec_10k_segment_revenue"
    ),
    BenchmarkDatasetItem(
        id="adv_76_consensus_heartbeat_distractor",
        query="What is the heartbeat broadcast interval used by a Raft leader to maintain leadership across followers?",
        category="adversarial_distractor",
        ground_truth_answer="HeartbeatTimeout = 50ms using AppendEntries RPCs.",
        ground_truth_context_fragments=['AppendEntries RPCs at an interval of HeartbeatTimeout = 50ms'],
        target_document_id="doc_raft_consensus"
    ),
    BenchmarkDatasetItem(
        id="adv_77_replica_scaling_formula_distractor",
        query="What exact mathematical formula determines the target number of server pods during autoscaling?",
        category="adversarial_distractor",
        ground_truth_answer="desiredReplicas = ceil[currentReplicas * (currentMetricValue / desiredMetricValue)]",
        ground_truth_context_fragments=['desiredReplicas = ceil[currentReplicas * (currentMetricValue / desiredMetricValue)]'],
        target_document_id="doc_k8s_autoscaling"
    ),
    BenchmarkDatasetItem(
        id="adv_78_postgresql_phantom_prevention",
        query="How does PostgreSQL prevent phantom reads under the Repeatable Read isolation level?",
        category="adversarial_distractor",
        ground_truth_answer="By taking a single transaction snapshot at transaction start under MVCC.",
        ground_truth_context_fragments=['Repeatable Read isolation prevents phantom reads using a single transaction snapshot taken at transaction start'],
        target_document_id="doc_database_acid_isolation"
    ),
    BenchmarkDatasetItem(
        id="adv_79_rrf_smoothing_parameter_k",
        query="What is the smoothing constant k in Reciprocal Rank Fusion and why is it set to 60?",
        category="adversarial_distractor",
        ground_truth_answer="k = 60 prevents top-1 rankings from excessively dominating positions 2 through 5, while preventing low-ranked tail results from distorting scores.",
        ground_truth_context_fragments=['k = 60: The canonical Cormack smoothing constant. Setting k = 60 ensures that top-1 rankings do not excessively dominate positions 2 through 5'],
        target_document_id="doc_rrf_fusion_algorithm"
    ),
    BenchmarkDatasetItem(
        id="adv_80_memtable_functionality",
        query="What role does the MemTable play in a log-structured merge storage engine?",
        category="adversarial_distractor",
        ground_truth_answer="It serves as the in-memory write buffer implemented via a concurrent skiplist before data is flushed to SSTables.",
        ground_truth_context_fragments=['MemTable: In-memory write buffer implemented via a concurrent skiplist'],
        target_document_id="doc_database_lsm_btree"
    ),
    BenchmarkDatasetItem(
        id="adv_81_hnsw_edge_parameter_m",
        query="In Qdrant HNSW indexing, what parameter limits the number of bidirectional connections per node?",
        category="adversarial_distractor",
        ground_truth_answer="Parameter m, typically tuned between 16 and 64.",
        ground_truth_context_fragments=['m (Maximum Edges per Node): Number of bidirectional links established per vertex, typically tuned between 16 and 64'],
        target_document_id="doc_qdrant_hnsw_indexing"
    ),
    BenchmarkDatasetItem(
        id="adv_82_scale_down_stabilization",
        query="What is the default stabilization window applied when reducing the number of pods in Kubernetes?",
        category="adversarial_distractor",
        ground_truth_answer="A default duration of 300 seconds to prevent rapid flapping.",
        ground_truth_context_fragments=['default duration of 300 seconds'],
        target_document_id="doc_k8s_autoscaling"
    ),
    BenchmarkDatasetItem(
        id="adv_83_raft_rpc_for_replication",
        query="Which remote procedure call does a Raft leader use to replicate log entries and maintain authority?",
        category="adversarial_distractor",
        ground_truth_answer="AppendEntries RPC.",
        ground_truth_context_fragments=['AppendEntries RPC: Invoked by the Leader for log replication and heartbeat transmission'],
        target_document_id="doc_raft_consensus"
    ),
    BenchmarkDatasetItem(
        id="adv_84_ipvs_ipset_advantage",
        query="Why does IPVS mode outperform iptables mode in large Kubernetes clusters?",
        category="adversarial_distractor",
        ground_truth_answer="IPVS uses ipset hash tables providing O(1) packet lookup latency rather than O(n) sequential rule traversal.",
        ground_truth_context_fragments=['IPVS Mode: Leverages Linux kernel IP Virtual Server using ipset hash tables, providing O(1) packet lookup latency regardless of service count'],
        target_document_id="doc_k8s_networking_cni"
    ),
    BenchmarkDatasetItem(
        id="adv_85_wal_purpose_in_storage",
        query="What component in an LSM-tree guarantees durability before changes are written to immutable SSTables?",
        category="adversarial_distractor",
        ground_truth_answer="The Write-Ahead Log (WAL), a sequential append-only disk log.",
        ground_truth_context_fragments=['Write-Ahead Log (WAL): Sequential append-only disk log providing durability across crashes'],
        target_document_id="doc_database_lsm_btree"
    ),
    BenchmarkDatasetItem(
        id="adv_86_bm25_parameter_b",
        query="What document characteristic does parameter b normalize in the Okapi BM25 formula?",
        category="adversarial_distractor",
        ground_truth_answer="Document length normalization against average document length (avgdl).",
        ground_truth_context_fragments=['b (Document Length Normalization): Controls the penalty applied to long documents'],
        target_document_id="doc_bm25_information_retrieval"
    ),
    BenchmarkDatasetItem(
        id="adv_87_flashattention_sram_bandwidth",
        query="What fast GPU memory does FlashAttention tile blocks into to bypass High-Bandwidth Memory bottlenecks?",
        category="adversarial_distractor",
        ground_truth_answer="On-chip SRAM with approximately 19 TB/s bandwidth.",
        ground_truth_context_fragments=['loading blocks of Q, K, and V into fast on-chip SRAM (~19 TB/s bandwidth)'],
        target_document_id="doc_llm_flashattention"
    ),
    BenchmarkDatasetItem(
        id="adv_88_quick_vs_current_ratio",
        query="How is the quick ratio distinct from the current ratio in corporate balance sheet analysis?",
        category="adversarial_distractor",
        ground_truth_answer="The quick ratio includes only cash, marketable securities, and accounts receivable, excluding inventories.",
        ground_truth_context_fragments=['The quick ratio (cash, marketable securities, and accounts receivable divided by total current liabilities)'],
        target_document_id="doc_sec_10k_balance_sheet"
    ),
    BenchmarkDatasetItem(
        id="adv_89_speculative_draft_model_role",
        query="What role does the draft model perform in speculative decoding systems?",
        category="adversarial_distractor",
        ground_truth_answer="It autoregressively generates K speculative tokens at high speed for parallel verification by the target model.",
        ground_truth_context_fragments=['Draft Generation: The draft model autoregressively generates K speculative tokens (e.g., K = 4 to 8) at high speed'],
        target_document_id="doc_llm_speculative_decoding"
    ),
    BenchmarkDatasetItem(
        id="adv_90_nf4_distribution_type",
        query="What statistical distribution does the NF4 data type optimize its quantile bins for?",
        category="adversarial_distractor",
        ground_truth_answer="A standard normal distribution N(0, 1).",
        ground_truth_context_fragments=['NF4 constructs an information-theoretically optimal quantile distribution where each bin has an equal number of expected data points from a standard normal distribution N(0, 1)'],
        target_document_id="doc_llm_qlora_quantization"
    ),
    BenchmarkDatasetItem(
        id="syn_91_cash_vs_commitments_coverage",
        query="Compare the Corporation's cash balance at fiscal year-end 2025 against scheduled undiscounted lease and cloud obligations for 2026.",
        category="synthesis_multihop",
        ground_truth_answer="Cash was 34812 million USD, which comfortably exceeds combined 2026 lease (11240 million USD) and cloud commitments (3450 million USD) totaling 14690 million USD.",
        ground_truth_context_fragments=['| Cash and cash equivalents | 29965 | 34812 | Current Asset |'],
        target_document_id="doc_sec_10k_balance_sheet"
    ),
    BenchmarkDatasetItem(
        id="syn_92_dense_sparse_complementarity",
        query="How do dense semantic embeddings and sparse lexical inverted indices complement each other when fused via RRF?",
        category="synthesis_multihop",
        ground_truth_answer="Dense embeddings capture conceptual meaning and synonyms, while sparse BM25 matches exact keywords and codes; RRF merges them without calibration issues.",
        ground_truth_context_fragments=['dense cosine similarity scores [0.0, 1.0] and sparse BM25 scores [0.0, inf)', 'operating solely on monotonic ordinal ranks'],
        target_document_id="doc_rrf_fusion_algorithm"
    ),
    BenchmarkDatasetItem(
        id="syn_93_two_stage_ir_efficiency",
        query="Why does a two-stage retrieval pipeline combining bi-encoder candidate selection with cross-encoder reranking balance speed and accuracy?",
        category="synthesis_multihop",
        ground_truth_answer="Stage 1 retrieves 50-100 candidates in 15-25ms via dense/sparse indices, and Stage 2 runs full cross-attention over those candidates in 30-50ms to eliminate distractors.",
        ground_truth_context_fragments=['First Stage: Rapid retrieval of 50 to 100 candidate chunks via dense vector search and sparse BM25', 'Second Stage: Cross-Encoder (e.g., ms-marco-MiniLM-L-6-v2) computes joint cross-attention logits'],
        target_document_id="doc_cross_encoders_reranking"
    ),
    BenchmarkDatasetItem(
        id="syn_94_layout_aware_table_benefit",
        query="How does layout-aware table extraction preserve financial statement context compared to blind sentence splitting?",
        category="synthesis_multihop",
        ground_truth_answer="It keeps tabular schemas and column headers intact and linearizes row pairs so embedding models can match financial values without losing header context.",
        ground_truth_context_fragments=['Consolidated Balance Sheets', '| Balance Sheet Item | FY 2024 (USD M) | FY 2025 (USD M) | Classification |'],
        target_document_id="doc_sec_10k_balance_sheet"
    ),
    BenchmarkDatasetItem(
        id="syn_95_raft_election_and_log_safety",
        query="How do Raft's randomized election timers and majority quorum requirements together prevent split brain and log loss?",
        category="synthesis_multihop",
        ground_truth_answer="Randomized timers (150-300ms) prevent split votes, and majority quorums guarantee that any elected leader contains all committed entries.",
        ground_truth_context_fragments=['randomized election timeouts uniformly distributed between 150ms and 300ms', 'committed only after successful replication across a strict majority quorum'],
        target_document_id="doc_raft_consensus"
    ),
    BenchmarkDatasetItem(
        id="syn_96_qlora_three_innovations",
        query="What three distinct architectural techniques enable QLoRA to fine-tune a 65B parameter model on a 48GB GPU?",
        category="synthesis_multihop",
        ground_truth_answer="4-bit NormalFloat (NF4) quantization, Double Quantization (DQ) of scales, and Paged Optimizers using CUDA Unified Memory.",
        ground_truth_context_fragments=['4-bit NormalFloat (NF4)', 'Double Quantization (DQ)', 'Paged Optimizers'],
        target_document_id="doc_llm_qlora_quantization"
    ),
    BenchmarkDatasetItem(
        id="syn_97_hnsw_tradeoff_dynamics",
        query="How do m, ef_construct, and ef_search work together to balance construction time, memory usage, and search recall in Qdrant?",
        category="synthesis_multihop",
        ground_truth_answer="m sets link density per node, ef_construct controls graph build quality, and ef_search tunes query-time candidate exploration without reindexing.",
        ground_truth_context_fragments=['m (Maximum Edges per Node): Number of bidirectional links established per vertex', 'ef_construct (Construction Beam Width)', 'ef_search (Query-Time Beam Width)'],
        target_document_id="doc_qdrant_hnsw_indexing"
    ),
    BenchmarkDatasetItem(
        id="syn_98_mvcc_and_ssi_write_skew",
        query="How does PostgreSQL combine MVCC tuple headers with Serializable Snapshot Isolation to eliminate write skew without pessimistic locks?",
        category="synthesis_multihop",
        ground_truth_answer="MVCC tracks versions via xmin/xmax, and SSI dynamically monitors read-write dependency cycles in an in-memory SIREAD lock graph.",
        ground_truth_context_fragments=['Each tuple header contains xmin (transaction ID that inserted the tuple) and xmax', 'Serializable Snapshot Isolation (SSI), which dynamically tracks read-write conflicts (rw-antidependencies) in an in-memory lock graph to detect dependency cycles'],
        target_document_id="doc_database_acid_isolation"
    ),
    BenchmarkDatasetItem(
        id="syn_99_flashattention_tiling_and_recomputation",
        query="Why does FlashAttention's strategy of recomputing attention in SRAM during the backward pass outperform loading stored attention from HBM?",
        category="synthesis_multihop",
        ground_truth_answer="Recomputing in SRAM (~19 TB/s) is faster than transferring N x N matrices to and from slower HBM (~1.5 TB/s), resolving the memory-bandwidth bottleneck.",
        ground_truth_context_fragments=['loading blocks of Q, K, and V into fast on-chip SRAM (~19 TB/s bandwidth)', 'does not read stored attention matrices from HBM; instead, it recomputes attention on-the-fly in fast SRAM'],
        target_document_id="doc_llm_flashattention"
    ),
    BenchmarkDatasetItem(
        id="syn_100_hpa_vs_vpa_resource_allocation",
        query="How do Kubernetes HPA and VPA differ in their approach to handling compute resource scaling?",
        category="synthesis_multihop",
        ground_truth_answer="HPA scales horizontally by modifying replica counts using desiredReplicas, while VPA scales vertically by resizing container CPU and memory requests.",
        ground_truth_context_fragments=['Vertical Pod Autoscaler (VPA) modifies container CPU and memory resource requests'],
        target_document_id="doc_k8s_autoscaling"
    ),
    BenchmarkDatasetItem(
        id="syn_101_bm25plus_long_tail_resilience",
        query="Why does BM25Plus's delta term solve the score vanishing problem on lengthy technical documentation?",
        category="synthesis_multihop",
        ground_truth_answer="Because standard BM25 length normalization penalizes long documents until scores approach zero, whereas adding delta = 1.0 guarantees a strictly positive score.",
        ground_truth_context_fragments=['Standard BM25 suffers when document length |D| is exceptionally large', 'adding a lower-bound smoothing constant delta = 1.0', 'guarantees that any document containing a matching query token receives a strictly non-zero score'],
        target_document_id="doc_bm25_information_retrieval"
    ),
    BenchmarkDatasetItem(
        id="syn_102_cloud_commitment_penalty_risk",
        query="What are the financial implications if corporate compute demand fails to meet the minimum threshold under multi-year cloud commitments?",
        category="synthesis_multihop",
        ground_truth_answer="The company must pay an 18% contractual underutilization penalty on unconsumed capacity out of its 14500 million USD commitment.",
        ground_truth_context_fragments=['minimum of 14500 million USD through fiscal 2030', '18% contractual underutilization penalty applies to unconsumed capacity'],
        target_document_id="doc_sec_10k_leases_commitments"
    ),
    BenchmarkDatasetItem(
        id="syn_103_speculative_decoding_acceleration",
        query="How does pairing a small draft model with a large target model overcome the arithmetic intensity bottleneck in autoregressive decoding?",
        category="synthesis_multihop",
        ground_truth_answer="The draft model rapidly proposes K speculative tokens, and the target model verifies all K in a single forward pass, amortizing HBM weight transfer.",
        ground_truth_context_fragments=['arithmetic intensity is less than 1 FLOP per byte', 'draft model autoregressively generates K speculative tokens', 'target model evaluates all K tokens simultaneously in a single parallel forward pass'],
        target_document_id="doc_llm_speculative_decoding"
    ),
    BenchmarkDatasetItem(
        id="syn_104_lsm_compaction_tradeoffs",
        query="Contrast Leveled Compaction against Size-Tiered Compaction in LSM-trees regarding write amplification and temporary disk space overhead.",
        category="synthesis_multihop",
        ground_truth_answer="Leveled Compaction has non-overlapping key ranges minimizing read amplification but has higher WAF (10-30); Size-Tiered has lower WAF but higher temporary disk overhead.",
        ground_truth_context_fragments=['Leveled Compaction: Each level has a fixed capacity', 'WAF typically 10 to 30', 'Size-Tiered Compaction: Flushes files of similar sizes together, reducing write amplification for write-intensive workloads at the expense of higher temporary disk space overhead'],
        target_document_id="doc_database_lsm_btree"
    ),
    BenchmarkDatasetItem(
        id="syn_105_geographic_sales_vs_margins",
        query="How did segment sales in Greater China compare to the Americas in fiscal 2025, and how did their operating margins compare?",
        category="synthesis_multihop",
        ground_truth_answer="Americas generated 162500 million USD sales at 37.4% margin, while Greater China generated 72600 million USD sales at a higher 41.2% operating margin.",
        ground_truth_context_fragments=['| Americas | 150200 | 162500 | 37.4% |', '| Greater China | 68200 | 72600 | 41.2% |'],
        target_document_id="doc_sec_10k_segment_revenue"
    ),
]
