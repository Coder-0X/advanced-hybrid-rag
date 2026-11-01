from fastapi import APIRouter
from fastapi.responses import HTMLResponse

ui_router = APIRouter(include_in_schema=False)

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Advanced Hybrid RAG Engine | Benchmark Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: { 50: '#eef2ff', 500: '#6366f1', 600: '#4f46e5', 700: '#4338ca', 900: '#1e1b4b' }
          }
        }
      }
    }
  </script>
  <style>
    body { background-color: #0b0f19; color: #f3f4f6; font-family: system-ui, -apple-system, sans-serif; }
    .glass { background: rgba(17, 24, 39, 0.75); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .metric-badge { font-family: ui-monospace, SFMono-Regular, monospace; }
  </style>
</head>
<body class="min-h-screen flex flex-col">
  <!-- Navbar -->
  <header class="glass sticky top-0 z-50 px-6 py-4 border-b border-gray-800 flex justify-between items-center">
    <div class="flex items-center gap-3">
      <div class="h-10 w-10 rounded-xl bg-indigo-600 flex items-center justify-center text-white shadow-lg shadow-indigo-500/30">
        <i class="fa-solid fa-layer-group text-lg"></i>
      </div>
      <div>
        <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
          Advanced Hybrid RAG Engine
          <span class="text-xs bg-indigo-900/60 text-indigo-300 border border-indigo-700/50 px-2 py-0.5 rounded-full font-mono">v1.0.0</span>
        </h1>
        <p class="text-xs text-gray-400">Dense Embeddings + Sparse BM25 + Reciprocal Rank Fusion + Cross-Encoder Reranker</p>
      </div>
    </div>
    <div class="flex items-center gap-4">
      <div id="indexStats" class="text-xs text-gray-400 bg-gray-900/80 px-3 py-1.5 rounded-lg border border-gray-800 font-mono">
        <i class="fa-solid fa-database text-indigo-400 mr-1.5"></i> Chunks: <span id="chunkCount" class="text-white font-bold">-</span>
      </div>
      <button onclick="loadSampleCorpus()" class="text-xs bg-indigo-600 hover:bg-indigo-500 text-white font-medium px-3.5 py-1.5 rounded-lg transition shadow-md shadow-indigo-600/20 flex items-center gap-1.5">
        <i class="fa-solid fa-seedling"></i> Load Sample Data
      </button>
      <a href="/docs" target="_blank" class="text-xs text-gray-400 hover:text-white px-2 py-1 rounded transition">
        API Docs <i class="fa-solid fa-arrow-up-right-from-square text-[10px] ml-0.5"></i>
      </a>
    </div>
  </header>

  <!-- Navigation Tabs -->
  <div class="px-6 pt-4 border-b border-gray-800 flex gap-6 bg-gray-950/40">
    <button onclick="switchTab('playground')" id="tabBtn-playground" class="pb-3 border-b-2 border-indigo-500 text-indigo-400 font-semibold text-sm flex items-center gap-2">
      <i class="fa-solid fa-terminal text-xs"></i> Retrieval & Query Playground
    </button>
    <button onclick="switchTab('benchmark')" id="tabBtn-benchmark" class="pb-3 border-b-2 border-transparent text-gray-400 hover:text-gray-200 font-semibold text-sm flex items-center gap-2">
      <i class="fa-solid fa-chart-line text-xs"></i> Quantitative Benchmark Suite
    </button>
    <button onclick="switchTab('documents')" id="tabBtn-documents" class="pb-3 border-b-2 border-transparent text-gray-400 hover:text-gray-200 font-semibold text-sm flex items-center gap-2">
      <i class="fa-solid fa-folder-open text-xs"></i> Document Corpus
    </button>
  </div>

  <!-- Main Content Container -->
  <main class="flex-1 p-6 max-w-7xl mx-auto w-full">

    <!-- TAB 1: PLAYGROUND -->
    <div id="tab-playground" class="space-y-6">
      <!-- Search & Control Panel -->
      <div class="glass p-5 rounded-2xl space-y-4">
        <div class="flex flex-col md:flex-row gap-4 justify-between items-start md:items-center">
          <!-- Retrieval Mode Pills -->
          <div>
            <label class="block text-xs uppercase font-mono tracking-wider text-gray-400 mb-2">Retrieval Mode</label>
            <div class="inline-flex p-1 bg-gray-900 rounded-xl border border-gray-800">
              <button onclick="setMode('naive')" id="mode-naive" class="px-3.5 py-1.5 rounded-lg text-xs font-medium text-gray-400 hover:text-white transition">
                Naive (Dense Only)
              </button>
              <button onclick="setMode('hybrid')" id="mode-hybrid" class="px-3.5 py-1.5 rounded-lg text-xs font-medium text-gray-400 hover:text-white transition">
                Hybrid (Dense + BM25 RRF)
              </button>
              <button onclick="setMode('reranked')" id="mode-reranked" class="px-3.5 py-1.5 rounded-lg text-xs font-medium bg-indigo-600 text-white shadow transition">
                Reranked (Hybrid + Cross-Encoder)
              </button>
            </div>
          </div>

          <!-- Hyperparameter Badges -->
          <div class="flex flex-wrap gap-4 text-xs text-gray-400">
            <div>
              <span class="text-gray-500">RRF k:</span>
              <span class="font-mono text-gray-300 font-bold ml-1">60</span>
            </div>
            <div>
              <span class="text-gray-500">Dense Weight:</span>
              <span class="font-mono text-gray-300 font-bold ml-1">0.5</span>
            </div>
            <div>
              <span class="text-gray-500">Sparse Weight:</span>
              <span class="font-mono text-gray-300 font-bold ml-1">0.5</span>
            </div>
            <div>
              <span class="text-gray-500">Final Top-K:</span>
              <span class="font-mono text-gray-300 font-bold ml-1">5</span>
            </div>
          </div>
        </div>

        <!-- Query Input Box -->
        <div class="relative">
          <input id="queryInput" type="text" 
            placeholder="Ask a technical question (e.g. 'What is the formula used by HPA in Kubernetes?')..."
            class="w-full bg-gray-900/90 text-white placeholder-gray-500 border border-gray-700/70 rounded-xl pl-4 pr-32 py-3.5 text-sm focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
            onkeydown="if(event.key==='Enter') executeQuery()"
          />
          <button onclick="executeQuery()" id="btnSubmitQuery" class="absolute right-2 top-2 bottom-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 rounded-lg transition flex items-center gap-1.5 shadow">
            <span>Query</span> <i class="fa-solid fa-paper-plane text-[10px]"></i>
          </button>
        </div>

        <!-- Query Suggestions -->
        <div class="flex flex-wrap items-center gap-2 pt-1">
          <span class="text-xs text-gray-500 font-mono">Try queries:</span>
          <button onclick="setQuery('What mathematical formula does Horizontal Pod Autoscaler use to calculate desired replicas?')" class="text-xs bg-gray-900 hover:bg-gray-800 text-gray-300 border border-gray-800 px-2.5 py-1 rounded-lg transition">
            HPA Formula
          </button>
          <button onclick="setQuery('What is the smoothing constant k in Reciprocal Rank Fusion and why is it used?')" class="text-xs bg-gray-900 hover:bg-gray-800 text-gray-300 border border-gray-800 px-2.5 py-1 rounded-lg transition">
            RRF Smoothing Constant
          </button>
          <button onclick="setQuery('Why do cross-encoders reduce hallucinations compared to bi-encoders?')" class="text-xs bg-gray-900 hover:bg-gray-800 text-gray-300 border border-gray-800 px-2.5 py-1 rounded-lg transition">
            Cross-Encoder Attention
          </button>
          <button onclick="setQuery('What does parameter b control in Okapi BM25 ranking?')" class="text-xs bg-gray-900 hover:bg-gray-800 text-gray-300 border border-gray-800 px-2.5 py-1 rounded-lg transition">
            BM25 Parameter b
          </button>
        </div>
      </div>

      <!-- Result View -->
      <div id="resultsContainer" class="hidden space-y-6">
        <!-- Answer & Latency Card -->
        <div class="glass p-6 rounded-2xl border-l-4 border-indigo-500 space-y-4">
          <div class="flex flex-wrap justify-between items-center gap-2 border-b border-gray-800/80 pb-3">
            <div class="flex items-center gap-2">
              <span class="text-xs font-mono uppercase bg-indigo-900/50 text-indigo-300 border border-indigo-700/50 px-2 py-0.5 rounded">Grounded Answer</span>
              <span id="groundingBadge" class="text-xs font-mono bg-emerald-950 text-emerald-400 border border-emerald-800/60 px-2 py-0.5 rounded flex items-center gap-1">
                <i class="fa-solid fa-shield-halved text-[10px]"></i> Faithfulness: <span id="groundingScore">1.0</span>
              </span>
            </div>
            <!-- Latency telemetry pills -->
            <div id="latencyPills" class="flex flex-wrap gap-2 text-xs font-mono text-gray-400">
              <!-- Dynamically populated -->
            </div>
          </div>
          <p id="generatedAnswer" class="text-gray-100 text-sm leading-relaxed whitespace-pre-wrap"></p>
        </div>

        <!-- Retrieved Chunks with Telemetry -->
        <div>
          <h3 class="text-sm font-semibold tracking-wide uppercase text-gray-400 mb-3 flex items-center gap-2 font-mono">
            <i class="fa-solid fa-list-ol text-indigo-400"></i> Retrieved Chunks & Ranking Telemetry
          </h3>
          <div id="retrievedChunksList" class="space-y-3">
            <!-- Dynamically populated -->
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: BENCHMARK SUITE -->
    <div id="tab-benchmark" class="hidden space-y-6">
      <div class="glass p-6 rounded-2xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h2 class="text-lg font-bold text-white flex items-center gap-2">
            Automated Quantitative Evaluation Benchmark
            <i class="fa-solid fa-gauge-high text-indigo-400"></i>
          </h2>
          <p class="text-xs text-gray-400 mt-1">
            Compares Naive RAG vs Hybrid RAG vs Reranked RAG across faithfulness, context precision, context recall, hit rate, MRR, and latency.
          </p>
        </div>
        <button onclick="triggerBenchmark()" id="btnRunBenchmark" class="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs px-5 py-2.5 rounded-xl transition shadow-lg shadow-indigo-600/30 flex items-center gap-2">
          <i class="fa-solid fa-play"></i> Run Quantitative Benchmark
        </button>
      </div>

      <!-- Benchmark Results View -->
      <div id="benchmarkResultsView" class="space-y-6">
        <!-- Comparative Summary Matrix -->
        <div class="glass p-6 rounded-2xl overflow-x-auto">
          <div class="flex justify-between items-center mb-4">
            <h3 class="text-sm font-bold text-white uppercase font-mono tracking-wider flex items-center gap-2">
              <i class="fa-solid fa-table-cells text-indigo-400"></i> Quantitative Comparison Matrix
            </h3>
            <span id="benchmarkTimestamp" class="text-xs font-mono text-gray-400"></span>
          </div>
          <table class="w-full text-left border-collapse text-xs">
            <thead>
              <tr class="border-b border-gray-800 text-gray-400 font-mono">
                <th class="py-3 px-4">Evaluation Metric</th>
                <th class="py-3 px-4">Naive RAG (Dense)</th>
                <th class="py-3 px-4">Hybrid RAG (Dense + BM25)</th>
                <th class="py-3 px-4">Reranked RAG (Cross-Encoder)</th>
                <th class="py-3 px-4">Hybrid Δ</th>
                <th class="py-3 px-4">Reranked Δ</th>
              </tr>
            </thead>
            <tbody id="benchmarkMatrixBody" class="divide-y divide-gray-800/60 font-mono text-gray-300">
              <!-- Dynamically populated -->
            </tbody>
          </table>
        </div>

        <!-- Metric Visualization Bars -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div class="glass p-5 rounded-2xl space-y-3">
            <h4 class="text-xs font-bold text-gray-300 uppercase font-mono flex items-center gap-2">
              <i class="fa-solid fa-bullseye text-indigo-400"></i> Context Precision Comparison
            </h4>
            <div id="barPrecision" class="space-y-2 pt-1 text-xs"></div>
          </div>
          <div class="glass p-5 rounded-2xl space-y-3">
            <h4 class="text-xs font-bold text-gray-300 uppercase font-mono flex items-center gap-2">
              <i class="fa-solid fa-shield-check text-emerald-400"></i> Faithfulness (Context Grounding)
            </h4>
            <div id="barFaithfulness" class="space-y-2 pt-1 text-xs"></div>
          </div>
          <div class="glass p-5 rounded-2xl space-y-3">
            <h4 class="text-xs font-bold text-gray-300 uppercase font-mono flex items-center gap-2">
              <i class="fa-solid fa-ranking-star text-amber-400"></i> Mean Reciprocal Rank (MRR)
            </h4>
            <div id="barMRR" class="space-y-2 pt-1 text-xs"></div>
          </div>
          <div class="glass p-5 rounded-2xl space-y-3">
            <h4 class="text-xs font-bold text-gray-300 uppercase font-mono flex items-center gap-2">
              <i class="fa-solid fa-stopwatch text-cyan-400"></i> Median Latency (P50 ms)
            </h4>
            <div id="barLatency" class="space-y-2 pt-1 text-xs"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: DOCUMENTS -->
    <div id="tab-documents" class="hidden space-y-6">
      <div class="glass p-6 rounded-2xl space-y-4">
        <h2 class="text-lg font-bold text-white flex items-center gap-2">
          Knowledge Base Ingestion
          <i class="fa-solid fa-database text-indigo-400"></i>
        </h2>
        <p class="text-xs text-gray-400">
          Documents added here are chunked, embedded into the Qdrant vector store, and tokenized into the BM25 inverted index concurrently.
        </p>

        <div class="space-y-3 pt-2">
          <input id="docTitle" type="text" placeholder="Document Title (e.g. 'Kubernetes Architecture')" class="w-full bg-gray-900 text-white text-xs border border-gray-800 rounded-xl px-4 py-2.5 focus:outline-none focus:border-indigo-500">
          <textarea id="docText" rows="5" placeholder="Document body text to chunk and index..." class="w-full bg-gray-900 text-white text-xs border border-gray-800 rounded-xl px-4 py-2.5 focus:outline-none focus:border-indigo-500"></textarea>
          <div class="flex justify-between items-center">
            <button onclick="ingestCustomDoc()" class="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 py-2 rounded-xl transition flex items-center gap-1.5">
              <i class="fa-solid fa-plus"></i> Ingest Document
            </button>
            <button onclick="clearAllIndices()" class="text-xs text-rose-400 hover:text-rose-300 px-3 py-1.5 rounded transition">
              <i class="fa-solid fa-trash mr-1"></i> Clear All Indices
            </button>
          </div>
        </div>
      </div>
    </div>
  </main>

  <footer class="glass px-6 py-4 border-t border-gray-800 text-center text-xs text-gray-500">
    Advanced Hybrid RAG Engine | Information Retrieval & Ragas Quantitative Benchmarking
  </footer>

  <script>
    let currentMode = 'reranked';

    function switchTab(tabId) {
      ['playground', 'benchmark', 'documents'].forEach(t => {
        document.getElementById(`tab-${t}`).classList.add('hidden');
        const btn = document.getElementById(`tabBtn-${t}`);
        btn.classList.remove('border-indigo-500', 'text-indigo-400');
        btn.classList.add('border-transparent', 'text-gray-400');
      });
      document.getElementById(`tab-${tabId}`).classList.remove('hidden');
      const activeBtn = document.getElementById(`tabBtn-${tabId}`);
      activeBtn.classList.add('border-indigo-500', 'text-indigo-400');
      activeBtn.classList.remove('border-transparent', 'text-gray-400');

      if (tabId === 'benchmark') {
        fetchLatestBenchmark();
      }
    }

    function setMode(mode) {
      currentMode = mode;
      ['naive', 'hybrid', 'reranked'].forEach(m => {
        const btn = document.getElementById(`mode-${m}`);
        if (m === mode) {
          btn.className = "px-3.5 py-1.5 rounded-lg text-xs font-medium bg-indigo-600 text-white shadow transition";
        } else {
          btn.className = "px-3.5 py-1.5 rounded-lg text-xs font-medium text-gray-400 hover:text-white transition";
        }
      });
    }

    function setQuery(q) {
      document.getElementById('queryInput').value = q;
      executeQuery();
    }

    async function updateStats() {
      try {
        const res = await fetch('/api/v1/stats');
        const data = await res.json();
        document.getElementById('chunkCount').innerText = data.total_chunks_vector_store;
      } catch (err) {
        console.error(err);
      }
    }

    async function loadSampleCorpus() {
      try {
        const res = await fetch('/api/v1/corpus/load-sample', { method: 'POST' });
        const data = await res.json();
        alert(data.message);
        updateStats();
      } catch (err) {
        alert("Failed to load sample data: " + err);
      }
    }

    async function executeQuery() {
      const query = document.getElementById('queryInput').value.trim();
      if (!query) return;

      const btn = document.getElementById('btnSubmitQuery');
      btn.disabled = true;
      btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i>`;

      try {
        const res = await fetch('/api/v1/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: query,
            mode: currentMode,
            top_k: 5
          })
        });
        const data = await res.json();
        renderResults(data);
        updateStats();
      } catch (err) {
        alert("Query error: " + err);
      } finally {
        btn.disabled = false;
        btn.innerHTML = `<span>Query</span> <i class="fa-solid fa-paper-plane text-[10px]"></i>`;
      }
    }

    function renderResults(data) {
      const container = document.getElementById('resultsContainer');
      container.classList.remove('hidden');

      document.getElementById('generatedAnswer').innerText = data.answer || "No answer generated.";
      
      const groundingScore = data.grounding_score !== null ? data.grounding_score : 1.0;
      document.getElementById('groundingScore').innerText = groundingScore;

      // Render latency pills
      const lat = data.latencies;
      const pills = document.getElementById('latencyPills');
      pills.innerHTML = `
        <span class="bg-gray-800/80 px-2 py-0.5 rounded text-gray-300">Dense: ${lat.dense_retrieval_ms}ms</span>
        <span class="bg-gray-800/80 px-2 py-0.5 rounded text-gray-300">BM25: ${lat.sparse_retrieval_ms}ms</span>
        <span class="bg-gray-800/80 px-2 py-0.5 rounded text-gray-300">RRF: ${lat.fusion_ms}ms</span>
        <span class="bg-gray-800/80 px-2 py-0.5 rounded text-gray-300">Rerank: ${lat.reranking_ms}ms</span>
        <span class="bg-gray-800/80 px-2 py-0.5 rounded text-gray-300">Gen: ${lat.generation_ms}ms</span>
        <span class="bg-indigo-900/60 text-indigo-300 px-2 py-0.5 rounded border border-indigo-700 font-bold">Total: ${lat.total_pipeline_ms}ms</span>
      `;

      // Render retrieved chunks
      const chunksList = document.getElementById('retrievedChunksList');
      chunksList.innerHTML = '';

      data.retrieved_chunks.forEach((chunk, idx) => {
        const title = chunk.metadata.title || `Doc ${chunk.document_id}`;
        
        let badges = [];
        if (chunk.dense_rank) {
          badges.push(`<span class="bg-blue-950/80 text-blue-300 border border-blue-800/60 px-2 py-0.5 rounded font-mono text-[11px]">Dense Rank #${chunk.dense_rank} (${chunk.dense_score})</span>`);
        }
        if (chunk.sparse_rank) {
          badges.push(`<span class="bg-amber-950/80 text-amber-300 border border-amber-800/60 px-2 py-0.5 rounded font-mono text-[11px]">BM25 Rank #${chunk.sparse_rank} (${chunk.sparse_score})</span>`);
        }
        if (chunk.rrf_score) {
          badges.push(`<span class="bg-purple-950/80 text-purple-300 border border-purple-800/60 px-2 py-0.5 rounded font-mono text-[11px]">RRF: ${chunk.rrf_score}</span>`);
        }
        if (chunk.rerank_score) {
          badges.push(`<span class="bg-emerald-950/80 text-emerald-300 border border-emerald-800/60 px-2 py-0.5 rounded font-mono text-[11px] font-bold">Cross-Encoder: ${chunk.rerank_score}</span>`);
        }

        const card = document.createElement('div');
        card.className = "glass p-4 rounded-xl border border-gray-800 hover:border-gray-700 transition space-y-2";
        card.innerHTML = `
          <div class="flex flex-wrap justify-between items-center gap-2">
            <span class="text-xs font-bold text-gray-200">[${idx + 1}] ${title}</span>
            <div class="flex flex-wrap gap-1.5">${badges.join('')}</div>
          </div>
          <p class="text-xs text-gray-300 leading-relaxed font-sans">${chunk.text}</p>
        `;
        chunksList.appendChild(card);
      });
    }

    async function triggerBenchmark() {
      const btn = document.getElementById('btnRunBenchmark');
      btn.disabled = true;
      btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running Benchmark...`;

      try {
        const res = await fetch('/api/v1/benchmark', { method: 'POST' });
        const report = await res.json();
        renderBenchmarkReport(report);
      } catch (err) {
        alert("Benchmark error: " + err);
      } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-play"></i> Run Quantitative Benchmark`;
      }
    }

    async function fetchLatestBenchmark() {
      try {
        const res = await fetch('/api/v1/benchmark/latest');
        if (res.ok) {
          const report = await res.json();
          renderBenchmarkReport(report);
        }
      } catch (err) {
        console.log("No existing benchmark report:", err);
      }
    }

    function renderBenchmarkReport(report) {
      document.getElementById('benchmarkTimestamp').innerText = `Run ID: ${report.run_id} | Samples: ${report.sample_count}`;
      const s = report.summary;
      const n = s.naive;
      const h = s.hybrid;
      const r = s.reranked;
      const dh = s.delta_hybrid_vs_naive;
      const dr = s.delta_reranked_vs_naive;

      const formatDelta = (val, isLatency = false) => {
        const sign = val > 0 ? '+' : '';
        const color = isLatency 
          ? (val < 0 ? 'text-emerald-400' : 'text-gray-400')
          : (val > 0 ? 'text-emerald-400' : (val < 0 ? 'text-rose-400' : 'text-gray-400'));
        return `<span class="${color}">${sign}${val}${isLatency ? 'ms' : ''}</span>`;
      };

      const rows = [
        { name: "Faithfulness (Groundedness)", n: n.faithfulness, h: h.faithfulness, r: r.faithfulness, dh: dh.faithfulness, dr: dr.faithfulness },
        { name: "Context Precision@K", n: n.context_precision, h: h.context_precision, r: r.context_precision, dh: dh.context_precision, dr: dr.context_precision },
        { name: "Context Recall", n: n.context_recall, h: h.context_recall, r: r.context_recall, dh: dh.context_recall, dr: dr.context_recall },
        { name: "Hit Rate@K", n: n.hit_rate, h: h.hit_rate, r: r.hit_rate, dh: dh.hit_rate, dr: dr.hit_rate },
        { name: "MRR (Mean Reciprocal Rank)", n: n.mrr, h: h.mrr, r: r.mrr, dh: dh.mrr, dr: dr.mrr },
        { name: "Answer Relevancy", n: n.answer_relevancy, h: h.answer_relevancy, r: r.answer_relevancy, dh: dh.answer_relevancy, dr: dr.answer_relevancy },
        { name: "Latency P50 (Median)", n: `${n.latency_p50_ms} ms`, h: `${h.latency_p50_ms} ms`, r: `${r.latency_p50_ms} ms`, dh: dh.latency_p50_ms, dr: dr.latency_p50_ms, isLat: true },
        { name: "Latency P95", n: `${n.latency_p95_ms} ms`, h: `${h.latency_p95_ms} ms`, r: `${r.latency_p95_ms} ms`, dh: '-', dr: '-', raw: true },
        { name: "Latency Mean", n: `${n.latency_mean_ms} ms`, h: `${h.latency_mean_ms} ms`, r: `${r.latency_mean_ms} ms`, dh: '-', dr: '-', raw: true },
      ];

      const tbody = document.getElementById('benchmarkMatrixBody');
      tbody.innerHTML = rows.map(row => `
        <tr class="hover:bg-gray-900/40">
          <td class="py-2.5 px-4 font-bold text-gray-200">${row.name}</td>
          <td class="py-2.5 px-4">${row.n}</td>
          <td class="py-2.5 px-4">${row.h}</td>
          <td class="py-2.5 px-4 font-bold text-indigo-300">${row.r}</td>
          <td class="py-2.5 px-4">${row.raw ? row.dh : formatDelta(row.dh, row.isLat)}</td>
          <td class="py-2.5 px-4 font-bold">${row.raw ? row.dr : formatDelta(row.dr, row.isLat)}</td>
        </tr>
      `).join('');

      renderBars('barPrecision', n.context_precision, h.context_precision, r.context_precision, 'bg-indigo-500');
      renderBars('barFaithfulness', n.faithfulness, h.faithfulness, r.faithfulness, 'bg-emerald-500');
      renderBars('barMRR', n.mrr, h.mrr, r.mrr, 'bg-amber-500');
      renderLatencyBars('barLatency', n.latency_p50_ms, h.latency_p50_ms, r.latency_p50_ms);
    }

    function renderBars(containerId, nVal, hVal, rVal, colorClass) {
      const container = document.getElementById(containerId);
      container.innerHTML = `
        <div class="space-y-1">
          <div class="flex justify-between text-[11px] text-gray-400 font-mono"><span>Naive (Dense)</span><span>${nVal}</span></div>
          <div class="w-full bg-gray-900 rounded-full h-2"><div class="${colorClass} opacity-60 h-2 rounded-full" style="width: ${nVal * 100}%"></div></div>
        </div>
        <div class="space-y-1">
          <div class="flex justify-between text-[11px] text-gray-400 font-mono"><span>Hybrid (BM25 + Dense)</span><span>${hVal}</span></div>
          <div class="w-full bg-gray-900 rounded-full h-2"><div class="${colorClass} opacity-80 h-2 rounded-full" style="width: ${hVal * 100}%"></div></div>
        </div>
        <div class="space-y-1">
          <div class="flex justify-between text-[11px] text-indigo-300 font-bold font-mono"><span>Reranked (Cross-Encoder)</span><span>${rVal}</span></div>
          <div class="w-full bg-gray-900 rounded-full h-2"><div class="${colorClass} h-2 rounded-full shadow-sm" style="width: ${rVal * 100}%"></div></div>
        </div>
      `;
    }

    function renderLatencyBars(containerId, nVal, hVal, rVal) {
      const maxVal = Math.max(nVal, hVal, rVal, 1.0) * 1.15;
      const container = document.getElementById(containerId);
      container.innerHTML = `
        <div class="space-y-1">
          <div class="flex justify-between text-[11px] text-gray-400 font-mono"><span>Naive</span><span>${nVal}ms</span></div>
          <div class="w-full bg-gray-900 rounded-full h-2"><div class="bg-cyan-500 opacity-60 h-2 rounded-full" style="width: ${(nVal / maxVal) * 100}%"></div></div>
        </div>
        <div class="space-y-1">
          <div class="flex justify-between text-[11px] text-gray-400 font-mono"><span>Hybrid</span><span>${hVal}ms</span></div>
          <div class="w-full bg-gray-900 rounded-full h-2"><div class="bg-cyan-500 opacity-80 h-2 rounded-full" style="width: ${(hVal / maxVal) * 100}%"></div></div>
        </div>
        <div class="space-y-1">
          <div class="flex justify-between text-[11px] text-cyan-300 font-bold font-mono"><span>Reranked</span><span>${rVal}ms</span></div>
          <div class="w-full bg-gray-900 rounded-full h-2"><div class="bg-cyan-400 h-2 rounded-full shadow-sm" style="width: ${(rVal / maxVal) * 100}%"></div></div>
        </div>
      `;
    }

    async function ingestCustomDoc() {
      const title = document.getElementById('docTitle').value.trim();
      const text = document.getElementById('docText').value.trim();
      if (!text) {
        alert("Please enter document body text.");
        return;
      }
      try {
        const res = await fetch('/api/v1/documents', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            documents: [{ title: title || "Untitled", text: text }]
          })
        });
        const data = await res.json();
        alert(`Ingested! Generated ${data.total_chunks} chunks in ${data.duration_ms}ms.`);
        document.getElementById('docTitle').value = '';
        document.getElementById('docText').value = '';
        updateStats();
      } catch (err) {
        alert("Ingest error: " + err);
      }
    }

    async function clearAllIndices() {
      if (!confirm("Are you sure you want to clear all document indices?")) return;
      try {
        await fetch('/api/v1/clear', { method: 'POST' });
        alert("All indices cleared.");
        updateStats();
      } catch (err) {
        alert("Clear error: " + err);
      }
    }

    // Auto-init
    updateStats();
  </script>
</body>
</html>
"""

@ui_router.get("/", response_class=HTMLResponse)
@ui_router.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    return HTMLResponse(content=HTML_CONTENT)
