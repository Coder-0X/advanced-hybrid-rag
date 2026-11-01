/**
 * Advanced Hybrid RAG Engine - Interactive Web UI Controller
 */
let currentMode = 'reranked';

function switchTab(tabId) {
  ['playground', 'benchmark', 'documents'].forEach(t => {
    const tabEl = document.getElementById(`tab-${t}`);
    if (tabEl) tabEl.classList.add('hidden');
    const btn = document.getElementById(`tabBtn-${t}`);
    if (btn) {
      btn.classList.remove('border-indigo-500', 'text-indigo-400');
      btn.classList.add('border-transparent', 'text-gray-400');
    }
  });

  const activeTabEl = document.getElementById(`tab-${tabId}`);
  if (activeTabEl) activeTabEl.classList.remove('hidden');

  const activeBtn = document.getElementById(`tabBtn-${tabId}`);
  if (activeBtn) {
    activeBtn.classList.add('border-indigo-500', 'text-indigo-400');
    activeBtn.classList.remove('border-transparent', 'text-gray-400');
  }

  if (tabId === 'benchmark') {
    fetchLatestBenchmark();
  }
}

function setMode(mode) {
  currentMode = mode;
  ['naive', 'hybrid', 'reranked'].forEach(m => {
    const btn = document.getElementById(`mode-${m}`);
    if (!btn) return;
    if (m === mode) {
      btn.className = "px-3.5 py-1.5 rounded-lg text-xs font-medium bg-indigo-600 text-white shadow transition";
    } else {
      btn.className = "px-3.5 py-1.5 rounded-lg text-xs font-medium text-gray-400 hover:text-white transition";
    }
  });
}

function setQuery(q) {
  const queryInput = document.getElementById('queryInput');
  if (queryInput) {
    queryInput.value = q;
    executeQuery();
  }
}

async function updateStats() {
  try {
    const res = await fetch('/api/v1/stats');
    if (!res.ok) return;
    const data = await res.json();
    const countEl = document.getElementById('chunkCount');
    if (countEl) {
      countEl.innerText = data.total_chunks_vector_store;
    }
  } catch (err) {
    console.error("Failed to update index stats:", err);
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
  const queryInput = document.getElementById('queryInput');
  if (!queryInput) return;
  const query = queryInput.value.trim();
  if (!query) return;

  const btn = document.getElementById('btnSubmitQuery');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i>`;
  }

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
    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Query execution failed");
    }
    const data = await res.json();
    renderResults(data);
    updateStats();
  } catch (err) {
    alert("Query error: " + err);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<span>Query</span> <i class="fa-solid fa-paper-plane text-[10px]"></i>`;
    }
  }
}

function renderResults(data) {
  const container = document.getElementById('resultsContainer');
  if (container) container.classList.remove('hidden');

  const ansEl = document.getElementById('generatedAnswer');
  if (ansEl) ansEl.innerText = data.answer || "No answer generated.";

  const groundEl = document.getElementById('groundingScore');
  if (groundEl) {
    const groundingScore = data.grounding_score !== null ? data.grounding_score : 1.0;
    groundEl.innerText = groundingScore;
  }

  // Render latency breakdown
  const lat = data.latencies;
  const pills = document.getElementById('latencyPills');
  if (pills && lat) {
    pills.innerHTML = `
      <span class="bg-gray-800/80 px-2 py-0.5 rounded text-gray-300">Dense: ${lat.dense_retrieval_ms}ms</span>
      <span class="bg-gray-800/80 px-2 py-0.5 rounded text-gray-300">BM25: ${lat.sparse_retrieval_ms}ms</span>
      <span class="bg-gray-800/80 px-2 py-0.5 rounded text-gray-300">RRF: ${lat.fusion_ms}ms</span>
      <span class="bg-gray-800/80 px-2 py-0.5 rounded text-gray-300">Rerank: ${lat.reranking_ms}ms</span>
      <span class="bg-gray-800/80 px-2 py-0.5 rounded text-gray-300">Gen: ${lat.generation_ms}ms</span>
      <span class="bg-indigo-900/60 text-indigo-300 px-2 py-0.5 rounded border border-indigo-700 font-bold">Total: ${lat.total_pipeline_ms}ms</span>
    `;
  }

  // Render retrieved chunks
  const chunksList = document.getElementById('retrievedChunksList');
  if (chunksList && data.retrieved_chunks) {
    chunksList.innerHTML = '';
    data.retrieved_chunks.forEach((chunk, idx) => {
      const title = (chunk.metadata && chunk.metadata.title) ? chunk.metadata.title : `Doc ${chunk.document_id}`;

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
}

async function triggerBenchmark() {
  const btn = document.getElementById('btnRunBenchmark');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running Benchmark...`;
  }

  try {
    const res = await fetch('/api/v1/benchmark', { method: 'POST' });
    if (!res.ok) throw new Error("Benchmark failed");
    const report = await res.json();
    renderBenchmarkReport(report);
  } catch (err) {
    alert("Benchmark error: " + err);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<i class="fa-solid fa-play"></i> Run Quantitative Benchmark`;
    }
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
  const tsEl = document.getElementById('benchmarkTimestamp');
  if (tsEl) {
    tsEl.innerText = `Run ID: ${report.run_id} | Samples: ${report.sample_count}`;
  }
  const s = report.summary;
  if (!s) return;

  const n = s.naive;
  const h = s.hybrid;
  const r = s.reranked;
  const dh = s.delta_hybrid_vs_naive || {};
  const dr = s.delta_reranked_vs_naive || {};

  const formatDelta = (val, isLatency = false) => {
    if (val === undefined || val === null) return '-';
    const sign = val > 0 ? '+' : '';
    const color = isLatency
      ? (val < 0 ? 'text-emerald-400' : 'text-gray-400')
      : (val > 0 ? 'text-emerald-400' : (val < 0 ? 'text-rose-400' : 'text-gray-400'));
    return `<span class="${color}">${sign}${val}${isLatency ? 'ms' : ''}</span>`;
  };

  const rows = [
    { name: "Context Precision@K", n: n.context_precision, h: h.context_precision, r: r.context_precision, dh: dh.context_precision, dr: dr.context_precision },
    { name: "Context Recall", n: n.context_recall, h: h.context_recall, r: r.context_recall, dh: dh.context_recall, dr: dr.context_recall },
    { name: "Hit Rate@K", n: n.hit_rate, h: h.hit_rate, r: r.hit_rate, dh: dh.hit_rate, dr: dr.hit_rate },
    { name: "MRR (Mean Reciprocal Rank)", n: n.mrr, h: h.mrr, r: r.mrr, dh: dh.mrr, dr: dr.mrr },
    { name: "Faithfulness (Groundedness)", n: n.faithfulness, h: h.faithfulness, r: r.faithfulness, dh: dh.faithfulness, dr: dr.faithfulness },
    { name: "Answer Relevancy", n: n.answer_relevancy, h: h.answer_relevancy, r: r.answer_relevancy, dh: dh.answer_relevancy, dr: dr.answer_relevancy },
    { name: "Latency P50 (Median)", n: `${n.latency_p50_ms} ms`, h: `${h.latency_p50_ms} ms`, r: `${r.latency_p50_ms} ms`, dh: dh.latency_p50_ms, dr: dr.latency_p50_ms, isLat: true },
    { name: "Latency P95", n: `${n.latency_p95_ms} ms`, h: `${h.latency_p95_ms} ms`, r: `${r.latency_p95_ms} ms`, dh: '-', dr: '-', raw: true },
    { name: "Latency Mean", n: `${n.latency_mean_ms} ms`, h: `${h.latency_mean_ms} ms`, r: `${r.latency_mean_ms} ms`, dh: '-', dr: '-', raw: true },
  ];

  const tbody = document.getElementById('benchmarkMatrixBody');
  if (tbody) {
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
  }

  renderBars('barPrecision', n.context_precision, h.context_precision, r.context_precision, 'bg-indigo-500');
  renderBars('barFaithfulness', n.faithfulness, h.faithfulness, r.faithfulness, 'bg-emerald-500');
  renderBars('barMRR', n.mrr, h.mrr, r.mrr, 'bg-amber-500');
  renderLatencyBars('barLatency', n.latency_p50_ms, h.latency_p50_ms, r.latency_p50_ms);
}

function renderBars(containerId, nVal, hVal, rVal, colorClass) {
  const container = document.getElementById(containerId);
  if (!container) return;
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
  const container = document.getElementById(containerId);
  if (!container) return;
  const maxVal = Math.max(nVal, hVal, rVal, 1.0) * 1.15;
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

// Auto-initialize stats on load
document.addEventListener("DOMContentLoaded", () => {
  updateStats();
});
