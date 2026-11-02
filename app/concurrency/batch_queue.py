"""
Dynamic Sliding-Window Micro-Batching Engine.

Gathers concurrent asynchronous reranking requests into variable-sized mini-batches
within a sliding micro-window (Delta_t = 8ms) up to B_max = 32.
Offloads heavy SIMD / ONNX / PyTorch computation to a dedicated worker thread pool
via loop.run_in_executor(), preventing Python GIL event-loop starvation.
"""
import asyncio
import time
import logging
from typing import List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class QueryItem:
    query: str
    passage: str
    future: asyncio.Future
    created_at: float


class DynamicMicroBatcher:
    """
    Sliding-window micro-batcher coordinating asynchronous I/O with batched SIMD inference.
    """

    def __init__(
        self,
        inference_fn: Callable[[List[Tuple[str, str]]], Any],
        max_batch_size: int = 32,
        window_ms: float = 8.0,
        max_queue_size: int = 2048,
        worker_threads: int = 4
    ):
        """
        Args:
            inference_fn: Synchronous function taking List[(query, passage)] and returning scores array.
            max_batch_size: Maximum batch size before triggering immediate inference.
            window_ms: Sliding window time budget in milliseconds.
            max_queue_size: Capacity bound to enforce backpressure under extreme load.
            worker_threads: Thread pool size for tensor computation.
        """
        self.inference_fn = inference_fn
        self.max_batch_size = max_batch_size
        self.window_sec = window_ms / 1000.0
        self.max_queue_size = max_queue_size
        self.queue: asyncio.Queue[QueryItem] = asyncio.Queue(maxsize=max_queue_size)
        self.executor = ThreadPoolExecutor(max_workers=worker_threads, thread_name_prefix="inference_worker")
        
        self._running = False
        self._batcher_task: Optional[asyncio.Task] = None
        self._total_batches = 0
        self._total_queries = 0

    async def start(self):
        """Start the background collector loop."""
        if not self._running:
            self._running = True
            self._batcher_task = asyncio.create_task(self._collector_loop())
            logger.info(
                f"DynamicMicroBatcher started (window={self.window_sec*1000:.1f}ms, "
                f"B_max={self.max_batch_size}, threads={self.executor._max_workers})."
            )

    async def stop(self):
        """Gracefully drain and terminate batcher."""
        self._running = False
        if self._batcher_task:
            self._batcher_task.cancel()
            try:
                await self._batcher_task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=False)
        logger.info("DynamicMicroBatcher stopped.")

    async def predict(self, query: str, passage: str) -> float:
        """
        Asynchronous caller interface.
        Enqueues request and yields control until batch inference resolves future.
        """
        if not self._running:
            await self.start()

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        item = QueryItem(
            query=query,
            passage=passage,
            future=fut,
            created_at=time.perf_counter()
        )

        try:
            # Non-blocking enqueue with backpressure
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            raise RuntimeError("Inference queue saturated: Backpressure triggered (HTTP 429/503).")

        return await fut

    async def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Enqueues multiple pairs concurrently and awaits resolution of all futures.
        """
        coros = [self.predict(q, p) for q, p in pairs]
        return await asyncio.gather(*coros)

    async def _collector_loop(self):
        """Sliding-window collector accumulating requests up to timeout or B_max."""
        loop = asyncio.get_running_loop()

        while self._running:
            try:
                # 1. Await initial request
                first_item = await self.queue.get()
            except asyncio.CancelledError:
                break

            batch: List[QueryItem] = [first_item]
            t_window_start = time.perf_counter()

            # 2. Collect additional requests within sliding window
            while len(batch) < self.max_batch_size:
                elapsed = time.perf_counter() - t_window_start
                remaining = self.window_sec - elapsed
                if remaining <= 0:
                    break

                try:
                    next_item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch.append(next_item)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    break

            # 3. Offload execution to worker thread pool
            pairs = [(item.query, item.passage) for item in batch]
            self._total_batches += 1
            self._total_queries += len(batch)

            try:
                # Offload synchronous tensor inference to thread pool
                scores = await loop.run_in_executor(
                    self.executor,
                    self.inference_fn,
                    pairs
                )

                # 4. Resolve futures
                for item, score in zip(batch, scores):
                    if not item.future.cancelled():
                        item.future.set_result(float(score))

            except Exception as exc:
                logger.error(f"Inference execution failed in worker: {exc}")
                for item in batch:
                    if not item.future.cancelled():
                        item.future.set_exception(exc)
            finally:
                for _ in batch:
                    self.queue.task_done()

    def stats(self) -> dict:
        """Runtime telemetry diagnostics."""
        return {
            "total_batches_executed": self._total_batches,
            "total_queries_processed": self._total_queries,
            "avg_batch_utilization": (self._total_queries / max(1, self._total_batches)),
            "queue_pending_items": self.queue.qsize(),
            "max_batch_size": self.max_batch_size,
            "window_ms": self.window_sec * 1000.0
        }
