"""
Anti-Stampede SingleFlight Engine.

Coalesces concurrent identical or semantically duplicate query requests
onto a single in-flight asynchronous computation promise, eliminating
cache stampedes (dogpiling) under high-concurrency production spikes.
"""
import asyncio
import logging
from typing import Dict, Any, Callable, TypeVar, Coroutine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SingleFlight:
    """
    Asynchronous request deduplication coordinator.
    Guarantees only 1 downstream execution per unique key during concurrent bursts.
    """

    def __init__(self):
        self._in_flight: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._total_calls = 0
        self._coalesced_calls = 0

    async def execute(self, key: str, func: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """
        Execute func only if key is not already actively in-flight.
        Otherwise, await the active in-flight worker's resolution.

        Args:
            key: Deduplication key (e.g., query hash or semantic cluster ID).
            func: Async nullary callable to execute if primary caller.

        Returns:
            Result from func (either executed directly or received from leader).
        """
        async with self._lock:
            self._total_calls += 1
            if key in self._in_flight:
                # Concurrent caller attaches to active in-flight execution
                future = self._in_flight[key]
                self._coalesced_calls += 1
                wait = True
            else:
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                self._in_flight[key] = future
                wait = False

        if wait:
            return await future

        try:
            result = await func()
            future.set_result(result)
            return result
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            async with self._lock:
                if key in self._in_flight:
                    del self._in_flight[key]

    def stats(self) -> Dict[str, Any]:
        """Telemetry on stampede suppression efficiency."""
        return {
            "total_calls": self._total_calls,
            "coalesced_calls": self._coalesced_calls,
            "suppression_ratio": (self._coalesced_calls / max(1, self._total_calls)),
            "active_in_flight_keys": len(self._in_flight)
        }
