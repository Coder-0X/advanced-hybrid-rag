"""
FastAPI Dependency Injection Providers.
Ensures engine lifecycle is managed cleanly via application state (lifespan)
rather than global module-level singletons.
"""
from fastapi import Request
from app.core.pipeline import HybridRAGEngine

def get_engine(request: Request) -> HybridRAGEngine:
    """
    Retrieve the initialized HybridRAGEngine from the FastAPI application state.
    Provides lazy fallback initialization if accessed outside of standard lifespan.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        engine = HybridRAGEngine()
        request.app.state.engine = engine
    return engine
