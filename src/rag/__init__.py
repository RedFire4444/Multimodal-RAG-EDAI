from .schemas import (
    Chunk,
    RawDocument,
    RetrievedChunk,
    RAGQueryRequest,
    RAGResponse,
    EvaluationSample,
)

__all__ = [
    "Chunk",
    "RawDocument",
    "RetrievedChunk",
    "RAGQueryRequest",
    "RAGResponse",
    "EvaluationSample",
    "RAGPipeline",
    "RAGService",
]


def __getattr__(name: str):
    if name == "RAGPipeline":
        from .pipeline import RAGPipeline
        return RAGPipeline
    elif name == "RAGService":
        from .service import RAGService
        return RAGService
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
