import os
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from ..rag.schemas import Chunk, RetrievedChunk
from ..vectorstore.qdrant_store import QdrantVectorStore


class BM25Retriever:
    """Sparse keyword retriever using persisted in-memory BM25Okapi or Qdrant Cloud bm25 inference.
    Guarantees BM25 is built only once via in-memory instance caching and persistent disk storage.
    """

    _MEMORY_CACHE: Dict[str, Any] = {}

    def __init__(
        self,
        vectorstore: Optional[QdrantVectorStore] = None,
        chunks: Optional[List[Chunk]] = None,
        cache_path: Optional[Union[str, Path]] = "data/.bm25_cache.pkl",
    ):
        self.vectorstore = vectorstore
        self.cache_path = Path(cache_path) if cache_path else None
        self.chunks: List[Chunk] = []
        self._bm25 = None
        self._tokenized_corpus = []

        cache_key = str(self.cache_path.resolve()) if self.cache_path else "default"

        # 1. Check in-memory process cache first (fastest, 0.00s)
        if cache_key in self._MEMORY_CACHE:
            cached_data = self._MEMORY_CACHE[cache_key]
            self.chunks = cached_data["chunks"]
            self._tokenized_corpus = cached_data["tokenized_corpus"]
            self._bm25 = cached_data["bm25"]
        elif chunks:
            # 2. If chunks provided, build once and cache
            self.index_chunks(chunks, save=True)
        elif self.cache_path and self.cache_path.exists():
            # 3. Load pre-built index from disk
            self.load_cache()

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        return re.findall(r"\b\w+\b", text)

    def index_chunks(self, chunks: List[Chunk], save: bool = True, force: bool = False):
        """Builds BM25 index only once unless force=True."""
        if not force and self._bm25 is not None and len(self.chunks) == len(chunks):
            return

        self.chunks = chunks
        try:
            from rank_bm25 import BM25Okapi
            self._tokenized_corpus = [self._tokenize(chunk.text) for chunk in chunks]
            self._bm25 = BM25Okapi(self._tokenized_corpus)

            # Store in process cache
            cache_key = str(self.cache_path.resolve()) if self.cache_path else "default"
            self._MEMORY_CACHE[cache_key] = {
                "chunks": self.chunks,
                "tokenized_corpus": self._tokenized_corpus,
                "bm25": self._bm25,
            }

            if save and self.cache_path:
                self.save_cache()
        except ImportError:
            pass

    def save_cache(self):
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(
                    {
                        "chunks": self.chunks,
                        "tokenized_corpus": self._tokenized_corpus,
                        "bm25": self._bm25,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception:
            pass

    def load_cache(self) -> bool:
        """Loads pre-built BM25 index from disk cache without re-tokenizing."""
        if not self.cache_path or not self.cache_path.exists():
            return False
        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
                self.chunks = data.get("chunks", [])
                self._tokenized_corpus = data.get("tokenized_corpus", [])
                self._bm25 = data.get("bm25")

            cache_key = str(self.cache_path.resolve()) if self.cache_path else "default"
            self._MEMORY_CACHE[cache_key] = {
                "chunks": self.chunks,
                "tokenized_corpus": self._tokenized_corpus,
                "bm25": self._bm25,
            }
            return bool(self._bm25)
        except Exception:
            return False


    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        # If Qdrant vectorstore is provided, attempt Qdrant cloud BM25 search first
        if self.vectorstore is not None:
            try:
                results = self.vectorstore.search_bm25(query=query, top_k=top_k)
                if results:
                    return results
            except Exception:
                pass

        # In-memory / persisted BM25 fallback
        if self._bm25 and self.chunks:
            tokenized_query = self._tokenize(query)
            doc_scores = self._bm25.get_scores(tokenized_query)

            sorted_indices = sorted(
                range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True
            )[:top_k]

            results: List[RetrievedChunk] = []
            for rank, idx in enumerate(sorted_indices, start=1):
                chunk = self.chunks[idx]
                results.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        title=chunk.title,
                        text=chunk.text,
                        score=float(doc_scores[idx]),
                        rank=rank,
                        source_method="bm25",
                        metadata=chunk.metadata,
                    )
                )
            return results

        return []

