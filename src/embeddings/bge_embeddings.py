from typing import List, Optional
import os


class SentenceTransformerEmbeddings:
    """Embedding model wrapper using sentence-transformers (default: all-MiniLM-L6-v2).
    Implements class-level caching so model weights are loaded only once in memory.
    """

    _CACHE = {}

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._dimension = None

    @property
    def model(self):
        cache_key = (self.model_name, self.device)
        if cache_key not in self._CACHE:
            try:
                from sentence_transformers import SentenceTransformer
                self._CACHE[cache_key] = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. Please install it via `pip install sentence-transformers`"
                )
        return self._CACHE[cache_key]


    @property
    def dimension(self) -> int:
        if self._dimension is None:
            try:
                if hasattr(self.model, "get_embedding_dimension"):
                    self._dimension = self.model.get_embedding_dimension()
                elif hasattr(self.model, "get_sentence_embedding_dimension"):
                    self._dimension = self.model.get_sentence_embedding_dimension()
                else:
                    self._dimension = int(os.getenv("EMBEDDING_DIM", "384"))
            except Exception:
                self._dimension = int(os.getenv("EMBEDDING_DIM", "384"))
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=len(texts) > 50,
        )
        return embeddings.tolist()

    _QUERY_CACHE = {}

    def embed_query(self, text: str) -> List[float]:
        """Embeds query text into dense vector, caching result to embed only once."""
        if not text:
            return []

        cache_key = (self.model_name, text, self.normalize_embeddings)
        if cache_key in self._QUERY_CACHE:
            return self._QUERY_CACHE[cache_key]

        embedding = self.model.encode(
            [text],
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )[0].tolist()

        # Cache query vector (cap cache size at 4096 entries)
        if len(self._QUERY_CACHE) >= 4096:
            self._QUERY_CACHE.pop(next(iter(self._QUERY_CACHE)))

        self._QUERY_CACHE[cache_key] = embedding
        return embedding



# Alias for backward compatibility
BGEM3Embeddings = SentenceTransformerEmbeddings
