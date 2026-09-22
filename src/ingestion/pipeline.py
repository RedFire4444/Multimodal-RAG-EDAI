import os
from pathlib import Path
from typing import List, Union, Optional
from ..rag.schemas import RawDocument, Chunk
from .loaders import DocumentLoader, DirectoryKnowledgeLoader
from .cleaner import TextCleaner
from .chunker import RecursiveChunker


class IngestionPipeline:
    """End-to-end ingestion pipeline: Load -> Clean -> Chunk -> Extract Metadata."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "200"))
        self.chunker = RecursiveChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.cleaner = TextCleaner()

    def process_directory(
        self,
        directory_path: Union[str, Path] = "data",
        recursive: bool = True,
        limit_per_file: Optional[int] = None,
    ) -> List[Chunk]:
        """Loads and chunks all documents from a directory (e.g. data/ folder)."""
        raw_docs = DirectoryKnowledgeLoader.load_directory(
            directory_path=directory_path,
            recursive=recursive,
            limit_per_file=limit_per_file,
        )
        return self.process_documents(raw_docs)

    def process_file(self, file_path: Union[str, Path], limit: Optional[int] = None) -> List[Chunk]:
        file_path = Path(file_path)
        if file_path.is_dir():
            return self.process_directory(file_path, limit_per_file=limit)

        doc = DocumentLoader.load_file(file_path)
        if not doc:
            return []
        return self.process_documents([doc])

    def process_documents(self, documents: List[RawDocument]) -> List[Chunk]:
        chunks = self.chunker.chunk_documents(documents)
        return chunks
