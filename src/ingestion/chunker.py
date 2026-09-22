import os
import re
from typing import List, Optional
from ..rag.schemas import RawDocument, Chunk
from .cleaner import TextCleaner


class NativeRecursiveCharacterTextSplitter:
    """A clean, standalone implementation of recursive character text splitting."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        _separator = "" if separator == "" else separator

        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, _separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(other_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits, _separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs: List[str] = []
        current_doc: List[str] = []
        total_len = 0

        for d in splits:
            _len = len(d)
            if total_len + _len + (len(separator) if current_doc else 0) > self.chunk_size:
                if total_len > 0:
                    doc = separator.join(current_doc).strip()
                    if doc:
                        docs.append(doc)
                    # Handle overlap
                    while total_len > self.chunk_overlap or (
                        total_len + _len + len(separator) > self.chunk_size and total_len > 0
                    ):
                        if not current_doc:
                            break
                        popped = current_doc.pop(0)
                        total_len -= len(popped) + (len(separator) if current_doc else 0)
                current_doc.append(d)
                total_len = sum(len(x) for x in current_doc) + len(separator) * (len(current_doc) - 1)
            else:
                current_doc.append(d)
                total_len += _len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc = separator.join(current_doc).strip()
            if doc:
                docs.append(doc)

        return docs

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self.separators)


class RecursiveChunker:
    """Splits documents into overlapping chunks using recursive character splitting while preserving metadata."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "200"))
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

        self.splitter = NativeRecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def chunk_document(self, doc: RawDocument) -> List[Chunk]:
        cleaned_text = TextCleaner.clean(doc.text)
        if not cleaned_text:
            return []

        text_splits = self.splitter.split_text(cleaned_text)
        chunks: List[Chunk] = []

        for idx, text_split in enumerate(text_splits):
            chunk_id = f"{doc.document_id}_chunk_{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=doc.document_id,
                    text=text_split,
                    title=doc.title,
                    source=doc.metadata.get("source", "knowledge_base"),
                    section=doc.metadata.get("section"),
                    page=doc.metadata.get("page"),
                    metadata={
                        **doc.metadata,
                        "chunk_index": idx,
                        "total_chunks": len(text_splits),
                    },
                )
            )

        return chunks

    def chunk_documents(self, docs: List[RawDocument]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
