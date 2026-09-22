from typing import Optional, Dict, Any
from ..rag.schemas import DocumentMetadata


class MetadataExtractor:
    """Extracts and standardizes metadata for documents and chunks."""

    @staticmethod
    def create_metadata(
        document_id: str,
        title: Optional[str] = None,
        source: str = "2WikiMultiHopQA",
        section: Optional[str] = None,
        page: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> DocumentMetadata:
        return DocumentMetadata(
            document_id=document_id,
            title=title or document_id,
            source=source,
            section=section,
            page=page,
            extra=extra or {},
        )
