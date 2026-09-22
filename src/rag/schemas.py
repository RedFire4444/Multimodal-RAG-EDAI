from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    document_id: str
    title: Optional[str] = None
    source: Optional[str] = "knowledge_base"
    section: Optional[str] = None
    page: Optional[int] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    title: Optional[str] = None
    source: Optional[str] = "knowledge_base"
    section: Optional[str] = None
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RawDocument(BaseModel):
    document_id: str
    title: str
    text: str
    sentences: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    title: Optional[str] = None
    text: str
    score: float
    rank: int
    source_method: str = "hybrid"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 5


class RAGResponse(BaseModel):
    query: str
    answer: str
    evidence_chunks: List[RetrievedChunk]
    mode: str = "hybrid"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationSample(BaseModel):
    sample_id: str
    question: str
    gold_answer: str
    gold_documents: List[str] = Field(default_factory=list)
    gold_passages: List[str] = Field(default_factory=list)
    question_type: Optional[str] = None
