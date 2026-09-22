from typing import List, Optional
from ..rag.schemas import RetrievedChunk
from .groq_client import GroqLLMClient
from .prompts import RAGPromptTemplates


class AnswerGenerator:
    """Coordinates prompt formulation and grounded answer generation with LLM."""

    def __init__(self, llm_client: Optional[GroqLLMClient] = None):
        self.llm_client = llm_client or GroqLLMClient()

    def generate_answer(self, query: str, evidence_chunks: List[RetrievedChunk]) -> str:
        system_prompt = RAGPromptTemplates.SYSTEM_PROMPT
        user_prompt = RAGPromptTemplates.format_user_prompt(query, evidence_chunks)
        return self.llm_client.generate(system_prompt=system_prompt, user_prompt=user_prompt)

    def generate_answer_stream(self, query: str, evidence_chunks: List[RetrievedChunk]):
        """Streams grounded answer tokens in real-time."""
        system_prompt = RAGPromptTemplates.SYSTEM_PROMPT
        user_prompt = RAGPromptTemplates.format_user_prompt(query, evidence_chunks)
        yield from self.llm_client.generate_stream(system_prompt=system_prompt, user_prompt=user_prompt)

