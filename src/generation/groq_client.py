import os
from typing import List, Dict, Any, Optional


class GroqLLMClient:
    """Wrapper for Groq API fast inference using Groq official SDK or langchain-groq."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.0):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError("groq package is not installed. Please install it via `pip install groq`")
        return self._client

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None) -> str:
        tokens = list(self.generate_stream(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens))
        return "".join(tokens).strip()

    def generate_stream(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None):
        """Yields generated token chunks in real-time."""
        if not self.api_key:
            yield "[Error: GROQ_API_KEY not configured. Set GROQ_API_KEY in .env]"
            return

        token_limit = max_tokens or int(os.getenv("MAX_OUTPUT_TOKENS", "512"))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        candidate_models = [
            self.model,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ]

        for m in dict.fromkeys(candidate_models):
            try:
                stream = self.client.chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=token_limit,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yield delta
                return
            except Exception:
                continue

        yield "[Generation Error: Failed to generate response from available Groq models]"


