import re
import unicodedata


class TextCleaner:
    """Cleans and normalizes raw text for ingestion and chunking."""

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""

        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)

        # Replace consecutive whitespace and normalize line breaks
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove null bytes or non-printable control characters (except newline, tab)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        return text.strip()
