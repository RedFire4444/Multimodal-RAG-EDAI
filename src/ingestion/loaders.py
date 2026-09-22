import os
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from ..rag.schemas import RawDocument


class DocumentLoader:
    """Loads text from multiple file types (.pdf, .docx, .txt, .md, .json, .csv)."""

    @staticmethod
    def read_pdf(file_path: Union[str, Path]) -> str:
        """Extract text from PDF using PyMuPDF (or pypdf fallback)."""
        file_path = Path(file_path)
        try:
            import pymupdf
            doc = pymupdf.open(str(file_path))
            pages_text = []
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():
                    pages_text.append(f"--- Page {page_num} ---\n{text}")
            return "\n\n".join(pages_text)
        except ImportError:
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                pages_text = [
                    f"--- Page {i+1} ---\n{page.extract_text() or ''}"
                    for i, page in enumerate(reader.pages)
                    if page.extract_text()
                ]
                return "\n\n".join(pages_text)
            except Exception:
                return ""
        except Exception:
            return ""

    @staticmethod
    def read_docx(file_path: Union[str, Path]) -> str:
        """Extract text from docx via XML extraction."""
        file_path = Path(file_path)
        try:
            with zipfile.ZipFile(file_path) as z:
                xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = []
            for p in tree.iterfind(".//w:p", ns):
                texts = [node.text for node in p.iterfind(".//w:t", ns) if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            return "\n\n".join(paragraphs)
        except Exception:
            return ""

    @classmethod
    def load_file(cls, file_path: Union[str, Path]) -> Optional[RawDocument]:
        """Loads a single document file into a RawDocument."""
        file_path = Path(file_path)
        if not file_path.exists() or not file_path.is_file():
            return None

        suffix = file_path.suffix.lower()
        title = file_path.name
        doc_id = f"doc_{file_path.stem}"
        text = ""

        if suffix in (".txt", ".md", ".markdown", ".csv"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
            except Exception:
                return None

        elif suffix == ".pdf":
            text = cls.read_pdf(file_path)

        elif suffix == ".docx":
            text = cls.read_docx(file_path)

        elif suffix in (".json", ".jsonl"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    text = data.get("text") or data.get("content") or json.dumps(data, indent=2)
                elif isinstance(data, list):
                    text = "\n\n".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in data
                    )
            except Exception:
                return None

        if not text:
            return None

        return RawDocument(
            document_id=doc_id,
            title=title,
            text=text,
            metadata={
                "source": str(file_path.name),
                "file_type": suffix,
                "file_size_bytes": file_path.stat().st_size,
            },
        )


class DirectoryKnowledgeLoader:
    """Recursively scans and loads all supported knowledge base documents from a directory."""

    @classmethod
    def load_directory(
        cls,
        directory_path: Union[str, Path] = "data",
        recursive: bool = True,
        limit_per_file: Optional[int] = None,
    ) -> List[RawDocument]:
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        documents: List[RawDocument] = []
        pattern = "**/*" if recursive else "*"
        all_files = sorted([p for p in dir_path.glob(pattern) if p.is_file() and not p.name.startswith(".")])

        for file_path in all_files:
            doc = DocumentLoader.load_file(file_path)
            if doc:
                documents.append(doc)

        return documents
