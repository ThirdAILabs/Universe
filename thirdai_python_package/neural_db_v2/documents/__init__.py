from ..core.documents import Document
from .csv import CSV
from .docx import DOCX
from .in_memory_text import InMemoryText
from .pdf import PDF
from .unstructured import PPTX, Email, TextFile
from .url import URL


def document_by_name(name: str, **kwargs) -> Document:
    # TODO: Add options here
    raise ValueError(f"Invalid document name {name}")
