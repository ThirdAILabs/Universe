from dataclasses import dataclass


DocId = int


@dataclass
class Document:
    doc_id: DocId
    text: str
    keywords: str
    metadata: dict
