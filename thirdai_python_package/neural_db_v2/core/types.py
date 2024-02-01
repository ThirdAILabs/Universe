from dataclasses import dataclass


# We typedef doc ID to anticipate switching over to string IDs
DocId = int


@dataclass
class Document:
    doc_id: DocId
    text: str
    keywords: str
    metadata: dict
