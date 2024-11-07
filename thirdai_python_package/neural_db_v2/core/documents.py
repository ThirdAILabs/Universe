import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional

from .types import NewChunkBatch


def validate_doc_metadata(doc_metadata):
    if doc_metadata is None:
        return
    for _, value in doc_metadata.items():
        if isinstance(value, list):
            if len(value) > 0:
                first_elem_type = type(value[0])
                if not all(isinstance(elem, first_elem_type) for elem in value):
                    raise ValueError(
                        f"Multivalue doc metadata does not have a consistent type."
                    )


class Document(ABC):
    def __init__(
        self, doc_id: Optional[str], doc_metadata: Optional[Dict[str, Any]] = None
    ):
        self._doc_id = doc_id or str(uuid.uuid4())

        validate_doc_metadata(doc_metadata)
        self.doc_metadata = doc_metadata

    @abstractmethod
    def chunks(self) -> Iterable[NewChunkBatch]:
        raise NotImplementedError

    def doc_id(self) -> str:
        return self._doc_id

    def __iter__(self) -> Iterable[NewChunkBatch]:
        return iter(self.chunks())
