from typing import Any, Dict, Iterable, Optional

import pandas as pd

from ..core.documents import Document
from ..core.types import NewChunkBatch
from .utils import join_metadata, series_from_value


class InMemoryText(Document):
    def __init__(
        self,
        document_name,
        text=[],
        chunk_metadata=None,
        doc_metadata=None,
        doc_id: Optional[str] = None,
    ):
        super().__init__(doc_id=doc_id)

        self.document_name = document_name
        self.text = pd.Series(text)
        self.chunk_metadata = (
            pd.DataFrame.from_records(chunk_metadata) if chunk_metadata else None
        )
        self.doc_metadata = doc_metadata

    def chunks(self) -> Iterable[NewChunkBatch]:
        metadata = join_metadata(
            n_rows=len(self.text),
            chunk_metadata=self.chunk_metadata,
            doc_metadata=self.doc_metadata,
        )

        return [
            NewChunkBatch(
                text=self.text,
                keywords=series_from_value("", len(self.text)),
                metadata=metadata,
                document=series_from_value(self.document_name, len(self.text)),
            )
        ]


class PrebatchedDoc(Document):
    def __init__(self, chunks: Iterable[NewChunkBatch], doc_id: Optional[str] = None):
        super().__init__(doc_id=doc_id)
        self._chunks = chunks

    def chunks(self) -> Iterable[NewChunkBatch]:
        return self._chunks
