from typing import Iterable, List

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
        custom_id=None,
    ):
        super().__init__()

        self.document_name = document_name
        self.text = pd.Series(text)
        self.chunk_metadata = (
            pd.DataFrame.from_records(chunk_metadata) if chunk_metadata else None
        )
        self.doc_metadata = doc_metadata
        self.custom_id = custom_id

    def chunks(self) -> Iterable[NewChunkBatch]:
        metadata = join_metadata(
            n_rows=len(self.text),
            chunk_metadata=self.chunk_metadata,
            doc_metadata=self.doc_metadata,
        )

        return [
            NewChunkBatch(
                custom_id=pd.Series(self.custom_id) if self.custom_id else None,
                text=self.text,
                keywords=series_from_value("", len(self.text)),
                metadata=metadata,
                document=series_from_value(self.document_name, len(self.text)),
            )
        ]
