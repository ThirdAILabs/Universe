from typing import Iterable, List

import pandas as pd

from ..core.documents import Document
from ..core.types import NewChunkBatch
from .utils import series_from_value


class InMemoryText(Document):
    def __init__(
        self,
        document_name,
        text=[],
        chunk_metadata=None,
        metadata=None,
    ):
        super().__init__()

        self.document_name = document_name
        self.text = pd.Series(text)
        self.chunk_metadata = (
            pd.DataFrame.from_records(chunk_metadata) if chunk_metadata else None
        )
        self.metadata = (
            pd.DataFrame.from_records([metadata] * len(self.text)) if metadata else None
        )

    def chunks(self) -> Iterable[NewChunkBatch]:
        if self.metadata is not None and self.chunk_metadata is not None:
            metadata = pd.concat([self.metadata, self.chunk_metadata], axis=1)
        elif self.metadata is not None:
            metadata = self.metadata
        elif self.chunk_metadata is not None:
            metadata = self.chunk_metadata
        else:
            metadata = None

        return [
            NewChunkBatch(
                custom_id=None,
                text=self.text,
                keywords=series_from_value("", len(self.text)),
                metadata=metadata,
                document=series_from_value(self.document_name, len(self.text)),
            )
        ]
