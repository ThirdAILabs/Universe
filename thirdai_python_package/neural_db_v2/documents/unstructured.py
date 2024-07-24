from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from thirdai.neural_db.parsing_utils.unstructured_parse import (
    EmlParse,
    PptxParse,
    TxtParse,
    UnstructuredParse,
)

from ..core.documents import Document
from ..core.types import NewChunkBatch
from .utils import join_metadata, series_from_value


class Unstructured(Document):
    def __init__(
        self,
        path: str,
        parser: UnstructuredParse,
        chunk_metadata_columns: List[str],
        doc_metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ):
        super().__init__(doc_id=doc_id)

        self.path = path
        self.parser = parser
        self.chunk_metadata_columns = chunk_metadata_columns
        self.doc_metadata = doc_metadata

    def chunks(self) -> Iterable[NewChunkBatch]:
        parser = self.parser(self.path)

        elements, success = parser.process_elements()

        if not success:
            raise ValueError(f"Could not read file: {self.path}")

        contents = parser.create_train_df(elements)

        text = contents["para"]

        metadata = join_metadata(
            n_rows=len(text),
            chunk_metadata=contents[self.chunk_metadata_columns],
            doc_metadata=self.doc_metadata,
        )

        return [
            NewChunkBatch(
                text=text,
                keywords=series_from_value("", len(text)),
                metadata=metadata,
                document=contents["filename"],
            )
        ]


class PPTX(Unstructured):
    def __init__(
        self,
        path: str,
        doc_metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ):
        super().__init__(
            path=path,
            parser=PptxParse,
            chunk_metadata_columns=["filetype", "page"],
            doc_metadata=doc_metadata,
            doc_id=doc_id,
        )


class TextFile(Unstructured):
    def __init__(
        self,
        path: str,
        doc_metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ):
        super().__init__(
            path=path,
            parser=TxtParse,
            chunk_metadata_columns=["filetype"],
            doc_metadata=doc_metadata,
            doc_id=doc_id,
        )


class Email(Unstructured):
    def __init__(
        self,
        path: str,
        doc_metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ):
        super().__init__(
            path=path,
            parser=EmlParse,
            chunk_metadata_columns=["filetype", "subject", "sent_from", "sent_to"],
            doc_metadata=doc_metadata,
            doc_id=doc_id,
        )
