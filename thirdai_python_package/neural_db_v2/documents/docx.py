from typing import Iterable

import pandas as pd

import thirdai_python_package.neural_db.parsing_utils.doc_parse as doc_parse

from ..core.documents import Document
from ..core.types import NewChunkBatch
from .utils import series_from_value


class DOCX(Document):
    def __init__(self, path, metadata=None):
        super().__init__()

        self.path = path
        self.metadata = metadata

    def chunks(self) -> Iterable[NewChunkBatch]:
        elements, success = doc_parse.get_elements(self.path)

        if not success:
            raise ValueError(f"Unable to parse docx file: '{self.path}'.")

        parsed_chunks = doc_parse.create_train_df(elements)

        text = parsed_chunks["para"]

        metadata = None
        if self.metadata:
            metadata = pd.DataFrame.from_records([self.metadata] * len(text))

        return [
            NewChunkBatch(
                custom_id=None,
                text=text,
                keywords=series_from_value("", len(text)),
                metadata=metadata,
                document=series_from_value(self.path, len(text)),
            )
        ]
