from typing import Iterable, List

import pandas as pd

from thirdai_python_package.neural_db.parsing_utils.unstructured_parse import (
    EmlParse,
    PptxParse,
    TxtParse,
    UnstructuredParse,
)

from ..core.documents import Document
from ..core.types import NewChunkBatch
from .utils import series_from_value


class Unstructured(Document):
    def __init__(
        self,
        path: str,
        parser: UnstructuredParse,
        metadata_columns: List[str],
        metadata: dict = None,
    ):
        super().__init__()

        self.path = path
        self.parser = parser
        self.metadata_columns = metadata_columns
        self.metadata = metadata

    def chunks(self) -> Iterable[NewChunkBatch]:
        parser = self.parser(self.path)

        elements, success = parser.process_elements()

        if not success:
            raise ValueError(f"Could not read file: {self.path}")

        contents = parser.create_train_df(elements)

        text = contents["para"]

        metadata = contents[self.metadata_columns]
        if self.metadata:
            metadata = pd.concat(
                [metadata, pd.DataFrame.from_records([self.metadata] * len(text))],
                axis=1,
            )

        return [
            NewChunkBatch(
                custom_id=None,
                text=text,
                keywords=series_from_value("", len(text)),
                metadata=metadata,
                document=contents["filename"],
            )
        ]


class PPTX(Unstructured):
    def __init__(self, path, metadata):
        super().__init__(
            path=path,
            parser=PptxParse,
            metadata_columns=["filetype", "page"],
            metadata=metadata,
        )


class TextFile(Unstructured):
    def __init__(self, path, metadata):
        super().__init__(
            path=path,
            parser=TxtParse,
            metadata_columns=["filetype"],
            metadata=metadata,
        )


class Email(Unstructured):
    def __init__(self, path, metadata):
        super().__init__(
            path=path,
            parser=EmlParse,
            metadata_columns=["filetype", "subject", "sent_from", "sent_to"],
            metadata=metadata,
        )
