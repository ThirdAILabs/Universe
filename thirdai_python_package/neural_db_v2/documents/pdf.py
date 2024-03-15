from ..core.documents import Document
from ..core.types import NewChunkBatch
from typing import Iterable, List, Any
import pandas as pd
import thirdai_python_package.neural_db.parsing_utils.sliding_pdf_parse as pdf_parse
from .utils import series_from_value


class PDF(Document):
    def __init__(
        self,
        path,
        chunk_size=100,
        stride=40,
        emphasize_first_words=0,
        ignore_header_footer=True,
        ignore_nonstandard_orientation=True,
        metadata=None,
    ):
        self.path = path
        self.chunk_size = chunk_size
        self.stride = stride
        self.emphasize_first_words = emphasize_first_words
        self.ignore_header_footer = ignore_header_footer
        self.ignore_nonstandard_orientation = ignore_nonstandard_orientation
        self.metadata = metadata

    def chunks(self) -> Iterable[NewChunkBatch]:
        parsed_chunks = pdf_parse.make_df(
            filename=self.path,
            chunk_words=self.chunk_size,
            stride_words=self.stride,
            emphasize_first_n_words=self.emphasize_first_words,
            ignore_header_footer=self.ignore_header_footer,
            ignore_nonstandard_orientation=self.ignore_nonstandard_orientation,
        )

        text = parsed_chunks["para"]
        keywords = parsed_chunks["emphasis"]

        metadata = parsed_chunks[["chunk_boxes", "page"]]
        if self.metadata:
            metadata = pd.concat(
                [metadata, pd.DataFrame.from_records([self.metadata] * len(text))],
                axis=1,
            )

        return [
            NewChunkBatch(
                custom_id=None,
                text=text,
                keywords=keywords,
                metadata=metadata,
                document=series_from_value(self.path, len(text)),
            )
        ]
