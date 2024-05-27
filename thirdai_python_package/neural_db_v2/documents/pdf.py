from typing import Iterable

import thirdai.neural_db.parsing_utils.sliding_pdf_parse as pdf_parse

from ..core.documents import Document
from ..core.types import NewChunkBatch
from .utils import join_metadata, series_from_value


class PDF(Document):
    def __init__(
        self,
        path,
        chunk_size=100,
        stride=40,
        emphasize_first_words=0,
        ignore_header_footer=True,
        ignore_nonstandard_orientation=True,
        doc_metadata=None,
        doc_keywords="",
        emphasize_section_titles=False,
        table_parsing=False,
    ):
        super().__init__()

        self.path = path
        self.chunk_size = chunk_size
        self.stride = stride
        self.emphasize_first_words = emphasize_first_words
        self.ignore_header_footer = ignore_header_footer
        self.ignore_nonstandard_orientation = ignore_nonstandard_orientation
        self.table_parsing = table_parsing
        self.doc_metadata = doc_metadata
        self.doc_keywords = doc_keywords
        self.emphasize_section_titles = emphasize_section_titles
        self.table_parsing = table_parsing

    def chunks(self) -> Iterable[NewChunkBatch]:
        parsed_chunks = pdf_parse.make_df(
            filename=self.path,
            chunk_words=self.chunk_size,
            stride_words=self.stride,
            emphasize_first_n_words=self.emphasize_first_words,
            ignore_header_footer=self.ignore_header_footer,
            ignore_nonstandard_orientation=self.ignore_nonstandard_orientation,
            doc_keywords=self.doc_keywords,
            emphasize_section_titles=self.emphasize_section_titles,
            table_parsing=self.table_parsing,
        )

        text = parsed_chunks["para"]
        keywords = parsed_chunks["emphasis"]

        metadata = join_metadata(
            n_rows=len(text),
            chunk_metadata=parsed_chunks[["chunk_boxes", "page"]],
            doc_metadata=self.doc_metadata,
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
