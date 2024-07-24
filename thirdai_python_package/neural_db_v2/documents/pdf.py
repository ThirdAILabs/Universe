from typing import Any, Dict, Iterable, Optional

import thirdai.neural_db.parsing_utils.sliding_pdf_parse as pdf_parse

from ..core.documents import Document
from ..core.types import NewChunkBatch
from .utils import join_metadata, series_from_value


class PDF(Document):
    def __init__(
        self,
        path: str,
        chunk_size: int = 100,
        stride: int = 40,
        emphasize_first_words: int = 0,
        ignore_header_footer: bool = True,
        ignore_nonstandard_orientation: bool = True,
        doc_metadata: Optional[Dict[str, Any]] = None,
        doc_keywords: str = "",
        emphasize_section_titles: bool = False,
        table_parsing: bool = False,
        doc_id: Optional[str] = None,
    ):
        super().__init__(doc_id=doc_id)

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
                text=text,
                keywords=keywords,
                metadata=metadata,
                document=series_from_value(self.path, len(text)),
            )
        ]
