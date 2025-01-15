import logging
from typing import Any, Dict, Iterable, Optional

from thirdai.neural_db.documents import process_pdf as pdf_parse_v1
from thirdai.neural_db.parsing_utils.pdf_parse import (
    highlighted_doc as highlighted_doc_v1,
)
from thirdai.neural_db.parsing_utils.sliding_pdf_parse import (
    highlighted_doc as highlighted_doc_v2,
)
from thirdai.neural_db.parsing_utils.sliding_pdf_parse import make_df as pdf_parse_v2

from ..core.documents import Document
from ..core.types import Chunk, NewChunkBatch
from .utils import join_metadata, series_from_value


class PDF(Document):
    def __init__(
        self,
        path: str,
        version: str = "v1",
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
        display_path: Optional[str] = None,
        with_images: bool = False,
        parallelize: bool = False,
    ):
        super().__init__(doc_id=doc_id, doc_metadata=doc_metadata)

        if version not in ["v1", "v2"]:
            raise ValueError("Invalid version, must be either 'v1' or 'v2'.")

        self.version = version
        self.path = path
        self.chunk_size = chunk_size
        self.stride = stride
        self.emphasize_first_words = emphasize_first_words
        self.ignore_header_footer = ignore_header_footer
        self.ignore_nonstandard_orientation = ignore_nonstandard_orientation
        self.table_parsing = table_parsing
        self.doc_keywords = doc_keywords
        self.emphasize_section_titles = emphasize_section_titles
        self.table_parsing = table_parsing
        self.display_path = display_path
        self.with_images = with_images
        self.parallelize = parallelize

    def chunks(self) -> Iterable[NewChunkBatch]:
        if self.version == "v1":
            parsed_chunks = pdf_parse_v1(self.path, self.with_images, self.parallelize)
        else:
            parsed_chunks = pdf_parse_v2(
                filepath=self.path,
                chunk_words=self.chunk_size,
                stride_words=self.stride,
                emphasize_first_n_words=self.emphasize_first_words,
                ignore_header_footer=self.ignore_header_footer,
                ignore_nonstandard_orientation=self.ignore_nonstandard_orientation,
                doc_keywords=self.doc_keywords,
                emphasize_section_titles=self.emphasize_section_titles,
                table_parsing=self.table_parsing,
                with_images=self.with_images,
                parallelize=self.parallelize,
            )

        text = parsed_chunks["para"]

        if len(text) == 0:
            logging.warning(f"Unable to parse content from pdf {self.path}.")
            return []

        keywords = (
            parsed_chunks["emphasis"]
            if self.version == "v2"
            else series_from_value(self.doc_keywords, len(text))
        )

        metadata_columns = (
            ["chunk_boxes", "page"] if self.version == "v2" else ["highlight", "page"]
        )

        metadata = join_metadata(
            n_rows=len(text),
            chunk_metadata=parsed_chunks[metadata_columns],
            doc_metadata=self.doc_metadata,
        )

        return [
            NewChunkBatch(
                text=text,
                keywords=keywords,
                metadata=metadata,
                document=series_from_value(self.display_path or self.path, len(text)),
            )
        ]

    @staticmethod
    def highlighted_doc(
        source: str, chunk: Chunk, with_images: bool = False, parallelize: bool = False
    ):
        v1_highlighted = highlighted_doc_v1(
            source, chunk.metadata, with_images, parallelize
        )
        if v1_highlighted:
            return v1_highlighted
        return highlighted_doc_v2(source, chunk.metadata)
