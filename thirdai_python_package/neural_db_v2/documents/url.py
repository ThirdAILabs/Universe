from typing import Any, Dict, Iterable, Optional

import thirdai.neural_db.parsing_utils.url_parse as url_parse
from requests.models import Response

from ..core.documents import Document
from ..core.types import NewChunkBatch
from .utils import join_metadata


class URL(Document):
    def __init__(
        self,
        url: str,
        response: Response = None,
        title_is_strong: bool = False,
        doc_metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ):
        super().__init__(doc_id=doc_id)

        self.url = url
        self.response = response
        self.title_is_strong = title_is_strong
        self.doc_metadata = doc_metadata

    def chunks(self) -> Iterable[NewChunkBatch]:
        elements, success = url_parse.process_url(self.url, self.response)

        if not success or not elements:
            raise ValueError(f"Could not retrieve data from URL: {self.url}")

        content = url_parse.create_train_df(elements)

        text = content["text"]
        keywords = content["title"] if self.title_is_strong else content["text"]

        metadata = join_metadata(
            n_rows=len(text), chunk_metadata=None, doc_metadata=self.doc_metadata
        )
        return [
            NewChunkBatch(
                text=text,
                keywords=keywords,
                metadata=metadata,
                document=content["url"],
            )
        ]
