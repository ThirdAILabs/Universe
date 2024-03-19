from typing import Iterable

import pandas as pd
from requests.models import Response

import thirdai_python_package.neural_db.parsing_utils.url_parse as url_parse

from ..core.documents import Document
from ..core.types import NewChunkBatch


class URL(Document):
    def __init__(
        self,
        url: str,
        response: Response = None,
        title_is_strong: bool = False,
        metadata=None,
    ):
        super().__init__()

        self.url = url
        self.response = response
        self.title_is_strong = title_is_strong
        self.metadata = metadata

    def chunks(self) -> Iterable[NewChunkBatch]:
        elements, success = url_parse.process_url(self.url, self.response)

        if not success or not elements:
            raise ValueError(f"Could not retrieve data from URL: {self.url}")

        content = url_parse.create_train_df(elements)

        text = content["text"]
        keywords = content["title"] if self.title_is_strong else content["text"]

        metadata = None
        if self.metadata is not None:
            metadata = pd.DataFrame.from_records([self.metadata] * len(text))

        return [
            NewChunkBatch(
                custom_id=None,
                text=text,
                keywords=keywords,
                metadata=metadata,
                document=content["url"],
            )
        ]
