from typing import Set, Iterable, List
from pathlib import Path

import pandas as pd
from core.types import DocId, Document
from core.index import Index
from utils.checkpointing import prepare_checkpoint_location


class DataFrameIndex(Index):
    def __init__(self):
        self.text_df = pd.DataFrame({"doc_id": [], "text": [], "keywords": []})
        self.text_df = self.text_df.set_index("doc_id")
        self.metadata_df = pd.DataFrame({"doc_id": [], "key": [], "value": []})
        self.metadata_df = self.metadata_df.set_index("doc_id")

    def insert_batch(
        self,
        docs: Iterable[Document],
        assign_new_unique_ids: bool = True,
        checkpoint: Path = None,
        **kwargs,
    ):
        if self.load_insertion_checkpoint(checkpoint):
            return

        def _doc_id(i: int, doc: Document):
            return self.text_df.index.max() + i if assign_new_unique_ids else doc.doc_id

        text_df_delta = pd.DataFrame.from_records(
            [
                {"doc_id": _doc_id(i, doc), "text": doc.text, "keywords": doc.keywords}
                for i, doc in enumerate(docs)
            ]
        ).set_index("doc_id")

        metadata_df_delta = pd.DataFrame.from_records(
            [
                {"doc_id": _doc_id(i, doc), "key": key, "value": value}
                for i, doc in enumerate(docs)
                for key, value in doc.metadata.items()
            ]
        ).set_index("doc_id")

        self.text_df = pd.concat([self.text_df, text_df_delta])
        self.metadata_df = pd.concat([self.metadata_df, metadata_df_delta])

        self.save_insertion_checkpoint(checkpoint)

    def delete(self, doc_id: DocId, **kwargs):
        self.text_df.drop([doc_id])
        self.metadata_df.drop([doc_id])

    def delete_batch(self, doc_ids: List[DocId], **kwargs):
        self.text_df.drop(doc_ids)
        self.metadata_df.drop(doc_ids)

    def get_doc(self, doc_id: DocId, **kwargs):
        text = self.text_df.loc[doc_id]
        metadata = self.metadata_df.loc[doc_id : doc_id + 1]
        return Document(
            doc_id=doc_id,
            text=text["text"],
            keywords=text["keywords"],
            metadata={
                key: value for key, value in zip(metadata["key"], metadata["value"])
            },
        )

    def get_doc_batch(self, doc_ids: List[DocId], **kwargs):
        # This is a very inefficient implementation. This is POC code.
        return [self.get_doc(doc_id) for doc_id in doc_ids]

    def matching_doc_ids(self, constraints: dict, **kwargs) -> Set[DocId]:
        pass
