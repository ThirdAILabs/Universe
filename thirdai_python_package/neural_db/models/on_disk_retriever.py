from pathlib import Path
from typing import Callable, List, Optional, Tuple

from thirdai import search

from ..documents import DocumentDataSource
from ..supervised_datasource import SupDataSource
from .model_interface import InferSamples, Model, Predictions, add_retriever_tag


class OnDiskRetriever(Model):
    def __init__(self, retriever: Optional[search.OnDiskIndex] = None):
        self.retriever = retriever or search.OnDiskIndex(db_name="sample_db")

    def index_from_start(
        self,
        intro_documents: DocumentDataSource,
        on_progress: Callable = lambda *args, **kwargs: None,
        batch_size=100000,
        **kwargs
    ):
        docs = []
        ids = []

        for row in intro_documents.row_iterator():
            docs.append(row.strong + " " + row.weak)
            ids.append(row.id)

            if len(docs) == batch_size:
                self.retriever.index(ids=ids, docs=docs)
                docs = []
                ids = []

        if len(docs):
            self.retriever.index(ids=ids, docs=docs)

    def forget_documents(self) -> None:
        self.retriever = search.OnDiskRetriever()

    def delete_entities(self, entities) -> None:
        return

    @property
    def searchable(self) -> bool:
        return True

    def get_query_col(self) -> str:
        return "QUERY"

    def get_id_col(self) -> str:
        return "DOC_ID"

    def get_id_delimiter(self) -> str:
        return ":"

    def infer_labels(
        self, samples: InferSamples, n_results: int, **kwargs
    ) -> Predictions:
        results = self.retriever.query(queries=samples, k=n_results)
        return add_retriever_tag(results, "OnDisk_retriever")

    def score(
        self, samples: InferSamples, entities: List[List[int]], n_results: int = None
    ) -> Predictions:
        return None

    def save_meta(self, directory: Path) -> None:
        pass

    def load_meta(self, directory: Path):
        pass

    def associate(self, pairs: List[Tuple[str, str]], retriever_strength=4, **kwargs):
        return None

    def upvote(self, pairs: List[Tuple[str, int]], **kwargs):
        return None

    def train_on_supervised_data_source(
        self, supervised_data_source: SupDataSource, **kwargs
    ):
        return None
    
    def get_model(self):
        return None

    def retrain(self, **kwargs):
        pass
