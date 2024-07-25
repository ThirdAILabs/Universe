import uuid
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from thirdai import search

from ..documents import DocumentDataSource
from ..supervised_datasource import SupDataSource
from .model_interface import InferSamples, Model, Predictions, add_retriever_tag


class FinetunableRetriever(Model):
    def __init__(
        self, retriever: Optional[search.FinetunableRetriever] = None, on_disk=False
    ):
        save_path = None
        if on_disk:
            save_path = f"{uuid.uuid4()}.db"
        self.retriever = retriever or search.FinetunableRetriever(save_path=save_path)

    def index_from_start(
        self,
        intro_documents: DocumentDataSource,
        on_progress: Callable = lambda *args, **kwargs: None,
        batch_size=100000,
        **kwargs,
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

                on_progress(self.retriever.size() / intro_documents.size)

        if len(docs):
            self.retriever.index(ids=ids, docs=docs)
            on_progress(self.retriever.size() / intro_documents.size)

        self.retriever.prune()

    def forget_documents(self) -> None:
        self.retriever = search.FinetunableRetriever()

    def delete_entities(self, entities) -> None:
        self.retriever.remove(entities)

    @property
    def searchable(self) -> bool:
        return self.retriever.size() > 0

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
        return add_retriever_tag(results, "finetunable_retriever")

    def score(
        self, samples: InferSamples, entities: List[List[int]], n_results: int = None
    ) -> Predictions:

        # retriever.rank() expects candidates to be a list of sets
        candidates = [set(ids) for ids in entities]
        results = self.retriever.rank(
            queries=samples, candidates=candidates, k=n_results
        )
        return add_retriever_tag(results, "finetunable_retriever")

    def save_meta(self, directory: Path) -> None:
        self.retriever.save(str(directory))

    def load_meta(self, directory: Path, read_only: bool = False):
        if not self.retriever:
            self.retriever = search.FinetunableRetriever.load(str(directory), read_only)

    def __getstate__(self) -> object:
        return {"retriever": None}

    def associate(self, pairs: List[Tuple[str, str]], retriever_strength=4, **kwargs):
        sources, targets = list(zip(*pairs))
        self.retriever.associate(
            sources=sources, targets=targets, strength=retriever_strength
        )

    def upvote(self, pairs: List[Tuple[str, int]], **kwargs):
        queries, ids = list(zip(*pairs))
        ids = [[y] for y in ids]
        self.retriever.finetune(doc_ids=ids, queries=queries)

    def train_on_supervised_data_source(
        self, supervised_data_source: SupDataSource, **kwargs
    ):
        for sup in supervised_data_source.data:
            labels = [list(map(int, y)) for y in supervised_data_source._labels(sup)]

            self.retriever.finetune(doc_ids=labels, queries=list(sup.queries))

    def get_model(self):
        return None

    def retrain(self, **kwargs):
        pass
