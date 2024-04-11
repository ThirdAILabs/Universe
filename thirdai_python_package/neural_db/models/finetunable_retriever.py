from pathlib import Path
from typing import List, Optional, Tuple

from thirdai import search

from ..documents import DocumentDataSource
from ..supervised_datasource import SupDataSource
from .models import InferSamples, Model, Predictions, add_retriever_tag


class FinetunableRetriever(Model):
    def __init__(self):
        self.retriever = search.FinetunableRetriever()

    def index_from_start(
        self, intro_documents: DocumentDataSource, batch_size=100000, **kwargs
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
        results = self.retriever.rank(queries=samples, candidates=entities, k=n_results)
        return add_retriever_tag(results)

    def save_meta(self, directory: Path) -> None:
        pass

    def load_meta(self, directory: Path):
        pass

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
