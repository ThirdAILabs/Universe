from typing import List, Tuple

from nltk.tokenize import word_tokenize
from thirdai import search

from .documents import DocumentDataSource


class InvertedIndex:
    def __init__(self):
        self.index = search.InvertedIndex()

    def insert(self, doc_data_source: DocumentDataSource) -> None:
        ids = []
        docs = []
        for row in doc_data_source.row_iterator():
            ids.append(row.id)
            docs.append(word_tokenize(row.strong) + word_tokenize(row.weak))

        self.index.index(ids=ids, docs=docs)

    def query(self, queries: str, k: int) -> List[List[Tuple[int, float]]]:
        return self.index.query(queries=[word_tokenize(q) for q in queries], k=k)

    def upvote(self, pairs: List[Tuple[str, int]]) -> None:
        self.index.update([x[1] for x in pairs], [word_tokenize(x[0]) for x in pairs])

    def associate(self, pairs: List[Tuple[str, str]]) -> None:
        sources = [word_tokenize(x[0]) for x in pairs]
        targets = [word_tokenize(x[1]) for x in pairs]

        top_results = self.index.query(targets, k=3)

        update_texts = []
        update_ids = []
        for source, results in zip(sources, top_results):
            for result in results:
                update_texts.append(source)
                update_ids.append(result[0])

        self.index.update(update_ids, update_texts)

    def forget(self, ids: List[int]) -> None:
        self.index.remove(ids)

    def clear(self) -> None:
        self.index = search.InvertedIndex()
