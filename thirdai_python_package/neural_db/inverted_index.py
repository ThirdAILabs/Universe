from typing import List, Tuple

from thirdai import search

from .documents import DocumentDataSource
from .supervised_datasource import SupDataSource


class ChunkedRowIterator:
    def __init__(self, iterator):
        self.iterator = iterator

    def next(self, n):
        ids = []
        docs = []

        for row in self.iterator:
            ids.append(row.id)
            docs.append(row.strong + " " + row.weak)

            if len(ids) == n:
                return ids, docs

        if len(ids) == 0:
            return None

        return ids, docs


class InvertedIndex:
    def __init__(self, max_shard_size: int = 8_000_000):
        self.indexes = []
        self.max_shard_size = max_shard_size

    def insert(self, doc_data_source: DocumentDataSource):
        if len(self.indexes) > 0 and self.indexes[-1].size() < self.max_shard_size:
            curr_index = self.indexes[-1]
        else:
            curr_index = search.InvertedIndex()

        chunked_iterator = ChunkedRowIterator(doc_data_source.row_iterator())

        while chunk := chunked_iterator.next(self.max_shard_size - curr_index.size()):
            curr_index.index(ids=chunk[0], docs=chunk[1])
            if curr_index.size() == self.max_shard_size:
                self.indexes.append(curr_index)
                curr_index = search.InvertedIndex()

        if curr_index.size() > 0:
            self.indexes.append(curr_index)

    def supervised_train(self, data_source: SupDataSource):
        queries = []
        ids = []
        for sup in data_source.data:
            for query, labels in zip(sup.queries, data_source._labels(sup)):
                for label in labels:
                    queries.append(query)
                    ids.append(int(label))
        for index in self.indexes:
            index.update(ids, queries, ignore_missing_ids=len(self.indexes) > 1)

    def upvote(self, pairs: List[Tuple[str, int]]) -> None:
        ids = [x[1] for x in pairs]
        phrases = [x[0] for x in pairs]
        for index in self.indexes:
            index.update(ids, phrases, ignore_missing_ids=len(self.indexes) > 1)

    def associate(self, pairs: List[Tuple[str, str]]) -> None:
        sources = [x[0] for x in pairs]
        targets = [x[1] for x in pairs]

        for index in self.indexes:
            top_results = index.query(targets, k=3)

            update_texts = []
            update_ids = []
            for source, results in zip(sources, top_results):
                for result in results:
                    update_texts.append(source)
                    update_ids.append(result[0])

            index.update(
                update_ids, update_texts, ignore_missing_ids=len(self.indexes) > 1
            )

    def query(self, queries: str, k: int):
        if len(self.indexes) == 0:
            raise ValueError("Cannot query before inserting documents.")

        if len(self.indexes) == 1:
            return self.indexes[0].query(queries=[q for q in queries], k=k)
        return [
            search.InvertedIndex.parallel_query(self.indexes, q, k=k) for q in queries
        ]

    def forget(self, ids):
        for index in self.indexes:
            index.remove(ids)

    def clear(self):
        self.indexes = []

    def __setstate__(self, state):
        if "indexes" not in state:
            state["indexes"] = [state["index"]]
            state["max_shard_size"] = 8_000_000
            del state["index"]
        self.__dict__.update(state)
