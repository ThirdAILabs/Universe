from nltk.tokenize import word_tokenize
from thirdai import search

from .documents import DocumentDataSource


class ChunkedRowIterator:
    def __init__(self, iterator):
        self.iterator = iterator

    def next(self, n):
        ids = []
        docs = []

        for row in self.iterator:
            ids.append(row.id)
            docs.append(word_tokenize(row.strong) + word_tokenize(row.weak))

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

    def query(self, queries: str, k: int):
        if len(self.indexes) == 0:
            raise ValueError("Cannot query before inserting documents.")

        if len(self.indexes) == 1:
            return self.indexes[0].query(
                queries=[word_tokenize(q) for q in queries], k=k
            )
        return [
            search.InvertedIndex.parallel_query(self.indexes, word_tokenize(q), k=k)
            for q in queries
        ]

    def forget(self, ids):
        for index in self.indexes:
            index.remove(ids)

    def clear(self):
        self.indexes = []
