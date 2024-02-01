from .documents import DocumentDataSource
from thirdai import search
from nltk.tokenize import word_tokenize


class InvertedIndex:
    def __init__(self):
        self.index = search.InvertedIndex()

    def insert(self, docs: DocumentDataSource):
        ids = []
        docs = []
        for row in docs.row_iterator():
            ids.append(row.id)
            docs.append(word_tokenize(row.strong + row.weak))

        self.index.index(ids=ids, docs=docs)

    def query(self, queries: str, k: int):
        return self.index.query(queries=[word_tokenize(q) for q in queries], k=k)

    def forget(self, ids):
        self.index.remove(ids)

    def clear(self):
        self.index = search.InvertedIndex()
