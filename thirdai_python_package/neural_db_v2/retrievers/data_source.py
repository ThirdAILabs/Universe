from typing import List

from thirdai.dataset.data_source import PyDataSource
from core.types import Document


class DocumentDataSource(PyDataSource):
    def __init__(self, docs: List[Document], doc_id, strong, weak, query):
        PyDataSource.__init__(self)
        self.docs = docs
        self.doc_id = doc_id
        self.strong = strong
        self.weak = weak
        self.query = query
        self._size = 0
        self.restart()

    @property
    def size(self):
        return self._size

    def _csv_line(self, element_id: str, strong: str, weak: str):
        csv_strong = '"' + strong.replace('"', '""') + '"'
        csv_weak = '"' + weak.replace('"', '""') + '"'
        return f"{element_id},{csv_strong},{csv_weak}"

    def _get_line_iterator(self):
        # First yield the header
        yield f"{self.doc_id},{self.strong},{self.weak}"
        # Then yield rows
        for doc in self.docs:
            yield self._csv_line(
                element_id=doc.doc_id, strong=doc.keywords, weak=doc.text
            )

    def resource_name(self) -> str:
        return "NeuralDB Documents"
