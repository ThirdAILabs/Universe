from typing import List, Tuple, Dict, Optional
from thirdai.dataset import PyDataSource
import pandas as pd


class Reference:
    def __init__(self, id: int, text: str, source: str, page: Optional[int]):
        self._id = id
        self._text = text
        self._source = source
        self._page = page

    def id(self):
        return self._id

    def text(self):
        return self._text

    def source(self):
        return self._source

    def page(self):
        return self._page


class Document:
    def hash(self) -> str:
        pass
    
    def size(self) -> int:
        pass
    
    def name(self) -> str:
        pass

    def strong_text(self, id: int) -> str:
        pass

    def weak_text(self, id: int) -> str:
        pass
    
    def reference(self, id: int) -> Reference:
        pass

    def context(self, id: int, radius) -> str:
        pass


class DocumentDataSource(PyDataSource):
    def __init__(self, id_column, strong_column, weak_column):
        PyDataSource.__init__(self)
        self.documents: List[Tuple[Document, int]] = []
        self.id_column = id_column
        self.strong_column = strong_column
        self.weak_column = weak_column
    
    def add(self, document: Document, start_id: int):
        self.documents.append((document, start_id))

    def _csv_line(self, id: str, strong: str, weak: str):
        df = pd.DataFrame({
            self.id_column: [id],
            self.strong_column: [strong],
            self.weak_column: [weak],
        })
        return df.to_csv(header=None, index=None).strip("\n")

    def _get_line_iterator(self):
        # First yield the header
        yield self._csv_line(self.id_column, self.strong_column, self.weak_column)
        # Then yield rows
        for doc, start_id in self.documents:
            for i in range(doc.size()):
                yield self._csv_line(
                    id=start_id + i,
                    strong=doc.strong_text(i),
                    weak=doc.weak_text(i))
    
    def resource_name(self) -> str:
        return "Documents:\n" + "\n".join([doc.name() for doc in self.documents])


class IntroAndTrainDocuments:
    def __init__(self, intro: DocumentDataSource, train: DocumentDataSource) -> None:
        self.intro = intro
        self.train = train


class DocumentManager:
    def __init__(self, id_column, strong_column, weak_column) -> None:
        self.id_column = id_column
        self.strong_column = strong_column
        self.weak_column = weak_column
        self.registry : Dict[str, Tuple[Document, int]] = {}
        self.id_sorted_docs : List[Tuple[Document, int]] = []
    
    def _next_id(self):
        if len(self.id_sorted_docs) == 0:
            return 0
        doc, start_id = self.id_sorted_docs[-1]
        return start_id + doc.size()

    def add(self, documents: List[Document]):
        intro = DocumentDataSource(self.id_column, self.strong_column, self.weak_column)
        train = DocumentDataSource(self.id_column, self.strong_column, self.weak_column)
        for doc in documents:
            doc_hash = doc.hash()
            if doc_hash not in self.registry:
                start_id = self._next_id()
                # Adding this tuple to two data structures does not double the
                # memory usage because Python uses references.
                doc_and_id = (doc, start_id)
                self.registry[doc_hash] = doc_and_id
                self.id_sorted_docs.append(doc_and_id)
                intro.add(doc, start_id)
            doc, start_id = self.registry[doc_hash]
            train.add(doc, start_id)

        return IntroAndTrainDocuments(intro=intro, train=train)
    
    def clear(self):
        self.registry = {}
        self.id_sorted_docs = []

    def _get_doc_and_start_id(self, id: int):
        # Iterate through docs in reverse order
        for i in range(len(self.id_sorted_docs) - 1, -1, -1):
            return self.id_sorted_docs[i]
        raise ValueError(f"Unable to find document that has id {id}.")

    def reference(self, id: int):
        doc, start_id = self._get_doc_and_start_id(id)
        return doc.reference(id - start_id)

    def context(self, id: int, radius: int):
        doc, start_id = self._get_doc_and_start_id(id)
        return doc.context(id - start_id, radius)