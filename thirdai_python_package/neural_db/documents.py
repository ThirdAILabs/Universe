import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd
from pytrie import StringTrie
from thirdai.dataset.data_source import PyDataSource

from .utils import hash_string


class Reference:
    pass


class Document:
    def size(self) -> int:
        raise NotImplementedError()

    def name(self) -> str:
        raise NotImplementedError()

    def reference(self, element_id: int) -> Reference:
        raise NotImplementedError()

    def hash(self) -> str:
        sha1 = hashlib.sha1()
        sha1.update(bytes(self.name(), "utf-8"))
        for i in range(self.size()):
            sha1.update(bytes(self.reference(i).text(), "utf-8"))
        return sha1.hexdigest()

    def strong_text(self, element_id: int) -> str:
        return self.reference(element_id).text()

    def weak_text(self, element_id: int) -> str:
        return self.reference(element_id).text()

    def context(self, element_id: int, radius: int) -> str:
        window_start = max(0, element_id - radius)
        window_end = min(self.size(), element_id + radius + 1)
        return " \n".join(
            [self.reference(elid).text() for elid in range(window_start, window_end)]
        )

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass


class Reference:
    def __init__(
        self,
        document: Document,
        element_id: int,
        text: str,
        source: str,
        metadata: dict,
    ):
        self._id = element_id
        self._text = text
        self._source = source
        self._metadata = metadata
        self._context_fn = lambda radius: document.context(element_id, radius)

    def id(self):
        return self._id

    def text(self):
        return self._text

    def context(self, radius: int):
        return self._context_fn(radius)

    def source(self):
        return self._source

    def metadata(self):
        return self._metadata


class DocumentRow:
    def __init__(self, element_id: int, strong: str, weak: str):
        self.id = element_id
        self.strong = strong
        self.weak = weak


class DocumentDataSource(PyDataSource):
    def __init__(self, id_column, strong_column, weak_column):
        PyDataSource.__init__(self)
        self.documents: List[Tuple[Document, int]] = []
        self.id_column = id_column
        self.strong_column = strong_column
        self.weak_column = weak_column
        self._size = 0
        self.restart()

    def add(self, document: Document, start_id: int):
        self.documents.append((document, start_id))
        self._size += document.size()

    def row_iterator(self):
        for doc, start_id in self.documents:
            for i in range(doc.size()):
                yield DocumentRow(
                    element_id=start_id + i,
                    strong=doc.strong_text(i),
                    weak=doc.weak_text(i),
                )

    def size(self):
        return self._size

    def _csv_line(self, element_id: str, strong: str, weak: str):
        df = pd.DataFrame(
            {
                self.id_column: [element_id],
                self.strong_column: [strong],
                self.weak_column: [weak],
            }
        )
        return df.to_csv(header=None, index=None).strip("\n")

    def _get_line_iterator(self):
        # First yield the header
        yield self._csv_line(self.id_column, self.strong_column, self.weak_column)
        # Then yield rows
        for row in self.row_iterator():
            yield self._csv_line(element_id=row.id, strong=row.strong, weak=row.weak)

    def resource_name(self) -> str:
        return "Documents:\n" + "\n".join([doc.name() for doc, _ in self.documents])


class IntroAndTrainDocuments:
    def __init__(self, intro: DocumentDataSource, train: DocumentDataSource) -> None:
        self.intro = intro
        self.train = train


class DocumentManager:
    def __init__(self, id_column, strong_column, weak_column) -> None:
        self.id_column = id_column
        self.strong_column = strong_column
        self.weak_column = weak_column
        self.registry: Dict[str, Tuple[Document, int]] = {}
        self.id_sorted_docs: List[Tuple[Document, int]] = []
        self.source_id_prefix_trie = StringTrie()

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
                self.source_id_prefix_trie[doc_hash] = doc_hash
                intro.add(doc, start_id)
            doc, start_id = self.registry[doc_hash]
            train.add(doc, start_id)

        return IntroAndTrainDocuments(intro=intro, train=train), [
            doc.hash() for doc in documents
        ]

    def sources(self):
        return {doc_hash: doc.name() for doc_hash, (doc, _) in self.registry.items()}

    def match_source_id_by_prefix(self, prefix: str) -> Document:
        if prefix in self.registry:
            return [prefix]
        return self.source_id_prefix_trie.values(prefix)

    def source_by_id(self, source_id: str):
        return self.registry[source_id]

    def clear(self):
        self.registry = {}
        self.id_sorted_docs = []
        self.source_id_prefix_trie = StringTrie()

    def _get_doc_and_start_id(self, element_id: int):
        # Iterate through docs in reverse order
        for i in range(len(self.id_sorted_docs) - 1, -1, -1):
            doc, start_id = self.id_sorted_docs[i]
            if start_id <= element_id:
                return self.id_sorted_docs[i]
        raise ValueError(f"Unable to find document that has id {id}.")

    def reference(self, element_id: int):
        doc, start_id = self._get_doc_and_start_id(element_id)
        doc_ref = doc.reference(element_id - start_id)
        doc_ref._id = element_id
        return doc_ref

    def context(self, element_id: int, radius: int):
        doc, start_id = self._get_doc_and_start_id(element_id)
        return doc.context(element_id - start_id, radius)

    def save_meta(self, directory: Path):
        for i, (doc, _) in enumerate(self.id_sorted_docs):
            subdir = directory / str(i)
            os.mkdir(subdir)
            doc.save_meta(subdir)

    def load_meta(self, directory: Path):
        for i, (doc, _) in enumerate(self.id_sorted_docs):
            subdir = directory / str(i)
            doc.load_meta(subdir)
