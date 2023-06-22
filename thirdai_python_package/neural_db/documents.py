import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from requests.models import Response


import pandas as pd
from .parsing_utils import doc_parse, pdf_parse, url_parse
from .qa import ContextArgs
from .utils import hash_string, hash_file
from thirdai.dataset.data_source import PyDataSource


class Reference:
    def __init__(
        self, element_id: int, text: str, source: str, metadata: dict, show_fn
    ):
        self._element_id = element_id
        self._text = text
        self._source = source
        self._metadata = metadata
        self._show_fn = show_fn

    def element_id(self):
        return self._element_id

    def text(self):
        return self._text

    def source(self):
        return self._source

    def metadata(self):
        return self._metadata

    def show(self):
        return self._show_fn(
            text=self._text,
            source=self._source,
            **self._metadata,
        )


class Document:
    def hash(self) -> str:
        raise NotImplementedError()

    def size(self) -> int:
        raise NotImplementedError()

    def name(self) -> str:
        raise NotImplementedError()

    def strong_text(self, element_id: int) -> str:
        raise NotImplementedError()

    def weak_text(self, element_id: int) -> str:
        raise NotImplementedError()

    def reference(self, element_id: int) -> Reference:
        raise NotImplementedError()

    def context(self, element_id: int, context_args: ContextArgs) -> str:
        raise NotImplementedError()

    def save_meta(self, directory: Path):
        raise NotImplementedError()

    def load_meta(self, directory: Path):
        raise NotImplementedError()


class Reference:
    def __init__(
        self, 
        document: Document,
        element_id: int, 
        text: str, 
        source: str, 
        metadata: dict, 
        show_fn: Callable = lambda *args, **kwargs: None,
    ):
        self._id = element_id
        self._text = text
        self._source = source
        self._metadata = metadata
        self._show_fn = show_fn
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

    def show(self):
        return self._show_fn(
            text=self._text,
            source=self._source,
            **self._metadata,
        )


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

    def _next_id(self):
        if len(self.id_sorted_docs) == 0:
            return 0
        doc, start_id = self.id_sorted_docs[-1]
        return start_id + doc.size()

    def add(self, documents: List[Document]):
        intro_dds = DocumentDataSource(self.id_column, self.strong_column, self.weak_column)
        train_dds = DocumentDataSource(self.id_column, self.strong_column, self.weak_column)
        for doc in documents:
            doc_hash = doc.hash()
            if doc_hash not in self.registry:
                start_id = self._next_id()
                # Adding this tuple to two data structures does not double the
                # memory usage because Python uses references.
                doc_and_id = (doc, start_id)
                self.registry[doc_hash] = doc_and_id
                self.id_sorted_docs.append(doc_and_id)
                intro_dds.add(doc, start_id)
            doc, start_id = self.registry[doc_hash]
            train_dds.add(doc, start_id)
        return IntroAndTrainDocuments(intro=intro_dds, train=train_dds)
    
    def sources(self):
        return [doc.name() for doc, _ in self.id_sorted_docs]

    def clear(self):
        self.registry = {}
        self.id_sorted_docs = []

    def _get_doc_and_start_id(self, element_id: int):
        # check if element_id is valid
        if not self.id_sorted_docs or (not 0 <= element_id < self.id_sorted_docs[-1][1] + self.id_sorted_docs[-1][0].size()):
            raise ValueError(f"Unable to find element that has id {element_id}.")
        # Iterate through docs in reverse order
        for i in range(len(self.id_sorted_docs) - 1, -1, -1):
            doc, start_id = self.id_sorted_docs[i]
            if start_id <= element_id:
                return self.id_sorted_docs[i]

    def reference(self, element_id: int):
        doc, start_id = self._get_doc_and_start_id(element_id)
        doc_ref = doc.reference(element_id - start_id)
        doc_ref._element_id = element_id
        return doc_ref

    def context(self, element_id: int, context_args: ContextArgs):
        doc, start_id = self._get_doc_and_start_id(element_id)
        return doc.context(
            element_id - start_id, context_args
        )

    def save_meta(self, directory: Path):
        for i, (doc, _) in enumerate(self.id_sorted_docs):
            subdir = directory / str(i)
            os.mkdir(subdir)
            doc.save_meta(subdir)

    def load_meta(self, directory: Path):
        for i, (doc, _) in enumerate(self.id_sorted_docs):
            subdir = directory / str(i)
            doc.load_meta(subdir)


# Base class for PDF and DOCX classes because they share the same logic.
class Extracted(Document):
    def __init__(
        self,
        filename: str,
    ):
        
        self.filename = filename
        self.df = self.process_data(filename)
        self.hash_val = hash_file(filename)

    def process_data(
        self,
        filename: str,
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def hash(self) -> str:
        return self.hash_val

    def size(self) -> int:
        return len(self.df)

    def name(self) -> str:
        return self.filename

    def strong_text(self, element_id: int) -> str:
        return self.df["passage"].iloc[element_id]

    def weak_text(self, element_id: int) -> str:
        return self.df["para"].iloc[element_id]
    
    def show_fn(text, source, **kwargs):
        return text

    def reference(self, element_id: int) -> Reference:
        return Reference(
            element_id=element_id,
            text=self.df["display"].iloc[element_id],
            source=self.filename,
            metadata={"page": self.df["page"].iloc[element_id]},
            show_fn=self.show_fn
        )

    def get_context(self, element_id, context_args) -> str:
        if not 0 <= element_id < self.size():
            raise ("Element id not in document.")

        if hasattr(context_args, "chunk_radius"):
            chunk_radius = context_args.chunk_radius

            center_chunk_idx = element_id

            window_start = center_chunk_idx - chunk_radius // 2
            window_end = window_start + chunk_radius

            if window_start < 0:
                window_start = 0
                window_end = min(chunk_radius, self.size())
            elif window_end > self.size():
                window_end = self.size()
                window_start = max(0, self.size() - chunk_radius)

            window_chunks = self.df.iloc[window_start:window_end]["passage"].tolist()
            return "\n".join(window_chunks)

        return ""


class PDF(Extracted):
    def __init__(
        self,
        filename: str,
    ):
        super().__init__(
            filename=filename
        )

    def process_data(
        self,
        filename: str,
    ) -> pd.DataFrame:
        elements, success = pdf_parse.process_pdf_file(filename)

        if not success:
            print(f"Could not read PDF file {filename}")
            return pd.DataFrame()

        elements_df = pdf_parse.create_train_df(elements)

        return elements_df


class DOCX(Extracted):
    def __init__(
        self,
        filename: str,
    ):
        super().__init__(
            filename=filename
        )

    def process_data(
        self,
        filename: str,
    ) -> pd.DataFrame:
        elements, success = doc_parse.get_elements(filename)

        if not success:
            print(f"Could not read DOCX file {filename}")
            return pd.DataFrame()

        elements_df = doc_parse.create_train_df(elements)

        return elements_df


class URL(Document):
    def __init__(
        self,
        url: str,
        url_response: Response = None
    ):
        self.url = url
        self.df = self.process_data(url, url_response)
        self.hash_val = hash_string(url)

    def process_data(self, url, url_response=None) -> pd.DataFrame:
        # Extract elements from each file
        elements, success = url_parse.process_url(url, url_response)

        if not success or not elements:
            return pd.DataFrame()

        elements_df = url_parse.create_train_df(elements)

        return elements_df

    def hash(self) -> str:
        return self.hash_val

    def size(self) -> int:
        return len(self.df)

    def name(self) -> str:
        return self.url

    def strong_text(self, element_id: int) -> str:
        return self.df["text"].iloc[element_id]

    def weak_text(self, element_id: int) -> str:
        return self.df["text"].iloc[element_id]
    
    def show_fn(text, source, **kwargs):
        return text

    def reference(self, element_id: int) -> Reference:
        return Reference(
            element_id=element_id,
            text=self.df["display"].iloc[element_id],
            source=self.url,
            metadata={},
            show_fn=self.show_fn
        )

    def get_context(self, element_id, context_args) -> str:
        if not 0 <= element_id < self.size():
            raise ("Element id not in document.")

        if hasattr(context_args, "chunk_radius"):
            chunk_radius = context_args.chunk_radius

            center_chunk_idx = element_id

            window_start = center_chunk_idx - chunk_radius // 2
            window_end = window_start + chunk_radius

            if window_start < 0:
                window_start = 0
                window_end = min(chunk_radius, self.size())
            elif window_end > self.size():
                window_end = self.size()
                window_start = max(0, self.size() - chunk_radius)

            window_chunks = self.df.iloc[window_start:window_end]["text"].tolist()
            return "\n".join(window_chunks)

        return ""


class CSV(Document):
    def __init__(self, filename, strong_cols, weak_cols):
        self.filename = filename
        self.strong_cols = strong_cols
        self.weak_cols = weak_cols
        self.df = pd.read_csv(self.filename)
        self.hash_val = hash_file(filename)

        for col in strong_cols + weak_cols:
            self.df[col] = self.df[col].fillna("")

    def hash(self) -> str:
        return self.hash_val

    def size(self) -> int:
        return len(self.df)

    def name(self) -> str:
        return self.filename

    def strong_text(self, element_id: int) -> str:
        return " ".join(self.df[self.strong_cols].iloc[element_id].tolist())
       
    def weak_text(self, element_id: int) -> str:
        return " ".join(self.df[self.weak_cols].iloc[element_id].tolist())
    
    def show_fn(text, source, **kwargs):
        return text

    def reference(self, element_id: int) -> Reference:
        return Reference(
            element_id=element_id,
            text=self.weak_text(element_id),
            source=self.filename,
            metadata={},
            show_fn=self.show_fn
        )
