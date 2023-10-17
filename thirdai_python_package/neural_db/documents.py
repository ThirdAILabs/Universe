import hashlib
import os
import pickle
import shutil
import string
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from pytrie import StringTrie
from requests.models import Response
from sqlalchemy import Integer, String
from sqlalchemy.engine.base import Connection as sqlConn
from thirdai import bolt
from thirdai.data import get_udt_col_types
from thirdai.dataset.data_source import PyDataSource

from .connectors import SQLConnector
from .constraint_matcher import ConstraintMatcher, ConstraintValue, Filter, to_filters
from .parsing_utils import doc_parse, pdf_parse, url_parse
from .parsing_utils.unstructured_parse import EmlParse, PptxParse, TxtParse
from .utils import hash_file, hash_string

ConnectorType = Type[Union[SQLConnector, None]]


class Reference:
    pass


def _raise_unknown_doc_error(element_id: int):
    raise ValueError(f"Unable to find document that has id {element_id}.")


def raise_attribute_error(text: str):
    raise AttributeError(text)


class Document:
    @property
    def size(self) -> int:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def hash(self) -> str:
        sha1 = hashlib.sha1()
        sha1.update(bytes(self.name, "utf-8"))
        for i in range(self.size):
            sha1.update(bytes(self.reference(i).text, "utf-8"))
        return sha1.hexdigest()

    @property
    def matched_constraints(self) -> Dict[str, ConstraintValue]:
        raise NotImplementedError()

    def all_entity_ids(self) -> List[int]:
        raise NotImplementedError()

    # This attribute allows certain things to be saved or not saved during
    # the pickling of a savable_state object. For example, if we set this
    # to True for CSV docs, we will save the actual csv file in the pickle.
    # Utilize this property in save_meta and load_meta of document objs.
    @property
    def save_extra_info(self) -> bool:
        return self._save_extra_info

    @save_extra_info.setter
    def save_extra_info(self, value: bool):
        self._save_extra_info = value

    def reference(self, element_id: int) -> Reference:
        raise NotImplementedError()

    def strong_text(self, element_id: int, chunk: pd.DataFrame = None) -> str:
        return self.reference(element_id).text

    def weak_text(self, element_id: int, chun: pd.DataFrame = None) -> str:
        return self.reference(element_id).text

    def context(self, element_id: int, radius: int) -> str:
        window_start = max(0, element_id - radius)
        window_end = min(self.size, element_id + radius + 1)
        return " \n".join(
            [self.reference(elid).text for elid in range(window_start, window_end)]
        )

    def save_meta(self, directory: Path):
        pass

    def load_meta(self, directory: Path):
        pass

    def row_iterator(self, start_id: int):
        for i in range(self.size):
            yield DocumentRow(
                element_id=start_id + i,
                strong=self.strong_text(i),
                weak=self.weak_text(i),
            )

    def save(self, directory: str):
        dirpath = Path(directory)
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)
        os.mkdir(dirpath)
        with open(dirpath / f"doc.pkl", "wb") as pkl:
            pickle.dump(self, pkl)
        os.mkdir(dirpath / "meta")
        self.save_meta(dirpath / "meta")

    @staticmethod
    def load(directory: str):
        dirpath = Path(directory)
        with open(dirpath / f"doc.pkl", "rb") as pkl:
            obj = pickle.load(pkl)
        obj.load_meta(dirpath / "meta")
        return obj


class Reference:
    def __init__(
        self,
        document: Document,
        element_id: int,
        text: str,
        source: str,
        metadata: dict,
        upvote_ids: List[int] = None,
    ):
        self._id = element_id
        self._upvote_ids = upvote_ids if upvote_ids is not None else [element_id]
        self._text = text
        self._source = source
        self._metadata = metadata
        self._context_fn = lambda radius: document.context(element_id, radius)
        self._score = 0

    @property
    def id(self):
        return self._id

    @property
    def upvote_ids(self):
        return self._upvote_ids

    @property
    def text(self):
        return self._text

    @property
    def source(self):
        return self._source

    @property
    def metadata(self):
        return self._metadata

    @property
    def score(self):
        return self._score

    def context(self, radius: int):
        return self._context_fn(radius)


class DocumentRow:
    def __init__(self, element_id: int, strong: str, weak: str):
        self.id = element_id
        self.strong = strong
        self.weak = weak


DocAndOffset = Tuple[Document, int]


class DocumentDataSource(PyDataSource):
    def __init__(self, id_column, strong_column, weak_column):
        PyDataSource.__init__(self)
        self.documents: List[DocAndOffset] = []
        self.id_column = id_column
        self.strong_column = strong_column
        self.weak_column = weak_column
        self._size = 0
        self.restart()

    def add(self, document: Document, start_id: int):
        self.documents.append((document, start_id))
        self._size += document.size

    def row_iterator(self):
        for doc, start_id in self.documents:
            for row1 in doc.row_iterator(start_id):
                yield row1

    @property
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
        return "Documents:\n" + "\n".join([doc.name for doc, _ in self.documents])


class IntroAndTrainDocuments:
    def __init__(self, intro: DocumentDataSource, train: DocumentDataSource) -> None:
        self.intro = intro
        self.train = train


class DocumentManager:
    def __init__(self, id_column, strong_column, weak_column) -> None:
        self.id_column = id_column
        self.strong_column = strong_column
        self.weak_column = weak_column

        # After python 3.8, we don't need to use OrderedDict as Dict is ordered by default
        self.registry: OrderedDict[str, DocAndOffset] = OrderedDict()
        self.source_id_prefix_trie = StringTrie()
        self.constraint_matcher = ConstraintMatcher[DocAndOffset]()

    def _next_id(self):
        if len(self.registry) == 0:
            return 0
        doc, start_id = next(reversed(self.registry.values()))
        return start_id + doc.size

    def add(self, documents: List[Document]):
        intro = DocumentDataSource(self.id_column, self.strong_column, self.weak_column)
        train = DocumentDataSource(self.id_column, self.strong_column, self.weak_column)
        for doc in documents:
            doc_hash = doc.hash
            if doc_hash not in self.registry:
                start_id = self._next_id()
                doc_and_id = (doc, start_id)
                self.registry[doc_hash] = doc_and_id
                self.source_id_prefix_trie[doc_hash] = doc_hash
                intro.add(doc, start_id)
                self.constraint_matcher.index(
                    item=(doc, start_id), constraints=doc.matched_constraints
                )
            doc, start_id = self.registry[doc_hash]
            train.add(doc, start_id)

        return IntroAndTrainDocuments(intro=intro, train=train), [
            doc.hash for doc in documents
        ]

    def entity_ids_by_constraints(self, constraints: Dict[str, Any]):
        filters = to_filters(constraints)
        return [
            start_id + entity_id
            for doc, start_id in self.constraint_matcher.match(filters)
            for entity_id in doc.all_entity_ids()
        ]

    def sources(self):
        return {doc_hash: doc for doc_hash, (doc, _) in self.registry.items()}

    def match_source_id_by_prefix(self, prefix: str) -> Document:
        if prefix in self.registry:
            return [prefix]
        return self.source_id_prefix_trie.values(prefix)

    def source_by_id(self, source_id: str):
        return self.registry[source_id]

    def clear(self):
        self.registry = OrderedDict()
        self.source_id_prefix_trie = StringTrie()

    def _get_doc_and_start_id(self, element_id: int):
        for doc, start_id in reversed(self.registry.values()):
            if start_id <= element_id:
                return doc, start_id

        _raise_unknown_doc_error(element_id)

    def reference(self, element_id: int):
        doc, start_id = self._get_doc_and_start_id(element_id)
        doc_ref = doc.reference(element_id - start_id)
        doc_ref._id = element_id
        doc_ref._upvote_ids = [start_id + uid for uid in doc_ref._upvote_ids]
        return doc_ref

    def context(self, element_id: int, radius: int):
        doc, start_id = self._get_doc_and_start_id(element_id)
        return doc.context(element_id - start_id, radius)

    def get_data_source(self) -> DocumentDataSource:
        data_source = DocumentDataSource(
            id_column=self.id_column,
            strong_column=self.strong_column,
            weak_column=self.weak_column,
        )

        for doc, start_id in self.registry.values():
            data_source.add(document=doc, start_id=start_id)

        return data_source

    def save_meta(self, directory: Path):
        for i, (doc, _) in enumerate(self.registry.values()):
            subdir = directory / str(i)
            os.mkdir(subdir)
            doc.save_meta(subdir)

    def load_meta(self, directory: Path):
        for i, (doc, _) in enumerate(self.registry.values()):
            subdir = directory / str(i)
            doc.load_meta(subdir)

        if not hasattr(self, "doc_constraints"):
            self.constraint_matcher = ConstraintMatcher[DocAndOffset]()
            for item in self.registry.values():
                self.constraint_matcher.index(item, item[0].matched_constraints)


class CSV(Document):
    def __init__(
        self,
        path: str,
        id_column: Optional[str] = None,
        strong_columns: Optional[List[str]] = None,
        weak_columns: Optional[List[str]] = None,
        reference_columns: Optional[List[str]] = None,
        save_extra_info=True,
        metadata={},
    ) -> None:
        self.df = pd.read_csv(path)

        if reference_columns is None:
            reference_columns = list(self.df.columns)

        if id_column is None:
            id_column = "thirdai_index"
            self.df[id_column] = range(self.df.shape[0])

        if strong_columns is None and weak_columns is None:
            # autotune column types
            text_col_names = []
            try:
                for col_name, udt_col_type in get_udt_col_types(path).items():
                    if type(udt_col_type) == type(bolt.types.text()):
                        text_col_names.append(col_name)
            except:
                text_col_names = list(self.df.columns)
                text_col_names.remove(id_column)
                self.df[text_col_names] = self.df[text_col_names].astype(str)
            strong_columns = []
            weak_columns = text_col_names
        elif strong_columns is None:
            strong_columns = []
        elif weak_columns is None:
            weak_columns = []

        self.df = self.df.sort_values(id_column)
        assert len(self.df[id_column].unique()) == len(self.df[id_column])
        assert self.df[id_column].min() == 0
        assert self.df[id_column].max() == len(self.df[id_column]) - 1

        for col in strong_columns + weak_columns:
            self.df[col] = self.df[col].fillna("")

        self.path = Path(path)
        self._hash = hash_file(path, metadata="csv-" + str(metadata))
        self.id_column = id_column
        self.strong_columns = strong_columns
        self.weak_columns = weak_columns
        self.reference_columns = reference_columns
        self._save_extra_info = save_extra_info
        self.doc_metadata = metadata

    @property
    def hash(self) -> str:
        return self._hash

    @property
    def size(self) -> int:
        return len(self.df)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def matched_constraints(self) -> Dict[str, ConstraintValue]:
        return {key: ConstraintValue(value) for key, value in self.doc_metadata.items()}

    def all_entity_ids(self) -> List[int]:
        return self.df[self.id_column].to_list()

    def strong_text(self, element_id: int, chunk: pd.DataFrame = None) -> str:
        row = self.df.iloc[element_id]
        return " ".join([str(row[col]).replace(",", "") for col in self.strong_columns])

    def weak_text(self, element_id: int) -> str:
        row = self.df.iloc[element_id]
        return " ".join([str(row[col]).replace(",", "") for col in self.weak_columns])

    def reference(self, element_id: int) -> Reference:
        if element_id >= len(self.df):
            _raise_unknown_doc_error(element_id)
        row = self.df.iloc[element_id]
        text = "\n\n".join([f"{col}: {row[col]}" for col in self.reference_columns])
        return Reference(
            document=self,
            element_id=element_id,
            text=text,
            source=str(self.path.absolute()),
            metadata={**row.to_dict(), **self.doc_metadata},
        )

    def context(self, element_id: int, radius) -> str:
        rows = self.df.iloc[
            max(0, element_id - radius) : min(len(self.df), element_id + radius + 1)
        ]

        return " ".join(
            [
                str(row[col])
                for col in self.reference_columns
                for _, row in rows.iterrows()
            ]
        )

    def __getstate__(self):
        state = self.__dict__.copy()

        # Remove the path attribute because it is not cross platform compatible
        del state["path"]

        # Save the filename so we can load it with the same name
        state["doc_name"] = self.name

        # End pickling functionality here to support old directory checkpoint save
        return state

    def __setstate__(self, state):
        # Add new attributes to state for older document object version backward compatibility
        if "_save_extra_info" not in state:
            state["_save_extra_info"] = True

        self.__dict__.update(state)

    def save_meta(self, directory: Path):
        # Let's copy the original CSV file to the provided directory
        if self.save_extra_info:
            shutil.copy(self.path, directory)

    def load_meta(self, directory: Path):
        # Since we've moved the CSV file to the provided directory, let's make
        # sure that we point to this CSV file.
        if hasattr(self, "doc_name"):
            self.path = directory / self.doc_name
        else:
            # this else statement handles the deprecated attribute "path" in self, we can remove this soon
            self.path = directory / self.path.name

        if not hasattr(self, "doc_metadata"):
            self.doc_metadata = {}


# Base class for PDF and DOCX classes because they share the same logic.
class Extracted(Document):
    def __init__(self, path: str, save_extra_info=True, metadata={}):
        path = str(path)
        self.df = self.process_data(path)
        self.hash_val = hash_file(path, metadata="extracted-" + str(metadata))
        self._save_extra_info = save_extra_info

        self.path = Path(path)
        self.doc_metadata = metadata

    def process_data(
        self,
        path: str,
    ) -> pd.DataFrame:
        raise NotImplementedError()

    @property
    def hash(self) -> str:
        return self.hash_val

    @property
    def size(self) -> int:
        return len(self.df)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def matched_constraints(self) -> Dict[str, ConstraintValue]:
        return {key: ConstraintValue(value) for key, value in self.doc_metadata.items()}

    def all_entity_ids(self) -> List[int]:
        return list(range(self.size))

    def strong_text(self, element_id: int, chunk: pd.DataFrame = None) -> str:
        return ""

    def weak_text(self, element_id: int, chunk: pd.DataFrame = None) -> str:
        return self.df["para"].iloc[element_id]

    def show_fn(text, source, **kwargs):
        return text

    def reference(self, element_id: int) -> Reference:
        if element_id >= len(self.df):
            _raise_unknown_doc_error(element_id)
        return Reference(
            document=self,
            element_id=element_id,
            text=self.df["display"].iloc[element_id],
            source=str(self.path.absolute()),
            metadata={**self.df.iloc[element_id].to_dict(), **self.doc_metadata},
        )

    def context(self, element_id, radius) -> str:
        if not 0 <= element_id or not element_id < self.size:
            raise ("Element id not in document.")

        rows = self.df.iloc[
            max(0, element_id - radius) : min(len(self.df), element_id + radius + 1)
        ]
        return "\n".join(rows["para"])

    def __getstate__(self):
        state = self.__dict__.copy()

        # Remove filename attribute because this is a deprecated attribute for Extracted
        if "filename" in state:
            del state["filename"]

        # In older versions of neural_db, we accidentally stored Path objects in the df.
        # This changes those objects to a string, because PosixPath can't be loaded in Windows
        def path_to_str(element):
            if isinstance(element, Path):
                return element.name
            return element

        state["df"] = state["df"].applymap(path_to_str)

        # Remove the path attribute because it is not cross platform compatible
        del state["path"]

        # Save the filename so we can load it with the same name
        state["doc_name"] = self.name

        return state

    def __setstate__(self, state):
        # Add new attributes to state for older document object version backward compatibility
        if "_save_extra_info" not in state:
            state["_save_extra_info"] = True
        if "filename" in state:
            state["path"] = state["filename"]

        self.__dict__.update(state)

    def save_meta(self, directory: Path):
        # Let's copy the original file to the provided directory
        if self.save_extra_info:
            shutil.copy(self.path, directory)

    def load_meta(self, directory: Path):
        # Since we've moved the file to the provided directory, let's make
        # sure that we point to this file.
        if hasattr(self, "doc_name"):
            self.path = directory / self.doc_name
        else:
            # this else statement handles the deprecated attribute "path" in self, we can remove this soon
            self.path = directory / self.path.name

        if not hasattr(self, "doc_metadata"):
            self.doc_metadata = {}


def process_pdf(path: str) -> pd.DataFrame:
    elements, success = pdf_parse.process_pdf_file(path)

    if not success:
        raise ValueError(f"Could not read PDF file: {path}")

    elements_df = pdf_parse.create_train_df(elements)

    return elements_df


def process_docx(path: str) -> pd.DataFrame:
    elements, success = doc_parse.get_elements(path)

    if not success:
        raise ValueError(f"Could not read DOCX file: {path}")

    elements_df = doc_parse.create_train_df(elements)

    return elements_df


class PDF(Extracted):
    def __init__(self, path: str, metadata={}):
        super().__init__(path=path, metadata=metadata)

    def process_data(
        self,
        path: str,
    ) -> pd.DataFrame:
        return process_pdf(path)


class DOCX(Extracted):
    def __init__(self, path: str, metadata={}):
        super().__init__(path=path, metadata=metadata)

    def process_data(
        self,
        path: str,
    ) -> pd.DataFrame:
        return process_docx(path)


class Unstructured(Extracted):
    def __init__(
        self, path: Union[str, Path], save_extra_info: bool = True, metadata={}
    ):
        super().__init__(path=path, save_extra_info=save_extra_info, metadata=metadata)

    def process_data(
        self,
        path: str,
    ) -> pd.DataFrame:
        if path.endswith(".pdf") or path.endswith(".docx"):
            raise NotImplementedError(
                "For PDF and DOCX FileTypes, use neuraldb.PDF and neuraldb.DOCX "
            )
        elif path.endswith(".pptx"):
            self.parser = PptxParse(path)

        elif path.endswith(".txt"):
            self.parser = TxtParse(path)

        elif path.endswith(".eml"):
            self.parser = EmlParse(path)

        else:
            raise Exception(f"File type is not yet supported")

        elements, success = self.parser.process_elements()

        if not success:
            raise ValueError(f"Could not read file: {path}")

        return self.parser.create_train_df(elements)


class URL(Document):
    def __init__(
        self,
        url: str,
        url_response: Response = None,
        save_extra_info: bool = True,
        title_is_strong: bool = False,
        metadata={},
    ):
        self.url = url
        self.df = self.process_data(url, url_response)
        self.hash_val = hash_string(url + str(metadata))
        self._save_extra_info = save_extra_info
        self._strong_column = "title" if title_is_strong else "text"
        self.doc_metadata = metadata

    def process_data(self, url, url_response=None) -> pd.DataFrame:
        # Extract elements from each file
        elements, success = url_parse.process_url(url, url_response)

        if not success or not elements:
            raise ValueError(f"Could not retrieve data from URL: {url}")

        elements_df = url_parse.create_train_df(elements)

        return elements_df

    @property
    def hash(self) -> str:
        return self.hash_val

    @property
    def size(self) -> int:
        return len(self.df)

    @property
    def name(self) -> str:
        return self.url

    @property
    def matched_constraints(self) -> Dict[str, ConstraintValue]:
        return {key: ConstraintValue(value) for key, value in self.doc_metadata.items()}

    def all_entity_ids(self) -> List[int]:
        return list(range(self.size))

    def strong_text(self, element_id: int, chunk: pd.DataFrame = None) -> str:
        return self.df[self._strong_column if self._strong_column else "text"].iloc[
            element_id
        ]

    def weak_text(self, element_id: int, chunk: pd.DataFrame = None) -> str:
        return self.df["text"].iloc[element_id]

    def reference(self, element_id: int) -> Reference:
        if element_id >= len(self.df):
            _raise_unknown_doc_error(element_id)
        return Reference(
            document=self,
            element_id=element_id,
            text=self.df["display"].iloc[element_id],
            source=self.url,
            metadata={"title": self.df["title"].iloc[element_id], **self.doc_metadata}
            if "title" in self.df.columns
            else self.doc_metadata,
        )

    def context(self, element_id, radius) -> str:
        if not 0 <= element_id or not element_id < self.size:
            raise ("Element id not in document.")
        rows = self.df.iloc[
            max(0, element_id - radius) : min(len(self.df), element_id + radius + 1)
        ]
        return "\n".join(rows["text"])

    def load_meta(self, directory: Path):
        if not hasattr(self, "doc_metadata"):
            self.doc_metadata = {}


class DocumentConnector(Document):
    def __init__(
        self,
        doc_name: str,
        connector: ConnectorType,
        doc_metadata={},
    ) -> None:
        self.doc_name = doc_name + ".csv"
        self._connector = connector
        self.doc_metadata = doc_metadata
        self.index_table: pd.DataFrame = None
        self.index_table_id_col = "Row_id"
        self._hash = self.hash_connection()

    @property
    def matched_constraints(self) -> Dict[str, ConstraintValue]:
        return {key: ConstraintValue(value) for key, value in self.doc_metadata.items()}

    def row_iterator(self, start_id: int):
        has_id_col = hasattr(self, "id_col")
        current_doc_row_id = 0
        for current_chunk in self.next_chunk():
            for idx in range(len(current_chunk)):
                if has_id_col:
                    row_id = current_chunk.iloc[idx][self.id_col]
                else:
                    row_id = current_doc_row_id
                    current_doc_row_id += 1

                DB_id = start_id + row_id

                yield DocumentRow(
                    element_id=DB_id,
                    strong=self.strong_text(
                        idx, current_chunk
                    ),  # Strong text from (idx)th row of the current_batch
                    weak=self.weak_text(
                        idx, current_chunk
                    ),  # Weak text from (idx)th row of the current_batch
                )
                self._add_entry(row_id)

    @property
    def connector(self):
        return self._connector

    @connector.setter
    def connector(self, connector: ConnectorType):
        self._connector = connector

    @property
    def name(self) -> str:
        return self.doc_name

    def next_chunk(self) -> pd.DataFrame:
        raise NotImplementedError()

    def reference(self, element_id: int) -> Reference:
        raise NotImplementedError()

    def strong_column(self, element_id, chunk: pd.DataFrame = None):
        raise NotImplementedError()

    def weak_column(self, element_id, chunk: pd.DataFrame = None):
        raise NotImplementedError()

    @property
    def hash(self):
        return self._hash

    def _add_entry(self, current_doc_row_id: int):
        if (
            len(
                self.index_table.loc[
                    self.index_table[self.index_table_id_col] == current_doc_row_id
                ]
            )
            > 0
        ):
            # row_iterator is being called twice, so add first time only
            return

        self.add_index_entry(current_doc_row_id)

    def add_index_entry(self, current_doc_row_id: int):
        raise NotImplementedError()

    def hash_connection(self):
        raise NotImplementedError()

    def save_meta(self, directory: Path):
        # Save the index table
        if self.save_extra_info:
            self.index_table.to_csv(path_or_buf=directory / self.doc_name, index=False)

    def load_meta(self, directory: Path):
        # Since we've moved the file to the provided directory, let's make
        # sure that we point to this file.
        if hasattr(self, "doc_name"):
            self.path = directory / self.doc_name
        else:
            # this else statement handles the deprecated attribute "path" in self, we can remove this soon
            self.path = directory / self.path.name

        if not hasattr(self, "doc_metadata"):
            self.doc_metadata = {}

    def __getstate__(self):
        state = self.__dict__.copy()

        del state["_connector"]

        return state

    def __setstate__(self, state):
        # Add new attributes to state for older document object version backward compatibility
        if "_save_extra_info" not in state:
            state["_save_extra_info"] = True

        self.__dict__.update(state)


class SQLDocument(DocumentConnector):
    def __init__(
        self,
        engine: sqlConn,
        table_name: str,
        id_col: str,
        strong_columns: List[str],
        weak_columns: List[str],
        reference_columns: List[str],
        chunk_size=100_00,
        doc_metadata={},
    ) -> None:
        self.table_name = table_name
        self._connector = SQLConnector(
            engine=engine,
            columns=[id_col] + strong_columns + weak_columns + reference_columns,
            chunk_size=chunk_size,
            table_name=table_name,
        )
        super().__init__(
            doc_name=table_name, connector=self._connector, doc_metadata=doc_metadata
        )
        self.id_col = id_col
        self.strong_columns = strong_columns
        self.weak_columns = weak_columns
        self.reference_columns = reference_columns

        self.total_rows = self._connector.total_rows()

        self.integrity_check()

        self.index_table = pd.DataFrame(columns=[self.index_table_id_col])

        self._doc_metadata = doc_metadata

    def next_chunk(self) -> pd.DataFrame:
        return self._connector.next_chunk()

    @property
    def size(self) -> int:
        return self.total_rows

    def hash_connection(self) -> str:
        self.engine_uq = str(self._connector.get_engine_url()) + f"/{self.table_name}"
        return hash_string(self.engine_uq)

    def all_entity_ids(self) -> List[int]:
        if self.index_table:
            return self.index_table[self.index_table_id_col].to_list()
        else:
            id_rows = self._connector.get_rows(cols=self.id_col).fetchall()
            return [temp[0] for temp in id_rows]

    def reference(self, element_id: int) -> Reference:
        if element_id >= self.size:
            _raise_unknown_doc_error(element_id)

        try:
            reference_texts = self._connector.exexute(
                query=f"SELECT {','.join(self.reference_columns)} FROM {self.table_name} WHERE {self.id_col} = {element_id}"
            ).fetchone()

            if len(self.reference_columns) == 1:
                # Returned result is in the form '(text, )'
                reference_texts = reference_texts[0]

            text = "\n\n".join(
                [
                    f"{col_name}: {col_text}"
                    for col_name, col_text in zip(
                        self.reference_columns, reference_texts
                    )
                ]
            )

        except Exception as e:
            text = f"Unable to connect to database, line no: {element_id}"

        return Reference(
            document=self,
            element_id=element_id,
            text=text,
            source=self.engine_uq,
            metadata={**self.doc_metadata},
        )

    def strong_text(self, element_id: int, chunk: pd.DataFrame = None) -> str:
        row = chunk.iloc[element_id]
        return " ".join([str(row[col]).replace(",", "") for col in self.strong_columns])

    def weak_text(self, element_id: int, chunk: pd.DataFrame = None) -> str:
        row = chunk.iloc[element_id]
        return " ".join([str(row[col]).replace(",", "") for col in self.weak_columns])

    def add_index_entry(self, current_doc_row_id: int):
        self.index_table.loc[len(self.index_table)] = {
            self.index_table_id_col: current_doc_row_id
        }

    def integrity_check(self):
        if not (
            len(self.strong_columns) > 0 and len(self.weak_columns) > 0 and self.id_col
        ):
            raise_attribute_error("Empty strong OR weak OR reference columns")
        all_cols = self._connector.cols_metadata()

        all_col_name = set([col["name"] for col in all_cols])

        if not (
            self.id_col in all_col_name
            and set(self.strong_columns).issubset(all_col_name)
            and set(self.weak_columns).issubset(all_col_name)
            and set(self.reference_columns).issubset(all_col_name)
        ):
            raise_attribute_error("Provided column name doesn't exists in the table")

        primary_keys = self._connector.get_primary_keys()
        if not primary_keys:
            raise_attribute_error(f"{self.id_col} needs to be a primary key")
        elif len(primary_keys) > 1:
            raise_attribute_error("Composite primary key is not allowed")

        for col in all_cols:
            if col["name"] == self.id_col and not isinstance(col["type"], Integer):
                raise_attribute_error("id column needs to be of type Integer")

            if col["name"] in self.strong_columns and not isinstance(
                col["type"], String
            ):
                raise_attribute_error(
                    f"strong column '{col['name']}' needs to be of type String"
                )

            if col["name"] in self.weak_columns and not isinstance(col["type"], String):
                raise_attribute_error(
                    f"weak column '{col['name']}' needs to be of type String"
                )

        min_id = self._connector.execute(
            query=f"SELECT MIN({self.id_col}) FROM {self.table_name}"
        ).fetchone()[0]

        max_id = self.connector.execute(
            query=f"SELECT MAX({self.id_col}) FROM {self.table_name}"
        ).fetchone()[0]

        if min_id != 0 or max_id != self.size - 1:
            raise_attribute_error(
                f"id column needs to be unique from 0 to {self.size - 1}"
            )


class SentenceLevelExtracted(Extracted):
    """Parses a document into sentences and creates a NeuralDB entry for each
    sentence. The strong column of the entry is the sentence itself while the
    weak column is the paragraph from which the sentence came. A NeuralDB
    reference produced by this object displays the paragraph instead of the
    sentence to increase recall.
    """

    def __init__(self, path: str, save_extra_info: bool = True, metadata={}):
        self.path = Path(path)
        self.df = self.parse_sentences(self.process_data(path))
        self.hash_val = hash_file(
            path, metadata="sentence-level-extracted-" + str(metadata)
        )
        self.para_df = self.df["para"].unique()
        self._save_extra_info = save_extra_info
        self.doc_metadata = metadata

    def not_just_punctuation(sentence: str):
        for character in sentence:
            if character not in string.punctuation and not character.isspace():
                return True
        return False

    def get_sentences(paragraph: str):
        return [
            sentence
            for sentence in sent_tokenize(paragraph)
            if SentenceLevelExtracted.not_just_punctuation(sentence)
        ]

    def parse_sentences(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        df["sentences"] = df["para"].apply(SentenceLevelExtracted.get_sentences)

        num_sents_cum_sum = np.cumsum(df["sentences"].apply(lambda sents: len(sents)))
        df["id_offsets"] = np.zeros(len(df))
        df["id_offsets"][1:] = num_sents_cum_sum[:-1]
        df["id_offsets"] = df["id_offsets"].astype(int)

        def get_ids(record):
            id_offset = record["id_offsets"]
            n_sents = len(record["sentences"])
            return list(range(id_offset, id_offset + n_sents))

        df = pd.DataFrame.from_records(
            [
                {
                    "sentence": sentence,
                    "para_id": para_id,
                    "sentence_id": i + record["id_offsets"],
                    "sentence_ids_in_para": get_ids(record),
                    **record,
                }
                for para_id, record in enumerate(df.to_dict(orient="records"))
                for i, sentence in enumerate(record["sentences"])
            ]
        )

        df.drop("sentences", axis=1, inplace=True)
        df.drop("id_offsets", axis=1, inplace=True)
        return df

    def process_data(
        self,
        path: str,
    ) -> pd.DataFrame:
        raise NotImplementedError()

    @property
    def hash(self) -> str:
        return self.hash_val

    @property
    def size(self) -> int:
        return len(self.df)

    @property
    def name(self) -> str:
        return self.path.name if self.path else None

    def strong_text(self, element_id: int) -> str:
        return self.df["sentence"].iloc[element_id]

    def weak_text(self, element_id: int) -> str:
        return self.df["para"].iloc[element_id]

    def show_fn(text, source, **kwargs):
        return text

    def reference(self, element_id: int) -> Reference:
        if element_id >= len(self.df):
            _raise_unknown_doc_error(element_id)
        return Reference(
            document=self,
            element_id=element_id,
            text=self.df["display"].iloc[element_id],
            source=str(self.path.absolute()),
            metadata={**self.df.iloc[element_id].to_dict(), **self.doc_metadata},
            upvote_ids=self.df["sentence_ids_in_para"].iloc[element_id],
        )

    def context(self, element_id, radius) -> str:
        if not 0 <= element_id or not element_id < self.size:
            raise ("Element id not in document.")

        para_id = self.df.iloc[element_id]["para_id"]

        rows = self.para_df[
            max(0, para_id - radius) : min(len(self.para_df), para_id + radius + 1)
        ]
        return "\n\n".join(rows)

    def save_meta(self, directory: Path):
        # Let's copy the original file to the provided directory
        if self.save_extra_info:
            shutil.copy(self.path, directory)

    def load_meta(self, directory: Path):
        # Since we've moved the file to the provided directory, let's make
        # sure that we point to this file.
        if hasattr(self, "doc_name"):
            self.path = directory / self.doc_name
        else:
            # deprecated, self.path should not be in self
            self.path = directory / self.path.name

        if not hasattr(self, "doc_metadata"):
            self.doc_metadata = {}


class SentenceLevelPDF(SentenceLevelExtracted):
    def __init__(self, path: str, metadata={}):
        super().__init__(path=path, metadata=metadata)

    def process_data(
        self,
        path: str,
    ) -> pd.DataFrame:
        return process_pdf(path)


class SentenceLevelDOCX(SentenceLevelExtracted):
    def __init__(self, path: str, metadata={}):
        super().__init__(path=path, metadata=metadata)

    def process_data(
        self,
        path: str,
    ) -> pd.DataFrame:
        return process_docx(path)
