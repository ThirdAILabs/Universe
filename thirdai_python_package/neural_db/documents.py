import hashlib
import json
import os
import pickle
import shutil
import string
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from office365.sharepoint.client_context import (
    ClientContext,
    ClientCredential,
    UserCredential,
)
from pytrie import StringTrie
from requests.models import Response
from simple_salesforce import Salesforce
from sqlalchemy import Integer, String, create_engine
from sqlalchemy.engine.base import Connection as sqlConn
from thirdai import bolt
from thirdai.data import get_udt_col_types
from thirdai.dataset.data_source import PyDataSource

from .connectors import SalesforceConnector, SharePointConnector, SQLConnector
from .constraint_matcher import (
    ConstraintMatcher,
    ConstraintValue,
    Filter,
    TableFilter,
    to_filters,
)
from .parsing_utils import doc_parse, pdf_parse, sliding_pdf_parse, url_parse
from .table import DaskDataFrameTable, DataFrameTable, SQLiteTable
from .utils import hash_file, hash_string, requires_condition


class Reference:
    pass


def _raise_unknown_doc_error(element_id: int):
    raise ValueError(f"Unable to find document that has id {element_id}.")


class Document:
    @property
    def size(self) -> int:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def source(self) -> str:
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

    def filter_entity_ids(self, filters: Dict[str, Filter]):
        return self.all_entity_ids()

    def id_map(self) -> Optional[Dict[str, int]]:
        return None

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

    def strong_text(self, element_id: int) -> str:
        return self.reference(element_id).text

    def weak_text(self, element_id: int) -> str:
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

    def row_iterator(self):
        for i in range(self.size):
            yield DocumentRow(
                element_id=i,
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
        retriever: str = None,
    ):
        self._id = element_id
        self._id_in_document = element_id
        self._upvote_ids = upvote_ids if upvote_ids is not None else [element_id]
        self._text = text
        self._source = source
        self._metadata = metadata
        self._context_fn = lambda radius: document.context(element_id, radius)
        self._score = 0
        self._document = document
        self._retriever = retriever

    @property
    def id(self):
        return self._id

    @property
    def id_in_document(self):
        return self._id_in_document

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

    @property
    def document(self):
        return self._document

    @property
    def retriever(self):
        return self._retriever

    def context(self, radius: int):
        return self._context_fn(radius)

    def __eq__(self, other):
        if isinstance(other, Reference):
            return (
                self.id == other.id
                and self.text == other.text
                and self.source == other.source
            )
        return False


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
        for col in [id_column, strong_column, weak_column]:
            if '"' in col or "," in col:
                raise RuntimeError(
                    "DocumentDataSource columns cannot contain '\"' or ','"
                )
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
            for row in doc.row_iterator():
                row.id = row.id + start_id
                yield row

    def indices(self):
        indices = []
        for doc, start_id in self.documents:
            for row in doc.row_iterator():
                indices.append(row.id + start_id)

        return indices

    @property
    def size(self):
        return self._size

    def _csv_line(self, element_id: str, strong: str, weak: str):
        csv_strong = '"' + strong.replace('"', '""') + '"'
        csv_weak = '"' + weak.replace('"', '""') + '"'
        return f"{element_id},{csv_strong},{csv_weak}"

    def _get_line_iterator(self):
        # First yield the header
        yield f"{self.id_column},{self.strong_column},{self.weak_column}"
        # Then yield rows
        for row in self.row_iterator():
            yield self._csv_line(element_id=row.id, strong=row.strong, weak=row.weak)

    def resource_name(self) -> str:
        return "Documents:\n" + "\n".join([doc.name for doc, _ in self.documents])

    def save(self, path: Path, save_interval=100_000):
        """
        DocumentDataSource is agnostic to the documents that are a part of it as the line_iterator is agnostic to the kind of document and returns data in a specific format. Hence, to serialize DocumentDataSource, we do not need to serialize the documents but rather, dump the lines yielded by the line iterator into a CSV. This makes the saving and loading logic simpler.
        """
        path.mkdir(exist_ok=True, parents=True)
        number_lines_in_buffer = 0
        with open(path / "source.csv", "w", encoding="utf-8") as f:
            for line in self._get_line_iterator():
                f.write(line + "\n")
                number_lines_in_buffer += 1
            if number_lines_in_buffer > save_interval:
                f.flush()
                number_lines_in_buffer = 0

        with open(path / "arguments.json", "w") as f:
            json.dump(
                {
                    "id_column": self.id_column,
                    "strong_column": self.strong_column,
                    "weak_column": self.weak_column,
                },
                f,
                indent=4,
            )
        self.restart()

    @staticmethod
    def load(path: Path):
        with open(path / "arguments.json", "r") as f:
            args = json.load(f)

        csv_document = CSV(
            path=path / "source.csv",
            id_column=args["id_column"],
            strong_columns=[args["strong_column"]],
            weak_columns=[args["weak_column"]],
            has_offset=True,
        )
        data_source = DocumentDataSource(**args)
        data_source.add(csv_document, start_id=0)
        return data_source


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

    def delete(self, source_ids):
        # TODO(Geordie): Error handling
        all_sources_exist = all(source_id in self.registry for source_id in source_ids)
        if not all_sources_exist:
            raise KeyError("At least one source not found in document manager.")

        deleted_entities = []
        for source_id in source_ids:
            doc, offset = self.registry[source_id]
            deleted_entities += [
                offset + entity_id for entity_id in doc.all_entity_ids()
            ]
            del self.registry[source_id]
            del self.source_id_prefix_trie[source_id]
            self.constraint_matcher.delete((doc, offset), doc.matched_constraints)

        return deleted_entities

    def entity_ids_by_constraints(self, constraints: Dict[str, Any]):
        filters = to_filters(constraints)
        return [
            start_id + entity_id
            for doc, start_id in self.constraint_matcher.match(filters)
            for entity_id in doc.filter_entity_ids(filters)
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
        self.constraint_matcher = ConstraintMatcher[DocAndOffset]()

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


def safe_has_offset(this):
    """Checks the value of the "has_offset" attribute of a class.
    Defaults to False when the attribute does not exist.
    This function is needed for backwards compatibility reasons.
    """
    if hasattr(this, "has_offset"):
        return this.has_offset
    return False


def create_table(df, on_disk):
    Table = (
        SQLiteTable
        if on_disk
        else DaskDataFrameTable if isinstance(df, dd.DataFrame) else DataFrameTable
    )
    return Table(df)


def metadata_with_source(metadata, source: str):
    if "source" in metadata:
        raise ValueError(
            "Document metadata cannot contain the key 'source'. 'source' is a reserved key."
        )
    return {**metadata, "source": source}


class CSV(Document):
    """
    A document containing the rows of a csv file.

    Args:
        path (str): The path to the csv file.
        id_column (Optional[str]). Optional, defaults to None. If provided then the
            ids in this column are used to identify the rows in NeuralDB. If not provided
            then ids are assigned.
        strong_columns (Optional[List[str]]): Optional, defaults to None. This argument
            can be used to provide NeuralDB with information about which columns are
            likely to contain the strongest signal in matching with a given query. For
            example this could be something like the name of a product.
        weak_columns (Optional[List[str]]): Optional, defaults to None. This argument
            can be used to provide NeuralDB with information about which columns are
            likely to contain weaker signals in matching with a given query. For
            example this could be something like the description of a product.
        reference_columns (Optional[List[str]]): Optional, defaults to None. If provided
            the specified columns are returned by NeuralDB as responses to queries. If
            not specifed all columns are returned.
        metadata (Dict[str, Any]): Optional, defaults to {}. Specifies metadata to
            associate with entities from this file. Queries to NeuralDB can provide
            constrains to restrict results based on the metadata.
    """

    def valid_id_column(column):
        if isinstance(column, dd.Series):
            unique_count = column.nunique().compute()
            min_val = column.min().compute()
            max_val = column.max().compute()
            length = column.size.compute()
            condition = (
                (unique_count == length) and (min_val == 0) and (max_val == length - 1)
            )
        else:
            condition = (
                (len(column.unique()) == len(column))
                and (column.min() == 0)
                and (column.max() == len(column) - 1)
            )

        return condition

    def remove_spaces(column_name):
        return column_name.replace(" ", "_")

    def remove_spaces_from_list(column_name_list):
        return [CSV.remove_spaces(col) for col in column_name_list]

    # blocksize (when using dask) Determines the size of each partition/chunk in bytes.
    # For example, setting blocksize=25e6 will aim for partitions of approximately 25MB.
    # If you decrease the block size, Dask will create more partitions, and
    # increasing it will result in fewer partitions.
    # Default value is computed based on available physical memory
    # and the number of cores, up to a maximum of 64MB.
    def __init__(
        self,
        path: str,
        id_column: Optional[str] = None,
        strong_columns: Optional[List[str]] = None,
        weak_columns: Optional[List[str]] = None,
        reference_columns: Optional[List[str]] = None,
        save_extra_info=True,
        metadata=None,
        has_offset=False,
        on_disk=False,
        use_dask=False,
        blocksize=None,
    ) -> None:
        if use_dask:
            df = (
                dd.read_csv(path, blocksize=blocksize)
                if blocksize
                else dd.read_csv(path)
            )
        else:
            df = pd.read_csv(path)

        # Convert spaces in column names to underscores because df.itertuples
        # does not work when there are spaces
        # https://stackoverflow.com/questions/45307376/pandas-df-itertuples-renaming-dataframe-columns-when-printing
        # While it's possible that saved models contain column names that have
        # spaces. We don't convert these columns during deserialization because
        # df.itertuples is only called during CSV construction and during
        # insertion, which would have completed before serialization.
        # Additionally, this document's hash takes column names into account.
        # Consider this scenario:
        # 1. User inserts CSV with spaced column names into an older version of NDB
        # 2. User saves the NDB model
        # 3. User upgrades the ThirdAI package
        # 4. User loads the saved NDB model
        # 5. User inserts the same CSV into the loaded model
        # Here, NeuralDB will actually treat the CSV as a new, unseen document,
        # so it will not invoke df.itertuples on a dataframe that has spaced
        # column names.
        cols_with_spaces = [col for col in df.columns if " " in col]
        self.with_space_to_no_space = {}
        if cols_with_spaces:
            for col in cols_with_spaces:
                self.with_space_to_no_space[col] = col.replace(" ", "_")
                while self.with_space_to_no_space[col] in df.columns:
                    self.with_space_to_no_space[col] += "_"

            def remove_spaces_from_list(cols):
                return [self.with_space_to_no_space.get(col, col) for col in cols]

            df.columns = remove_spaces_from_list(df.columns)
            if id_column:
                id_column = self.with_space_to_no_space.get(id_column, id_column)
            if strong_columns:
                strong_columns = remove_spaces_from_list(strong_columns)
            if weak_columns:
                weak_columns = remove_spaces_from_list(weak_columns)
            if reference_columns:
                reference_columns = remove_spaces_from_list(reference_columns)

        self.no_space_to_with_space = {
            val: key for key, val in self.with_space_to_no_space.items()
        }

        # This variable is used to check whether the id's in the CSV are supposed to start with 0 or with some custom offset. We need the latter when we shard the datasource.
        self.has_offset = has_offset

        if reference_columns is None:
            reference_columns = list(df.columns)

        self.orig_to_assigned_id = None
        self.id_column = id_column
        orig_id_column = id_column
        if self.id_column and (has_offset or CSV.valid_id_column(df[self.id_column])):
            df = df.sort_values(self.id_column)
        else:
            self.id_column = "thirdai_index"
            if use_dask:
                # sets dask df index column to range(len(df))
                df[self.id_column] = (
                    df.assign(partition_count=1).partition_count.cumsum() - 1
                )
            else:
                df[self.id_column] = range(df.shape[0])

            if orig_id_column:
                self.orig_to_assigned_id = {
                    str(getattr(row, orig_id_column)): getattr(row, self.id_column)
                    for row in df.itertuples(index=True)
                }

        if strong_columns is None and weak_columns is None:
            # autotune column types
            text_col_names = []
            try:
                for col_name, udt_col_type in get_udt_col_types(path).items():
                    if type(udt_col_type) == type(bolt.types.text()):
                        text_col_names.append(CSV.remove_spaces(col_name))
            except:
                text_col_names = list(df.columns)
                text_col_names.remove(id_column)
                if orig_id_column:
                    text_col_names.remove(orig_id_column)
                df[text_col_names] = df[text_col_names].astype(str)
            strong_columns = []
            weak_columns = text_col_names
        elif strong_columns is None:
            strong_columns = []
        elif weak_columns is None:
            weak_columns = []

        for col in strong_columns + weak_columns:
            df[col] = df[col].fillna("")

        if use_dask:
            # The 'sorted=True' parameter is used to indicate that the column is already sorted.
            # This optimization helps Dask to avoid expensive data shuffling operations, improving performance.
            df = df.set_index(self.id_column, sorted=True)
        else:
            # Pandas automatically manages the index without needing to explicitly sort it here.
            df = df.set_index(self.id_column)

        self.table = create_table(df, on_disk)

        self.path = Path(path)
        self.strong_columns = strong_columns
        self.weak_columns = weak_columns
        self.reference_columns = [
            col for col in reference_columns if col != self.id_column
        ]
        self._save_extra_info = save_extra_info
        self.doc_metadata = metadata_with_source(metadata or {}, Path(path).name)
        self.doc_metadata_keys = set(self.doc_metadata.keys())
        # Add column names to hash metadata so that CSVs with different
        # hyperparameters are treated as different documents. Otherwise, this
        # may break training.
        self._hash = hash_file(
            path,
            metadata="csv-"
            + str(self.id_column)
            + str(sorted(self.strong_columns))
            + str(sorted(self.weak_columns))
            + str(sorted(self.reference_columns))
            + str(sorted(list(self.doc_metadata.items())))
            + str(sorted(self.table.columns)),
        )

    @property
    def hash(self) -> str:
        return self._hash

    @property
    def size(self) -> int:
        return self.table.size

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def source(self) -> str:
        return str(self.path.absolute())

    @requires_condition(
        check_func=lambda self: not safe_has_offset(self),
        method_name="matched_constraints",
        method_class="CSV(Document)",
        condition_unmet_string=" when there is an offset in the CSV document",
    )
    @property
    def matched_constraints(self) -> Dict[str, ConstraintValue]:
        metadata_constraints = {
            key: ConstraintValue(value) for key, value in self.doc_metadata.items()
        }
        indexed_column_constraints = {
            self.no_space_to_with_space.get(key, key): ConstraintValue(is_any=True)
            for key in self.table.columns
        }
        return {**metadata_constraints, **indexed_column_constraints}

    def all_entity_ids(self) -> List[int]:
        return self.table.ids

    def filter_entity_ids(self, filters: Dict[str, Filter]):
        table_filter = TableFilter(
            {
                self.with_space_to_no_space.get(k, k): v
                for k, v in filters.items()
                if k not in self.doc_metadata_keys
            }
        )
        return self.table.apply_filter(table_filter)

    def id_map(self) -> Optional[Dict[str, int]]:
        return self.orig_to_assigned_id

    def strong_text_from_row(self, row) -> str:
        return " ".join(str(row[col]) for col in self.strong_columns)

    def strong_text(self, element_id: int) -> str:
        row = self.table.row_as_dict(element_id)
        return self.strong_text_from_row(row)

    def weak_text_from_row(self, row) -> str:
        return " ".join(str(row[col]) for col in self.weak_columns)

    def weak_text(self, element_id: int) -> str:
        row = self.table.row_as_dict(element_id)
        return self.weak_text_from_row(row)

    def row_iterator(self):
        for row_id, row in self.table.iter_rows_as_dicts():
            yield DocumentRow(
                element_id=row_id,
                strong=self.strong_text_from_row(row),
                weak=self.weak_text_from_row(row),
            )

    @requires_condition(
        check_func=lambda self: not safe_has_offset(self),
        method_name="reference",
        method_class="CSV(Document)",
        condition_unmet_string=" when there is an offset in the CSV document",
    )
    def reference(self, element_id: int) -> Reference:
        if element_id >= self.table.size:
            _raise_unknown_doc_error(element_id)
        row = self.table.row_as_dict(element_id)
        text = "\n\n".join(
            [
                f"{self.no_space_to_with_space.get(col, col)}: {row[col]}"
                for col in self.reference_columns
            ]
        )
        row = {
            self.no_space_to_with_space.get(col, col): val
            for col, val in row.items()
            if col != "thirdai_index"
        }
        return Reference(
            document=self,
            element_id=element_id,
            text=text,
            source=self.source,
            metadata={**row, **self.doc_metadata},
        )

    def context(self, element_id: int, radius) -> str:
        rows = self.table.range_rows_as_dicts(
            from_row_id=max(0, element_id - radius),
            to_row_id=min(self.table.size, element_id + radius + 1),
        )

        return " ".join(
            [
                "\n\n".join(
                    [
                        f"{self.no_space_to_with_space.get(col, col)}: {row[col]}"
                        for col in self.reference_columns
                    ]
                )
                for row in rows
            ]
        )

    def __getstate__(self):
        state = self.__dict__.copy()

        # Save the filename so we can load it with the same name
        state["doc_name"] = self.name

        state["path"] = str(self.path)

        # End pickling functionality here to support old directory checkpoint save
        return state

    def __setstate__(self, state):
        # Add new attributes to state for older document object version backward compatibility
        if "_save_extra_info" not in state:
            state["_save_extra_info"] = True

        if "path" in state:
            state["path"] = Path(state["path"])

        self.__dict__.update(state)

    @requires_condition(
        check_func=lambda self: not safe_has_offset(self),
        method_name="save_meta",
        method_class="CSV(Document)",
        condition_unmet_string=" when there is an offset in the CSV document",
    )
    def save_meta(self, directory: Path):
        # Let's copy the original CSV file to the provided directory
        if self.save_extra_info:
            shutil.copy(self.path, directory)
        self.table.save_meta(directory)

    @requires_condition(
        check_func=lambda self: not safe_has_offset(self),
        method_name="load_meta",
        method_class="CSV(Document)",
        condition_unmet_string=" when there is an offset in the CSV document",
    )
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
        if not hasattr(self, "doc_metadata_keys"):
            self.doc_metadata_keys = set()
        if not hasattr(self, "orig_to_assigned_id"):
            self.orig_to_assigned_id = None
        if not hasattr(self, "has_offset"):
            self.has_offset = False

        if hasattr(self, "df"):
            if self.df.index.name != self.id_column:
                self.reference_columns = [
                    col for col in self.reference_columns if col != self.id_column
                ]
                self.df = self.df.set_index(self.id_column)
            self.table = DataFrameTable(self.df)
            del self.df
        else:
            self.table.load_meta(directory)

        if hasattr(self, "with_space_to_no_space"):
            self.no_space_to_with_space = {
                val: key for key, val in self.with_space_to_no_space.items()
            }
        else:
            self.with_space_to_no_space = {}
            self.no_space_to_with_space = {}


# Base class for PDF, DOCX and Unstructured classes because they share the same logic.
class Extracted(Document):
    def __init__(
        self,
        path: str,
        save_extra_info=True,
        metadata=None,
        strong_column=None,
        on_disk=False,
    ):
        path = str(path)
        df = self.process_data(path)
        self.table = create_table(df, on_disk)
        self.hash_val = hash_file(path, metadata="extracted-" + str(metadata))
        self._save_extra_info = save_extra_info

        self.path = Path(path)
        self.doc_metadata = metadata_with_source(metadata or {}, Path(path).name)
        self.strong_column = strong_column
        if self.strong_column and self.strong_column not in self.table.columns:
            raise RuntimeError(
                f"Strong column '{self.strong_column}' not found in the dataframe."
            )

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
        return self.table.size

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def source(self) -> str:
        return str(self.path.absolute())

    @property
    def matched_constraints(self) -> Dict[str, ConstraintValue]:
        return {key: ConstraintValue(value) for key, value in self.doc_metadata.items()}

    def all_entity_ids(self) -> List[int]:
        return list(range(self.size))

    def strong_text(self, element_id: int) -> str:
        return (
            ""
            if not self.strong_column
            else self.table.field(element_id, self.strong_column)
        )

    def weak_text(self, element_id: int) -> str:
        return self.table.field(element_id, "para")

    def show_fn(text, source, **kwargs):
        return text

    def reference(self, element_id: int) -> Reference:
        if element_id >= self.table.size:
            _raise_unknown_doc_error(element_id)
        return Reference(
            document=self,
            element_id=element_id,
            text=self.table.field(element_id, "display"),
            source=self.source,
            metadata={**self.table.row_as_dict(element_id), **self.doc_metadata},
        )

    def context(self, element_id, radius) -> str:
        if not 0 <= element_id or not element_id < self.size:
            raise ("Element id not in document.")

        rows = self.table.range_rows_as_dicts(
            from_row_id=max(0, element_id - radius),
            to_row_id=min(self.table.size, element_id + radius + 1),
        )
        return "\n".join(row["para"] for row in rows)

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

        if "df" in state:
            state["df"] = state["df"].applymap(path_to_str)

        # Save the filename so we can load it with the same name
        state["doc_name"] = self.name

        state["path"] = str(self.path)

        return state

    def __setstate__(self, state):
        # Add new attributes to state for older document object version backward compatibility
        if "_save_extra_info" not in state:
            state["_save_extra_info"] = True
        if "filename" in state:
            state["path"] = state["filename"]

        if "path" in state:
            state["path"] = Path(state["path"])

        self.__dict__.update(state)

    def save_meta(self, directory: Path):
        # Let's copy the original file to the provided directory
        if self.save_extra_info:
            shutil.copy(self.path, directory)
        self.table.save_meta(directory)

    def load_meta(self, directory: Path):
        # Since we've moved the file to the provided directory, let's make
        # sure that we point to this file.
        if self.save_extra_info:
            if hasattr(self, "doc_name"):
                self.path = directory / self.doc_name
            else:
                # this else statement handles the deprecated attribute "path" in self, we can remove this soon
                self.path = directory / self.path.name

        if not hasattr(self, "doc_metadata"):
            self.doc_metadata = {}

        if not hasattr(self, "strong_column"):
            self.strong_column = None

        if hasattr(self, "df"):
            self.table = DataFrameTable(self.df)
            del self.df
        elif hasattr(self, "table"):
            self.table.load_meta(directory)


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
    """
    Parses a PDF document into chunks of text that can be indexed by NeuralDB.

    Args:
        path (str): path to PDF file
        version (str): Either "v1" or "v2". If "v1", the parser splits the PDF
            into paragraphs. If "v2", the parser creates overlapping chunks
            comprised of entire lines from the PDF. "v2" does more data cleaning
            and therefore supports more options, which are outlined below.
        chunk_size (int): Only relevant if version = "v2".
            The number of words in each chunk of text. Defaults to 100
        stride (int): Only relevant if version = "v2". The number of words
            between each chunk of text. When stride < chunk_size, the text
            chunks overlap. When stride = chunk_size, the text chunks do not
            overlap. Defaults to 40 so adjacent chunks have a 60% overlap.
        emphasize_first_words (int): Only relevant if version = "v2".
            The number of words at the beginning of the document to be passed
            into NeuralDB as a strong signal. For example, if your document
            starts with a descriptive title that is 3 words long, then you can
            set emphasize_first_words to 3 so that NeuralDB captures this strong
            signal. Defaults to 0.
        ignore_header_footer (bool): Only relevant if version = "v2". Whether
            the parser should remove headers and footers. Defaults to True;
            headers and footers are removed by default.
        ignore_nonstandard_orientation (bool): Only relevant if version = "v2".
            Whether the parser should remove lines of text that have a
            nonstandard orientation, such as margins that are oriented
            vertically. Defaults to True; lines with nonstandard orientation are
            removed by default.
        metadata (Dict[str, Any]): Optional, defaults to {}. Specifies metadata to
            associate with entities from this file. Queries to NeuralDB can provide
            constrains to restrict results based on the metadata.
        on_disk (bool): If True, the processed chunks will be stored in a
            lightweight on-disk database. Otherwise, processed chunks will be
            stored in-memory. Defaults to False.
        doc_keywords (str):  Only relevant if version = "v2". If provided, the
            keywords will be prepended to every chunk in the document. It is
            helpful for use cases where a NeuralDB instance contains multiple
            documents. Defaults to an empty string.
        emphasize_section_titles (bool): Only relevant if version = "v2". If
            True, infers section titles based on font properties and prepends
            the latest section title to each chunk. Defaults to False.
        table_parsing (bool): Only relevant if version = "v2". If True, the
            contents of a table are considered to be contained in a single line,
            ensuring that any chunk that contains a table contains the entire
            table. Defaults to False.
        save_extra_info (bool): If True, the original PDF file will be saved in
            .ndb checkpoint. Defaults to True.
    """

    def __init__(
        self,
        path: str,
        version: str = "v1",
        chunk_size=100,
        stride=40,
        emphasize_first_words=0,
        ignore_header_footer=True,
        ignore_nonstandard_orientation=True,
        metadata=None,
        on_disk=False,
        doc_keywords="",
        emphasize_section_titles=False,
        table_parsing=False,
        save_extra_info=True,
    ):
        self.version = version

        if version == "v1":
            super().__init__(
                path=path,
                metadata=metadata,
                on_disk=on_disk,
                save_extra_info=save_extra_info,
            )
            return

        if version != "v2":
            raise ValueError(
                f"Received invalid version '{version}'. Choose between 'v1' and 'v2'"
            )

        self.chunk_size = chunk_size
        self.stride = stride
        self.emphasize_first_words = emphasize_first_words
        self.ignore_header_footer = ignore_header_footer
        self.ignore_nonstandard_orientation = ignore_nonstandard_orientation
        self.doc_keywords = doc_keywords
        self.emphasize_section_titles = emphasize_section_titles
        self.table_parsing = table_parsing
        # Add pdf version, chunk size, and stride metadata. The metadata will be
        # incorporated in the document hash so that the same PDF inserted with
        # different hyperparameters are treated as different documents.
        # Otherwise, this may break training.
        super().__init__(
            path=path,
            metadata={
                **(metadata or {}),
                "__version__": version,
                "__chunk_size__": chunk_size,
                "__stride__": stride,
            },
            strong_column="emphasis",
            on_disk=on_disk,
            save_extra_info=save_extra_info,
        )

    def process_data(
        self,
        path: str,
    ) -> pd.DataFrame:
        if not hasattr(self, "version") or self.version == "v1":
            return process_pdf(path)
        return sliding_pdf_parse.make_df(
            path,
            self.chunk_size,
            self.stride,
            self.emphasize_first_words,
            self.ignore_header_footer,
            self.ignore_nonstandard_orientation,
            self.doc_keywords,
            self.emphasize_section_titles,
            self.table_parsing,
        )

    @staticmethod
    def highlighted_doc(reference: Reference):
        old_highlights = pdf_parse.highlighted_doc(reference.source, reference.metadata)
        if old_highlights:
            return old_highlights
        return sliding_pdf_parse.highlighted_doc(reference.source, reference.metadata)


class DOCX(Extracted):
    def __init__(self, path: str, metadata=None, on_disk=False):
        super().__init__(path=path, metadata=metadata, on_disk=on_disk)

    def process_data(
        self,
        path: str,
    ) -> pd.DataFrame:
        return process_docx(path)


class Unstructured(Extracted):
    def __init__(
        self,
        path: Union[str, Path],
        save_extra_info: bool = True,
        metadata=None,
        on_disk=False,
    ):
        super().__init__(
            path=path,
            save_extra_info=save_extra_info,
            metadata=metadata,
            on_disk=on_disk,
        )

    def process_data(
        self,
        path: str,
    ) -> pd.DataFrame:
        if path.endswith(".pdf") or path.endswith(".docx"):
            raise NotImplementedError(
                "For PDF and DOCX FileTypes, use neuraldb.PDF and neuraldb.DOCX "
            )
        elif path.endswith(".pptx"):
            from .parsing_utils.unstructured_parse import PptxParse

            self.parser = PptxParse(path)

        elif path.endswith(".txt"):
            from .parsing_utils.unstructured_parse import TxtParse

            self.parser = TxtParse(path)

        elif path.endswith(".eml"):
            from .parsing_utils.unstructured_parse import EmlParse

            self.parser = EmlParse(path)

        else:
            raise Exception(f"File type is not yet supported")

        elements, success = self.parser.process_elements()

        if not success:
            raise ValueError(f"Could not read file: {path}")

        return self.parser.create_train_df(elements)


class URL(Document):
    """
    A URL document takes the data found at the provided URL (or in the provided reponse)
    and creates entities that can be inserted into NeuralDB.

    Args:
        url (str): The URL where the data is located.
        url_response (Reponse): Optional, defaults to None. If provided then the
            data in the response is used to create the entities, otherwise a get request
            is sent to the url.
        title_is_strong (bool): Optional, defaults to False. If true then the title is
            used as a strong signal for NeuralDB.
        metadata (Dict[str, Any]): Optional, defaults to {}. Specifies metadata to
            associate with entities from this file. Queries to NeuralDB can provide
            constrains to restrict results based on the metadata.
    """

    def __init__(
        self,
        url: str,
        url_response: Response = None,
        save_extra_info: bool = True,
        title_is_strong: bool = False,
        metadata=None,
        on_disk=False,
    ):
        self.url = url
        df = self.process_data(url, url_response)
        self.table = create_table(df, on_disk)
        self.hash_val = hash_string(url + str(metadata))
        self._save_extra_info = save_extra_info
        self._strong_column = "title" if title_is_strong else "text"
        self.doc_metadata = metadata_with_source(metadata or {}, url)

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
        return self.table.size

    @property
    def name(self) -> str:
        return self.url

    @property
    def source(self) -> str:
        return self.url

    @property
    def matched_constraints(self) -> Dict[str, ConstraintValue]:
        return {key: ConstraintValue(value) for key, value in self.doc_metadata.items()}

    def all_entity_ids(self) -> List[int]:
        return list(range(self.size))

    def strong_text(self, element_id: int) -> str:
        return self.table.field(
            row_id=element_id,
            column=self._strong_column if self._strong_column else "text",
        )

    def weak_text(self, element_id: int) -> str:
        return self.table.field(element_id, "text")

    def reference(self, element_id: int) -> Reference:
        if element_id >= self.table.size:
            _raise_unknown_doc_error(element_id)
        return Reference(
            document=self,
            element_id=element_id,
            text=self.table.field(element_id, "display"),
            source=self.source,
            metadata=(
                {"title": self.table.field(element_id, "title"), **self.doc_metadata}
                if "title" in self.table.columns
                else self.doc_metadata
            ),
        )

    def context(self, element_id, radius) -> str:
        if not 0 <= element_id or not element_id < self.size:
            raise ("Element id not in document.")
        rows = self.table.range_rows_as_dicts(
            from_row_id=max(0, element_id - radius),
            to_row_id=min(self.table.size, element_id + radius + 1),
        )
        return "\n".join(row["text"] for row in rows)

    def save_meta(self, directory: Path):
        self.table.save_meta(directory)

    def load_meta(self, directory: Path):
        if not hasattr(self, "doc_metadata"):
            self.doc_metadata = {}
        if hasattr(self, "df"):
            self.table = DataFrameTable(self.df)
            del self.df
        elif hasattr(self, "table"):
            self.table.load_meta(directory)


class DocumentConnector(Document):
    @property
    def hash(self) -> str:
        raise NotImplementedError()

    @property
    def meta_table(self) -> Optional[pd.DataFrame]:
        """
        It stores the mapping from id_in_document to meta_data of the document. It could be used to fetch the minimal document result if the connection is lost.
        """
        raise NotImplementedError()

    @property
    def meta_table_id_col(self) -> str:
        return "id_in_document"

    def _get_connector_object_name(self):
        raise NotImplementedError()

    def get_strong_columns(self):
        raise NotImplementedError()

    def get_weak_columns(self):
        raise NotImplementedError()

    def row_iterator(self):
        id_in_document = 0
        for current_chunk in self.chunk_iterator():
            for idx in range(len(current_chunk)):
                yield DocumentRow(
                    element_id=id_in_document,
                    strong=self.strong_text_from_chunk(
                        id_in_chunk=idx, chunk=current_chunk
                    ),  # Strong text from (idx)th row of the current_chunk
                    weak=self.weak_text_from_chunk(
                        id_in_chunk=idx, chunk=current_chunk
                    ),  # Weak text from (idx)th row of the current_chunk
                )
                id_in_document += 1

    def chunk_iterator(self) -> pd.DataFrame:
        raise NotImplementedError()

    def strong_text_from_chunk(self, id_in_chunk: int, chunk: pd.DataFrame) -> str:
        try:
            row = chunk.iloc[id_in_chunk]
            return " ".join([str(row[col]) for col in self.get_strong_columns()])
        except Exception as e:
            return ""

    def weak_text_from_chunk(self, id_in_chunk: int, chunk: pd.DataFrame) -> str:
        try:
            row = chunk.iloc[id_in_chunk]
            return " ".join([str(row[col]) for col in self.get_weak_columns()])
        except Exception as e:
            return ""

    def reference(self, element_id: int) -> Reference:
        raise NotImplementedError()

    def context(self, element_id, radius) -> str:
        if not 0 <= element_id or not element_id < self.size:
            raise ("Element id not in document.")

        reference_texts = [
            self.reference(i).text
            for i in range(
                max(0, element_id - radius), min(self.size, element_id + radius + 1)
            )
        ]
        return "\n".join(reference_texts)

    def save_meta(self, directory: Path):
        # Save the index table
        if self.save_extra_info and self.meta_table is not None:
            self.meta_table.to_csv(
                path_or_buf=directory / (self.name + ".csv"), index=False
            )

    def __getstate__(self):
        # Document Connectors are expected to remove their connector(s) object
        state = self.__dict__.copy()

        del state[self._get_connector_object_name()]

        return state


class SQLDatabase(DocumentConnector):
    """
    class for handling SQL database connections and data retrieval for training the neural_db model

    This class encapsulates functionality for connecting to an SQL database, executing SQL queries, and retrieving
    data for use in training the model.

    NOTE: It is being expected that the table will remain static in terms of both rows and columns.
    """

    def __init__(
        self,
        engine: sqlConn,
        table_name: str,
        id_col: str,
        strong_columns: Optional[List[str]] = None,
        weak_columns: Optional[List[str]] = None,
        reference_columns: Optional[List[str]] = None,
        chunk_size: int = 10_000,
        save_extra_info: bool = False,
        metadata: dict = {},
    ) -> None:
        self.table_name = table_name
        self.id_col = id_col
        self.strong_columns = strong_columns
        self.weak_columns = weak_columns
        self.reference_columns = reference_columns
        self.chunk_size = chunk_size
        self._save_extra_info = save_extra_info
        self.doc_metadata = metadata

        self._connector = SQLConnector(
            engine=engine,
            table_name=self.table_name,
            id_col=self.id_col,
            chunk_size=self.chunk_size,
        )
        self.total_rows = self._connector.total_rows()
        if not self.total_rows > 0:
            raise FileNotFoundError("Empty table")

        self.database_name = engine.url.database
        self.url = str(engine.url)
        self.engine_uq = self.url + f"/{self.table_name}"
        self._hash = hash_string(string=self.engine_uq)

        # Integrity checks
        self.assert_valid_id()
        self.assert_valid_columns()

        # setting the columns in the conector object
        self._connector.columns = list(set(self.strong_columns + self.weak_columns))

    @property
    def name(self):
        return self.database_name + "-" + self.table_name

    @property
    def source(self) -> str:
        return str(self.engine_uq)

    @property
    def hash(self):
        return self._hash

    @property
    def size(self) -> int:
        # It is verfied by the uniqueness assertion of the id column.
        return self.total_rows

    def setup_connection(self, engine: sqlConn):
        """
        This is a helper function to re-establish the connection upon loading the
        saved ndb model containing this SQLDatabase document.

        Args:
            engine: SQLAlchemy Connection object
                    NOTE: Provide the same connection object.

        NOTE: Same table would be used to establish connection
        """
        try:
            # The idea is to check for the connector object existence
            print(
                "Connector object already exists with url:"
                f" {self._connector.get_engine_url()}"
            )
        except AttributeError as e:
            assert engine.url.database == self.database_name
            assert str(engine.url) == self.url
            self._connector = SQLConnector(
                engine=engine,
                table_name=self.table_name,
                id_col=self.id_col,
                columns=list(set(self.strong_columns + self.weak_columns)),
                chunk_size=self.chunk_size,
            )

    def _get_connector_object_name(self):
        return "_connector"

    def get_strong_columns(self):
        return self.strong_columns

    def get_weak_columns(self):
        return self.weak_columns

    def get_engine(self):
        try:
            return self._connector._engine
        except AttributeError as e:
            raise AttributeError("engine is not available")

    @property
    def meta_table(self) -> Optional[pd.DataFrame]:
        return None

    def strong_text_from_chunk(self, id_in_chunk: int, chunk: pd.DataFrame) -> str:
        try:
            row = chunk.iloc[id_in_chunk]
            return " ".join([str(row[col]) for col in self.get_strong_columns()])
        except Exception as e:
            return ""

    def weak_text_from_chunk(self, id_in_chunk: int, chunk: pd.DataFrame) -> str:
        try:
            row = chunk.iloc[id_in_chunk]
            return " ".join([str(row[col]) for col in self.get_weak_columns()])
        except Exception as e:
            return ""

    def chunk_iterator(self) -> pd.DataFrame:
        return self._connector.chunk_iterator()

    def all_entity_ids(self) -> List[int]:
        return list(range(self.size))

    def reference(self, element_id: int) -> Reference:
        if element_id >= self.size:
            _raise_unknown_doc_error(element_id)

        try:
            reference_texts = self._connector.execute(
                query=(
                    f"SELECT {','.join(self.reference_columns)} FROM"
                    f" {self.table_name} WHERE {self.id_col} = {element_id}"
                )
            ).fetchone()

            text = "\n\n".join(
                [
                    f"{col_name}: {col_text}"
                    for col_name, col_text in zip(
                        self.reference_columns, reference_texts
                    )
                ]
            )

        except Exception as e:
            text = (
                f"Unable to connect to database, Referenced row with {self.id_col}:"
                f" {element_id} "
            )

        return Reference(
            document=self,
            element_id=element_id,
            text=text,
            source=self.source,
            metadata={
                "Database": self.database_name,
                "Table": self.table_name,
                **self.doc_metadata,
            },
        )

    @property
    def matched_constraints(self) -> Dict[str, ConstraintValue]:
        """
        This method is called when the document is being added to a DocumentManager in order to build an index for constrained search.
        """
        return {key: ConstraintValue(value) for key, value in self.doc_metadata.items()}

    def assert_valid_id(self):
        all_cols = self._connector.cols_metadata()

        id_col_meta = list(filter(lambda col: col["name"] == self.id_col, all_cols))
        if len(id_col_meta) == 0:
            raise AttributeError("id column not present in the table")
        elif not isinstance(id_col_meta[0]["type"], Integer):
            raise AttributeError("id column needs to be of type Integer")

        primary_keys = self._connector.get_primary_keys()
        if len(primary_keys) > 1:
            raise AttributeError("Composite primary key is not allowed")
        elif len(primary_keys) == 0 or primary_keys[0] != self.id_col:
            raise AttributeError(f"{self.id_col} needs to be a primary key")

        min_id = self._connector.execute(
            query=f"SELECT MIN({self.id_col}) FROM {self.table_name}"
        ).fetchone()[0]

        max_id = self._connector.execute(
            query=f"SELECT MAX({self.id_col}) FROM {self.table_name}"
        ).fetchone()[0]

        if min_id != 0 or max_id != self.size - 1:
            raise AttributeError(
                f"id column needs to be unique from 0 to {self.size - 1}"
            )

    def assert_valid_columns(self):
        all_cols = self._connector.cols_metadata()

        columns_set = set([col["name"] for col in all_cols])

        # Checking for strong, weak and reference columns (if provided) to be present in column list of the table
        if (self.strong_columns is not None) and (
            not set(self.strong_columns).issubset(columns_set)
        ):
            raise AttributeError(
                f"Strong column(s) doesn't exists in the table '{self.table_name}'"
            )
        if (self.weak_columns is not None) and (
            not set(self.weak_columns).issubset(columns_set)
        ):
            raise AttributeError(
                f"Weak column(s) doesn't exists in the table '{self.table_name}'"
            )
        if (self.reference_columns is not None) and (
            not set(self.reference_columns).issubset(columns_set)
        ):
            raise AttributeError(
                f"Reference column(s) doesn't exists in the table '{self.table_name}'"
            )

        # Checking for strong and weak column to have the correct column type
        for col in all_cols:
            if (
                self.strong_columns is not None
                and col["name"] in self.strong_columns
                and not isinstance(col["type"], String)
            ):
                raise AttributeError(
                    f"strong column '{col['name']}' needs to be of type String"
                )
            elif (
                self.weak_columns is not None
                and col["name"] in self.weak_columns
                and not isinstance(col["type"], String)
            ):
                raise AttributeError(
                    f"weak column '{col['name']}' needs to be of type String"
                )

        if self.strong_columns is None and self.weak_columns is None:
            self.strong_columns = []
            self.weak_columns = []
            for col in all_cols:
                if isinstance(col["type"], String):
                    self.weak_columns.append(col["name"])
        elif self.strong_columns is None:
            self.strong_columns = []
        elif self.weak_columns is None:
            self.weak_columns = []

        if self.reference_columns is None:
            self.reference_columns = list(columns_set)


class SharePoint(DocumentConnector):
    """
    Class for handling sharepoint connection, retrieving documents, processing and training the neural_db model

    Args:
        ctx (ClientContext): A ClientContext object for SharePoint connection.
        library_path (str): The server-relative directory path where documents
            are stored. Default: 'Shared Documents'
        chunk_size (int): The maximum amount of data (in bytes) that can be fetched
            at a time. (This limit may not apply if there are no files within this
            range.) Default: 10MB
        metadata (Dict[str, Any]): Optional, defaults to {}. Specifies metadata to
            associate with entities from this file. Queries to NeuralDB can provide
            constrains to restrict results based on the metadata.
    """

    def __init__(
        self,
        ctx: ClientContext,
        library_path: str = "Shared Documents",
        chunk_size: int = 10485760,
        save_extra_info: bool = False,
        metadata: dict = {},
    ) -> None:
        # Executing a dummy query to check for the authentication for the ctx object
        try:
            SharePoint.dummy_query(ctx=ctx)
        except Exception as e:
            raise Exception("Invalid ClientContext Object. Error: " + str(e))

        self._connector = SharePointConnector(
            ctx=ctx, library_path=library_path, chunk_size=chunk_size
        )
        self.library_path = library_path
        self.chunk_size = chunk_size
        self._save_extra_info = save_extra_info
        self.doc_metadata = metadata

        self.strong_column = "strong_text"
        self.weak_column = "weak_text"
        self.build_meta_table()
        self._name = (
            self._connector.site_name + "-" + (self.library_path).replace(" ", "_")
        )
        self.url = self._connector.url
        self._source = self.url + "/" + library_path
        self._hash = hash_string(self._source)

    @property
    def size(self) -> int:
        return len(self.meta_table)

    @property
    def name(self) -> str:
        return self._name

    @property
    def source(self) -> str:
        return self._source

    @property
    def hash(self) -> str:
        return self._hash

    def setup_connection(self, ctx: ClientContext):
        """
        This is a helper function to re-establish the connection upon loading the saved ndb model containing this Sharepoint document.

        Args:
            engine: SQLAlchemy Connection object. NOTE: Provide the same connection object.
        NOTE: Same library path would be used
        """
        try:
            # The idea is to check for the connector object existence
            print(f"Connector object already exists with url: {self._connector.url}")
        except AttributeError as e:
            assert self.url == ctx.web.get().execute_query().url
            self._connector = SharePointConnector(
                ctx=ctx, library_path=self.library_path, chunk_size=self.chunk_size
            )

    def get_strong_columns(self):
        return [self.strong_column]

    def get_weak_columns(self):
        return [self.weak_column]

    def build_meta_table(self):
        num_files = self._connector.num_files()

        print(f"Found {num_files} supported files")
        self._meta_table = pd.DataFrame(
            columns=[
                "internal_doc_id",
                "server_relative_url",
                "page",
            ]
        )
        self._meta_table = pd.concat(
            [
                current_chunk.drop(
                    columns=self.get_strong_columns() + self.get_weak_columns()
                )
                for current_chunk in self.chunk_iterator()
            ],
            ignore_index=True,
        )

        self._meta_table[self.meta_table_id_col] = range(len(self._meta_table))
        self._meta_table.set_index(keys=self.meta_table_id_col, inplace=True)

    @property
    def matched_constraints(self) -> Dict[str, ConstraintValue]:
        """
        Each constraint will get applied to each supported document on the sharepoint. This method is called when the document is being added to a DocumentManager in order to build an index for constrained search.
        """
        return {key: ConstraintValue(value) for key, value in self.doc_metadata.items()}

    def all_entity_ids(self) -> List[int]:
        return list(range(self.size))

    def reference(self, element_id: int) -> Reference:
        if element_id >= self.size:
            _raise_unknown_doc_error(element_id)

        filename = self.meta_table.iloc[element_id]["server_relative_url"].split(
            sep="/"
        )[-1]
        return Reference(
            document=self,
            element_id=element_id,
            text=f"filename: {filename}"
            + (
                f", page no: {self.meta_table.iloc[element_id]['page']}"
                if self.meta_table.iloc[element_id]["page"] is not None
                else ""
            ),
            source=self.source + "/" + filename,
            metadata={
                **self.meta_table.loc[element_id].to_dict(),
                **self.doc_metadata,
            },
        )

    @property
    def meta_table(self) -> Optional[pd.DataFrame]:
        return self._meta_table

    def chunk_iterator(self) -> pd.DataFrame:
        chunk_df = pd.DataFrame(
            columns=[
                self.strong_column,
                self.weak_column,
                "internal_doc_id",
                "server_relative_url",
                "page",
            ]
        )

        for file_dict in self._connector.chunk_iterator():
            chunk_df.drop(chunk_df.index, inplace=True)
            temp_dfs = []
            for server_relative_url, filepath in file_dict.items():
                if filepath.endswith(".pdf"):
                    doc = PDF(path=filepath, metadata=self.doc_metadata)
                elif filepath.endswith(".docx"):
                    doc = DOCX(path=filepath, metadata=self.doc_metadata)
                else:
                    doc = Unstructured(
                        path=filepath,
                        save_extra_info=self._save_extra_info,
                        metadata=self.doc_metadata,
                    )

                temp_df = pd.DataFrame(
                    columns=chunk_df.columns.tolist(), index=range(doc.size)
                )
                strong_text, weak_text, internal_doc_id, page = zip(
                    *[
                        (
                            doc.strong_text(i),
                            doc.weak_text(i),
                            i,
                            doc.reference(i).metadata.get("page", None),
                        )
                        for i in range(doc.size)
                    ]
                )
                temp_df[self.strong_column] = strong_text
                temp_df[self.weak_column] = weak_text
                temp_df["internal_doc_id"] = internal_doc_id
                temp_df["server_relative_url"] = [server_relative_url] * doc.size
                temp_df["page"] = page

                temp_dfs.append(temp_df)

            chunk_df = pd.concat(temp_dfs, ignore_index=True)
            yield chunk_df

    def _get_connector_object_name(self):
        return "_connector"

    @staticmethod
    def dummy_query(ctx: ClientContext):
        # Authenticatiion fails if this dummy query execution fails
        ctx.web.get().execute_query()

    @staticmethod
    def setup_clientContext(
        base_url: str, credentials: Dict[str, str]
    ) -> ClientContext:
        """
        Method to create a ClientContext object given base_url and credentials in the form (username, password) OR (client_id, client_secret)
        """
        ctx = None
        try:
            if all([cred in credentials.keys() for cred in ("username", "password")]):
                user_credentials = UserCredential(
                    user_name=credentials["username"], password=credentials["password"]
                )
                ctx = ClientContext(base_url=base_url).with_credentials(
                    user_credentials
                )
            SharePoint.dummy_query(ctx=ctx)
        except Exception as userCredError:
            try:
                if all(
                    [
                        cred in credentials.keys()
                        for cred in ("client_id", "client_secret")
                    ]
                ):
                    client_credentials = ClientCredential(
                        client_id=credentials["client_id"],
                        client_secret=credentials["client_secret"],
                    )
                    ctx = ClientContext(base_url=base_url).with_credentials(
                        client_credentials
                    )
                    SharePoint.dummy_query(ctx=ctx)
            except Exception as clientCredError:
                pass

        if ctx:
            return ctx
        raise AttributeError("Incorrect or insufficient credentials")


class SalesForce(DocumentConnector):
    """
    Class for handling the Salesforce object connections and data retrieval for
    training the neural_db model

    This class encapsulates functionality for connecting to an object, executing
    Salesforce Object Query Language (SOQL) queries, and retrieving

    NOTE: Allow the Bulk API access for the provided object. Also, it is being
    expected that the table will remain static in terms of both rows and columns.
    """

    def __init__(
        self,
        instance: Salesforce,
        object_name: str,
        id_col: str,
        strong_columns: Optional[List[str]] = None,
        weak_columns: Optional[List[str]] = None,
        reference_columns: Optional[List[str]] = None,
        save_extra_info: bool = True,
        metadata: dict = {},
    ) -> None:
        self.object_name = object_name
        self.id_col = id_col
        self.strong_columns = strong_columns
        self.weak_columns = weak_columns
        self.reference_columns = reference_columns
        self._save_extra_info = save_extra_info
        self.doc_metadata = metadata
        self._connector = SalesforceConnector(
            instance=instance, object_name=object_name
        )

        self.total_rows = self._connector.total_rows()
        if not self.total_rows > 0:
            raise FileNotFoundError("Empty Object")
        self._hash = hash_string(self._connector.sf_instance + self._connector.base_url)
        self._source = self._connector.sf_instance + self.object_name

        # Integrity_checks
        self.assert_valid_id()
        self.assert_valid_fields()

        # setting the columns in the connector object
        self._connector._fields = [self.id_col] + list(
            set(self.strong_columns + self.weak_columns)
        )

    @property
    def name(self) -> str:
        return self.object_name

    @property
    def source(self) -> str:
        return self._source

    @property
    def hash(self) -> str:
        return self._hash

    @property
    def size(self) -> int:
        return self.total_rows

    def setup_connection(self, instance: Salesforce):
        """
        This is a helper function to re-establish the connection upon loading a saved ndb model containing this SalesForce document.

        Args:
            instance: Salesforce instance. NOTE: Provide the same connection object.

        NOTE: Same object name would be used to establish connection
        """
        try:
            # The idea is to check for the connector object existence
            print(
                f"Connector object already exists with url: {self._connector.base_url}"
            )
        except AttributeError as e:
            assert self.hash == hash_string(instance.sf_instance + instance.base_url)
            self._connector = SalesforceConnector(
                instance=instance,
                object_name=self.object_name,
                fields=[self.id_col]
                + list(set(self.strong_columns + self.weak_columns)),
            )

    def _get_connector_object_name(self):
        return "_connector"

    def row_iterator(self):
        for current_chunk in self.chunk_iterator():
            for idx in range(len(current_chunk)):
                """
                * Since we are not able to retrieve the rows in sorted order, we have to do this so that (id, strong_text, weak_text) gets mapped correctly.
                * We cannot sort because the id_col needs to be of type 'autoNumber' which is a string. Neither we can do 'SELECT row FROM object_name ORDER BY LEN(id_col), id_col' because there is no LEN function in SOQL (by default). Owner of the object have to create a formula LEN() to use such query.
                """
                yield DocumentRow(
                    element_id=int(current_chunk.iloc[idx][self.id_col]),
                    strong=self.strong_text_from_chunk(
                        id_in_chunk=idx, chunk=current_chunk
                    ),  # Strong text from (idx)th row of the current_chunk
                    weak=self.weak_text_from_chunk(
                        id_in_chunk=idx, chunk=current_chunk
                    ),  # Weak text from (idx)th row of the current_chunk
                )

    def get_strong_columns(self):
        return self.strong_columns

    def get_weak_columns(self):
        return self.weak_columns

    @property
    def meta_table(self) -> Optional[pd.DataFrame]:
        return None

    def strong_text_from_chunk(self, id_in_chunk: int, chunk: pd.DataFrame) -> str:
        try:
            row = chunk.iloc[id_in_chunk]
            return " ".join([str(row[col]) for col in self.get_strong_columns()])
        except Exception as e:
            return ""

    def weak_text_from_chunk(self, id_in_chunk: int, chunk: pd.DataFrame) -> str:
        try:
            row = chunk.iloc[id_in_chunk]
            return " ".join([str(row[col]) for col in self.get_weak_columns()])
        except Exception as e:
            return ""

    def chunk_iterator(self) -> pd.DataFrame:
        return self._connector.chunk_iterator()

    def all_entity_ids(self) -> List[int]:
        return list(range(self.size))

    def reference(self, element_id: int) -> Reference:
        if element_id >= self.size:
            _raise_unknown_doc_error(element_id)

        try:
            result = self._connector.execute(
                query=(
                    f"SELECT {','.join(self.reference_columns)} FROM"
                    f" {self.object_name} WHERE {self.id_col} = '{element_id}'"
                )
            )["records"][0]
            del result["attributes"]
            text = "\n\n".join(
                [f"{col_name}: {col_text}" for col_name, col_text in result.items()]
            )

        except Exception as e:
            text = (
                "Unable to connect to the object instance, Referenced row with"
                f" {self.id_col}: {element_id} "
            )

        return Reference(
            document=self,
            element_id=element_id,
            text=text,
            source=self.source,
            metadata={
                "object_name": self.object_name,
                **self.doc_metadata,
            },
        )

    @property
    def matched_constraints(self) -> Dict[str, ConstraintValue]:
        """
        This method is called when the document is being added to a DocumentManager in order to build an index for constrained search.
        """
        return {key: ConstraintValue(value) for key, value in self.doc_metadata.items()}

    def assert_valid_id(self):
        all_fields = self._connector.field_metadata()

        all_field_name = [field["name"] for field in all_fields]

        if self.id_col not in all_field_name:
            raise AttributeError("Id Columns is not present in the object")

        # Uniqueness or primary constraint
        id_field_meta = list(
            filter(lambda field: field["name"] == self.id_col, all_fields)
        )
        if len(id_field_meta) == 0:
            raise AttributeError("id col not present in the object")
        id_field_meta = id_field_meta[0]

        """
        Reason behinds using AutoNumber as the id column type:

            1. Salesforce doesn't have typical table constraints. Object in a salesforce (or table in conventional sense) uses an alpha-numeric string as a primary key.
            2. Salesforce doesn't have a pure integer field. It have one in which we can set the decimal field of the double data-type to 0 but it is only for display purpose.
            3. Only option left is one Auto-number field that can be used but it limits some options.
        """
        if not id_field_meta["autoNumber"]:
            raise AttributeError("id col must be of type Auto-Number")
        else:
            # id field is auto-incremented string. Have to check for the form of A-{0}

            result = self._connector.execute(
                query=f"SELECT {self.id_col} FROM {self.object_name} LIMIT 1"
            )
            value: str = result["records"][0][self.id_col]
            if not value.isdigit():
                raise AttributeError("id column needs to be of the form \{0\}")

        expected_min_row_id = 0
        min_id = self._connector.execute(
            query=(
                f"SELECT {self.id_col} FROM {self.object_name} WHERE {self.id_col} ="
                f" '{expected_min_row_id}'"
            )
        )

        # This one is not required probably because user can't put the auto-number field mannually.
        # User just can provide the start of the auto-number so if the min_id is 0, then max_id should be size - 1
        expected_max_row_id = self.size - 1
        max_id = self._connector.execute(
            query=(
                f"SELECT {self.id_col} FROM {self.object_name} WHERE {self.id_col} ="
                f" '{expected_max_row_id}'"
            )
        )

        if not (min_id["totalSize"] == 1 and max_id["totalSize"] == 1):
            raise AttributeError(
                f"id column needs to be unique from 0 to {self.size - 1}"
            )

    def assert_valid_fields(
        self, supported_text_types: Tuple[str] = ("string", "textarea")
    ):
        all_fields = self._connector.field_metadata()
        self.assert_field_inclusion(all_fields)
        self.assert_field_type(all_fields, supported_text_types)
        self.default_fields(all_fields, supported_text_types)

    def assert_field_inclusion(self, all_fields: List[OrderedDict]):
        fields_set = set([field["name"] for field in all_fields])

        # Checking for strong, weak and reference columns (if provided) to be present in column list of the table
        column_name_error = (
            "Remember if it is a custom column, salesforce requires it to be appended"
            " with __c."
        )
        if (self.strong_columns is not None) and (
            not set(self.strong_columns).issubset(fields_set)
        ):
            raise AttributeError(
                f"Strong column(s) doesn't exists in the object. {column_name_error}"
            )
        if (self.weak_columns is not None) and (
            not set(self.weak_columns).issubset(fields_set)
        ):
            raise AttributeError(
                f"Weak column(s) doesn't exists in the object. {column_name_error}"
            )
        if (self.reference_columns is not None) and (
            not set(self.reference_columns).issubset(fields_set)
        ):
            raise AttributeError(
                f"Reference column(s) doesn't exists in the object. {column_name_error}"
            )

    def assert_field_type(
        self, all_fields: List[OrderedDict], supported_text_types: Tuple[str]
    ):
        # Checking for strong and weak column to have the correct column type
        for field in all_fields:
            if (
                self.strong_columns is not None
                and field["name"] in self.strong_columns
                and field["type"] not in supported_text_types
            ):
                raise AttributeError(
                    f"Strong column '{field['name']}' needs to be type from"
                    f" {supported_text_types}"
                )
            if (
                self.weak_columns is not None
                and field["name"] in self.weak_columns
                and field["type"] not in supported_text_types
            ):
                raise AttributeError(
                    f"Weak column '{field['name']}' needs to be type"
                    f" {supported_text_types}"
                )

    def default_fields(
        self, all_fields: List[OrderedDict], supported_text_types: Tuple[str]
    ):
        if self.strong_columns is None and self.weak_columns is None:
            self.strong_columns = []
            self.weak_columns = []
            for field in all_fields:
                if field["type"] in supported_text_types:
                    self.weak_columns.append(field["name"])
        elif self.strong_columns is None:
            self.strong_columns = []
        elif self.weak_columns is None:
            self.weak_columns = []

        if self.reference_columns is None:
            self.reference_columns = [self.id_col]
            for field in all_fields:
                if field["type"] in supported_text_types:
                    self.reference_columns.append(field["name"])


class SentenceLevelExtracted(Extracted):
    """Parses a document into sentences and creates a NeuralDB entry for each
    sentence. The strong column of the entry is the sentence itself while the
    weak column is the paragraph from which the sentence came. A NeuralDB
    reference produced by this object displays the paragraph instead of the
    sentence to increase recall.
    """

    def __init__(
        self, path: str, save_extra_info: bool = True, metadata=None, on_disk=False
    ):
        self.path = Path(path)
        self.hash_val = hash_file(
            path, metadata="sentence-level-extracted-" + str(metadata)
        )
        df = self.parse_sentences(self.process_data(path))
        self.table = create_table(df, on_disk)
        para_df = pd.DataFrame({"para": df["para"].unique()})
        self.para_table = create_table(para_df, on_disk)
        self._save_extra_info = save_extra_info
        self.doc_metadata = metadata_with_source(metadata or {}, Path(path).name)

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
                    "sentence_ids_in_para": str(get_ids(record)),
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
        return self.table.size

    def get_strong_columns(self):
        return ["sentence"]

    @property
    def name(self) -> str:
        return self.path.name if self.path else None

    @property
    def source(self) -> str:
        return str(self.path.absolute())

    def strong_text(self, element_id: int) -> str:
        return self.table.field(element_id, "sentence")

    def weak_text(self, element_id: int) -> str:
        return self.table.field(element_id, "para")

    def show_fn(text, source, **kwargs):
        return text

    def reference(self, element_id: int) -> Reference:
        if element_id >= self.table.size:
            _raise_unknown_doc_error(element_id)
        return Reference(
            document=self,
            element_id=element_id,
            text=self.table.field(element_id, "display"),
            source=self.source,
            metadata={**self.table.row_as_dict(element_id), **self.doc_metadata},
            upvote_ids=eval(self.table.field(element_id, "sentence_ids_in_para")),
        )

    def context(self, element_id, radius) -> str:
        if not 0 <= element_id or not element_id < self.size:
            raise ("Element id not in document.")

        # Cast to int because the actual return type is numpy.int64, which
        # causes problems in the self.para_table.range_rows_as_dicts line.
        para_id = int(self.table.field(element_id, "para_id"))

        rows = self.para_table.range_rows_as_dicts(
            from_row_id=max(0, para_id - radius),
            to_row_id=min(self.para_table.size, para_id + radius + 1),
        )
        return "\n\n".join(row["para"] for row in rows)

    def save_meta(self, directory: Path):
        # Let's copy the original file to the provided directory
        if self.save_extra_info:
            shutil.copy(self.path, directory)
        self.table.save_meta(directory)

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

        if hasattr(self, "df"):
            self.df["sentence_ids_in_para"] = self.df["sentence_ids_in_para"].apply(str)
            self.table = DataFrameTable(self.df)
            self.para_table = DataFrameTable(pd.DataFrame({"para": self.para_df}))
            del self.df
            del self.para_df
        elif hasattr(self, "table"):
            self.table.load_meta(directory)
            self.para_table.load_meta(directory)


class SentenceLevelPDF(SentenceLevelExtracted):
    """
    Parses a document into sentences and creates a NeuralDB entry for each
    sentence. The strong column of the entry is the sentence itself while the
    weak column is the paragraph from which the sentence came. A NeuralDB
    reference produced by this object displays the paragraph instead of the
    sentence to increase recall.

    Args:
        path (str): The path to the pdf file.
        metadata (Dict[str, Any]): Optional, defaults to {}. Specifies metadata to
            associate with entities from this file. Queries to NeuralDB can provide
            constrains to restrict results based on the metadata.
    """

    def __init__(self, path: str, metadata=None, on_disk=False):
        super().__init__(path=path, metadata=metadata, on_disk=on_disk)

    def process_data(
        self,
        path: str,
    ) -> pd.DataFrame:
        return process_pdf(path)


class SentenceLevelDOCX(SentenceLevelExtracted):
    """
    Parses a document into sentences and creates a NeuralDB entry for each
    sentence. The strong column of the entry is the sentence itself while the
    weak column is the paragraph from which the sentence came. A NeuralDB
    reference produced by this object displays the paragraph instead of the
    sentence to increase recall.

    Args:
        path (str): The path to the docx file.
        metadata (Dict[str, Any]): Optional, defaults to {}. Specifies metadata to
            associate with entities from this file. Queries to NeuralDB can provide
            constrains to restrict results based on the metadata.
    """

    def __init__(self, path: str, metadata=None, on_disk=False):
        super().__init__(path=path, metadata=metadata, on_disk=on_disk)

    def process_data(
        self,
        path: str,
    ) -> pd.DataFrame:
        return process_docx(path)


class InMemoryText(Document):
    """
    A wrapper around a batch of texts and their metadata to fit it in the
    NeuralDB Document framework.

    Args:
        name (str): A name for the batch of texts.
        texts (List[str]): A batch of texts.
        metadatas (List[Dict[str, Any]]): Optional. Metadata for each text.
        global_metadata (Dict[str, Any]): Optional. Metadata for the whole batch
        of texts.
    """

    def __init__(
        self,
        name: str,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        global_metadata=None,
        on_disk=False,
    ):
        self._name = name
        df = pd.DataFrame({"texts": texts})
        self.metadata_columns = []
        if metadatas:
            metadata_df = pd.DataFrame.from_records(metadatas)
            df = pd.concat([df, metadata_df], axis=1)
            self.metadata_columns = metadata_df.columns
        self.table = create_table(df, on_disk)
        self.hash_val = hash_string(str(texts) + str(metadatas))
        self.global_metadata = global_metadata or {}

    @property
    def hash(self) -> str:
        return self.hash_val

    @property
    def size(self) -> int:
        return self.table.size

    @property
    def name(self) -> str:
        return self._name

    @property
    def source(self) -> str:
        return self._name

    @property
    def matched_constraints(self) -> Dict[str, ConstraintValue]:
        metadata_constraints = {
            key: ConstraintValue(value) for key, value in self.global_metadata.items()
        }
        indexed_column_constraints = {
            key: ConstraintValue(is_any=True) for key in self.metadata_columns
        }
        return {**metadata_constraints, **indexed_column_constraints}

    def all_entity_ids(self) -> List[int]:
        return list(range(self.size))

    def filter_entity_ids(self, filters: Dict[str, Filter]):
        table_filter = TableFilter(
            {k: v for k, v in filters.items() if k not in self.global_metadata.keys()}
        )
        return self.table.apply_filter(table_filter)

    def strong_text(self, element_id: int) -> str:
        return ""

    def weak_text(self, element_id: int) -> str:
        return self.table.field(element_id, "texts")

    def reference(self, element_id: int) -> Reference:
        if element_id >= self.table.size:
            _raise_unknown_doc_error(element_id)
        return Reference(
            document=self,
            element_id=element_id,
            text=self.table.field(element_id, "texts"),
            source=self.source,
            metadata={**self.table.row_as_dict(element_id), **self.global_metadata},
        )

    def context(self, element_id, radius) -> str:
        # We don't return neighboring texts because they are not necessarily
        # related.
        return self.table.field(element_id, "texts")

    def save_meta(self, directory: Path):
        self.table.save_meta(directory)

    def load_meta(self, directory: Path):
        self.table.load_meta(directory)
