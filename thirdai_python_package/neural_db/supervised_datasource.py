import json
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from thirdai.dataset.data_source import PyDataSource

from .documents import DocumentManager

"""Sup and SupDataSource are classes that manage entity IDs for supervised
training.

Entity = an item that can be retrieved by NeuralDB. If we insert an ndb.CSV
object into NeuralDB, then each row of the CSV file is an entity. If we insert 
an ndb.PDF object, then each paragraph is an entity. If we insert an 
ndb.SentenceLevelDOCX object, then each sentence is an entity.

If this still doesn't make sense, consider a scenario where you insert a CSV 
file into NeuralDB and want to improve the performance of the database by
training it on supervised training samples. That is, you want the model to 
learn from (query, ID) pairs.

Since you only inserted one file, the ID of each entity in NeuralDB's index
is the same as the ID given in the file. Thus, the model can directly ingest
the (query, ID) pairs from your supervised dataset. However, this is not the 
case if you inserted multiple CSV files. For example, suppose you insert file A 
containing entities with IDs 0 through 100 and also insert file B containing 
its own set of entities with IDs 0 through 100. To disambiguate between entities
from different files, NeuralDB automatically offsets the IDs of the second file.
Consequently, you also have to adjust the labels of supervised training samples
corresponding to entities in file B. 

Instead of leaking the abstraction by making the user manually change the labels
of their dataset, we created Sup and SupDataSource to handle this.

If the user would rather use the database-assigned IDs instead of IDs from the 
original document, this can be done by passing uses_db_id = True to Sup(). This
is useful for cases where the user does not know the IDs of the entities in the
original documents. For example, if the original document is a PDF, then it is
NeuralDB that parses it into paragraphs; the user does not know the ID of each
paragraph beforehand. In this scenario, it is much easier for the user to just
use the database-assigned IDs.
"""


class Sup:
    """An object that contains supervised samples. This object is to be passed
    into NeuralDB.supervised_train().

    It can be initialized either with a CSV file, in which case it needs query
    and ID column names, or with sequences of queries and labels. It also needs
    to know which source object (i.e. which inserted CSV or PDF object) contains
    the relevant entities to the supervised samples.

    If uses_db_id is True, then the labels are assumed to use database-assigned
    IDs and will not be converted.
    """

    def __init__(
        self,
        csv: str = None,
        query_column: str = None,
        id_column: str = None,
        id_delimiter: str = None,
        queries: Sequence[str] = None,
        labels: Sequence[Sequence[int]] = None,
        source_id: str = "",
        uses_db_id: bool = False,
    ):
        if csv is not None and query_column is not None and id_column is not None:
            df = pd.read_csv(csv)
            self.queries = df[query_column].fillna("")
            self.labels = df[id_column]
            for i, label in enumerate(self.labels):
                if label == None or label == "":
                    raise ValueError(
                        "Got a supervised sample with an empty label, query:"
                        f" '{self.queries[i]}'"
                    )
            if id_delimiter:
                self.labels = self.labels.apply(
                    lambda label: list(
                        str(label).strip(id_delimiter).split(id_delimiter)
                    )
                )
            else:
                self.labels = self.labels.apply(lambda label: [str(label)])

        elif queries is not None and labels is not None:
            if len(queries) != len(labels):
                raise ValueError(
                    "Queries and labels sequences must be the same length."
                )
            self.queries = queries
            self.labels = labels
        else:
            raise ValueError(
                "Sup must be initialized with csv, query_column and id_column, or"
                " queries and labels."
            )
        self.source_id = source_id
        self.uses_db_id = uses_db_id

    @property
    def size(self):
        return len(self.queries)


class SupDataSource(PyDataSource):
    """Combines supervised samples from multiple Sup objects into a single data
    source. This allows NeuralDB's underlying model to train on all provided
    supervised datasets simultaneously.
    """

    def __init__(
        self,
        query_col: str,
        data: List[Sup],
        id_delimiter: Optional[str],
        doc_manager: Optional[DocumentManager] = None,
        id_column: Optional[str] = None,
    ):
        PyDataSource.__init__(self)
        self.query_col = query_col
        self.data = data
        self.id_delimiter = id_delimiter
        if not self.id_delimiter:
            print("WARNING: this model does not fully support multi-label datasets.")

        self.doc_manager = doc_manager

        if not self.doc_manager and not id_column:
            raise Exception(
                "Cannot initialize a SupDataSource with None values for both doc_manager and id_column"
            )

        self.id_column = id_column if id_column else doc_manager.id_column

        self.restart()

    def _csv_line(self, label: str, query: str):
        query = '"' + query.replace('"', '""') + '"'
        return f"{label},{query}"

    def _source_for_sup(self, sup: Sup):
        if not self.doc_manager:
            raise Exception(
                "Cannot get document ids for a SupDataSource with no document manager"
            )

        source_ids = self.doc_manager.match_source_id_by_prefix(sup.source_id)
        if len(source_ids) == 0:
            raise ValueError(f"Cannot find source with id {sup.source_id}")
        if len(source_ids) > 1:
            raise ValueError(f"Multiple sources match the prefix {sup.source_id}")
        return self.doc_manager.source_by_id(source_ids[0])

    def _labels(self, sup: Sup):
        if sup.uses_db_id:
            return [map(str, labels) for labels in sup.labels]

        doc, start_id = self._source_for_sup(sup)
        doc_id_map = doc.id_map()
        if doc_id_map:
            mapper = lambda label: str(doc_id_map[label] + start_id)
        else:
            mapper = lambda label: str(int(label) + start_id)

        return [map(mapper, labels) for labels in sup.labels]

    def _get_line_iterator(self, concat_labels=True):
        """
        If concat_labels is True and id_delimiter is not None, then the labels are joined using id_delimiter and yielded in a single row. Return one label per row in all other cases.

        This is done to enable data sharding for SupDataSource as currently we can only shard data sources that have a label per line without any delimiters.
        """
        # First yield the header
        yield self._csv_line(self.id_column, self.query_col)
        # Then yield rows
        for sup in self.data:
            for query, labels in zip(sup.queries, self._labels(sup)):
                if query == "":
                    continue
                if self.id_delimiter and concat_labels:
                    yield self._csv_line(
                        self.id_delimiter.join(labels),
                        query,
                    )
                else:
                    for label in labels:
                        yield self._csv_line(
                            label,
                            query,
                        )

    def indices(self):
        indices = set()
        for sup in self.data:
            for _, labels in zip(sup.queries, self._labels(sup)):
                for label in labels:
                    indices.add(label)

        return list(indices)

    def resource_name(self) -> str:
        return "Supervised training samples"

    @property
    def size(self):
        sizes_sup = [sup.size for sup in self.data]
        return sum(sizes_sup)

    def save(self, path: Path, save_interval=100_000):
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
                    "query_column": self.query_col,
                    "id_column": self.id_column,
                    "id_delimiter": self.id_delimiter,
                },
                f,
                indent=4,
            )

    @staticmethod
    def load(path: Path):
        with open(path / "arguments.json", "r") as f:
            args = json.load(f)

        sup_data = Sup(
            csv=path / "source.csv",
            query_column=args["query_column"],
            id_column=args["id_column"],
            id_delimiter=args["id_delimiter"],
            uses_db_id=True,
        )
        data_source = SupDataSource(
            query_col=args["query_column"],
            data=[sup_data],
            id_delimiter=args["id_delimiter"],
            id_column=args["id_column"],
        )
        return data_source
