import os
import shutil
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from thirdai import bolt, demos
from thirdai.neural_db import Document, Reference
from thirdai.neural_db import documents as docs

# We don't have a test on just the Document interface since it is just an
# interface.


class MockDocument(Document):
    def __init__(self, identifier: str, size: int) -> None:
        self._identifier = identifier
        self._size = size
        self._save_meta_called = 0
        self._save_meta_dir = None
        self._load_meta_called = 0
        self._load_meta_dir = None

    # We don't implement hash to test the default implementation

    @property
    def size(self) -> int:
        return self._size

    @property
    def name(self) -> str:
        return self._identifier

    # Expected strings have commas (delimiter) to test that the data source
    # converts it to proper CSV strings.
    def expected_strong_text_for_id(doc_id: str, element_id: int):
        return f"Strong text from {doc_id}, with id {element_id}"

    def expected_weak_text_for_id(doc_id: str, element_id: int):
        return f"Weak text from {doc_id}, with id {element_id}"

    def expected_reference_text_for_id(doc_id: str, element_id: int):
        return f"Reference text from {doc_id}, with id {element_id}"

    def expected_context_for_id_and_radius(doc_id: str, element_id: int, radius: int):
        return f"Context from {doc_id}, with id {element_id} and radius {radius}"

    def check_id(self, element_id: int):
        if element_id >= self._size:
            raise ValueError("Out of range")

    def strong_text(self, element_id: int) -> str:
        self.check_id(element_id)
        return MockDocument.expected_strong_text_for_id(self._identifier, element_id)

    def weak_text(self, element_id: int) -> str:
        self.check_id(element_id)
        return MockDocument.expected_weak_text_for_id(self._identifier, element_id)

    def reference(self, element_id: int) -> Reference:
        self.check_id(element_id)

        return Reference(
            document=self,
            element_id=element_id,
            text=MockDocument.expected_reference_text_for_id(
                self._identifier, element_id
            ),
            source=self._identifier,
            metadata={},
        )

    def context(self, element_id: int, radius) -> str:
        self.check_id(element_id)
        return MockDocument.expected_context_for_id_and_radius(
            self._identifier, element_id, radius
        )

    def save_meta(self, directory: Path):
        self._save_meta_called += 1
        self._save_meta_dir = directory

    def load_meta(self, directory: Path):
        self._load_meta_called += 1
        self._load_meta_dir = directory


id_column = "id"
strong_column = "strong"
weak_column = "weak"
first_id = "first"
second_id = "second"
first_size = 5
second_size = 10
first_doc = MockDocument(first_id, first_size)
second_doc = MockDocument(second_id, second_size)


def data_source_to_df(data_source):
    csv_string = ""
    for _ in range(data_source.size + 1):  # +1 for header
        csv_string += data_source.next_line() + "\n"

    return pd.read_csv(StringIO(csv_string))


def check_first_doc(df):
    for i in range(first_size):
        assert df.iloc[i][id_column] == i
        assert df.iloc[i][strong_column] == MockDocument.expected_strong_text_for_id(
            first_id, i
        )
        assert df.iloc[i][weak_column] == MockDocument.expected_weak_text_for_id(
            first_id, i
        )


def check_second_doc(df, position_offset, id_offset):
    for i in range(second_size):
        assert df.iloc[position_offset + i][id_column] == id_offset + i
        assert df.iloc[position_offset + i][
            strong_column
        ] == MockDocument.expected_strong_text_for_id(second_id, i)
        assert df.iloc[position_offset + i][
            weak_column
        ] == MockDocument.expected_weak_text_for_id(second_id, i)


@pytest.mark.unit
def test_document_data_source():
    data_source = docs.DocumentDataSource(id_column, strong_column, weak_column)

    data_source.add(first_doc, start_id=0)
    data_source.add(second_doc, start_id=first_size)

    assert data_source.size == first_size + second_size

    df = data_source_to_df(data_source)
    assert len(df) == (first_size + second_size)
    check_first_doc(df)
    check_second_doc(df, position_offset=first_size, id_offset=first_size)


@pytest.mark.unit
def test_document_manager_splits_intro_and_train():
    doc_manager = docs.DocumentManager(id_column, strong_column, weak_column)
    data_sources, _ = doc_manager.add([first_doc])

    assert data_sources.train.size == first_size
    assert data_sources.intro.size == first_size
    check_first_doc(data_source_to_df(data_sources.train))
    check_first_doc(data_source_to_df(data_sources.intro))

    data_sources, _ = doc_manager.add([first_doc, second_doc])

    assert data_sources.train.size == first_size + second_size
    assert data_sources.intro.size == second_size
    check_first_doc(data_source_to_df(data_sources.train))
    data_sources.train.restart()
    check_second_doc(
        data_source_to_df(data_sources.train),
        position_offset=first_size,
        id_offset=first_size,
    )
    check_second_doc(
        data_source_to_df(data_sources.intro), position_offset=0, id_offset=first_size
    )


@pytest.mark.unit
def test_document_manager_save_load():
    doc_manager = docs.DocumentManager(id_column, strong_column, weak_column)
    doc_manager.add([first_doc, second_doc])

    save_path = Path(os.getcwd()) / "neural_db_docs_test"
    os.mkdir(save_path)
    doc_manager.save_meta(save_path)
    doc_manager.load_meta(save_path)

    assert first_doc._save_meta_called == 1
    assert first_doc._load_meta_called == 1
    assert str(save_path) in str(first_doc._save_meta_dir)
    assert first_doc._save_meta_dir == first_doc._load_meta_dir

    assert second_doc._save_meta_called == 1
    assert second_doc._load_meta_called == 1
    assert str(save_path) in str(second_doc._save_meta_dir)
    assert second_doc._save_meta_dir == second_doc._load_meta_dir

    shutil.rmtree(save_path)


@pytest.mark.unit
def test_document_manager_sources():
    doc_manager = docs.DocumentManager(id_column, strong_column, weak_column)
    doc_manager.add([first_doc, second_doc])
    assert first_doc.hash in doc_manager.sources().keys()
    assert second_doc.hash in doc_manager.sources().keys()
    assert first_doc.name in doc_manager.sources().values()
    assert second_doc.name in doc_manager.sources().values()


@pytest.mark.unit
def test_document_manager_reference():
    doc_manager = docs.DocumentManager(id_column, strong_column, weak_column)
    doc_manager.add([first_doc, second_doc])

    reference_3 = doc_manager.reference(3)
    assert reference_3.id == 3
    assert reference_3.text == MockDocument.expected_reference_text_for_id(
        first_id, 3
    )
    assert reference_3.source == first_id

    reference_10 = doc_manager.reference(10)
    assert reference_10.id == 10
    assert reference_10.text == MockDocument.expected_reference_text_for_id(
        second_id, 10 - first_size
    )
    assert reference_10.source == second_id


@pytest.mark.unit
def test_document_manager_context():
    doc_manager = docs.DocumentManager(id_column, strong_column, weak_column)
    doc_manager.add([first_doc, second_doc])
    assert doc_manager.context(
        3, 10
    ) == MockDocument.expected_context_for_id_and_radius(
        first_id, element_id=3, radius=10
    )
    assert doc_manager.context(
        10, 3
    ) == MockDocument.expected_context_for_id_and_radius(
        second_id, element_id=10 - first_size, radius=3
    )


# @pytest.mark.unit
# def test_udt_cold_start_on_csv_document():
#     class CSV(Document):
#         def __init__(self, csv_path, id_column, strong_columns, weak_columns) -> None:
#             self.df = pd.read_csv(csv_path)
#             self.df = self.df.sort_values(id_column)
#             assert len(self.df[id_column].unique()) == len(self.df[id_column])
#             assert self.df[id_column].min() == 0
#             assert self.df[id_column].max() == len(self.df[id_column]) - 1

#             for col in strong_columns + weak_columns:
#                 self.df[col] = self.df[col].fillna("")

#             self.csv_path = csv_path
#             self.id_column = id_column
#             self.strong_columns = strong_columns
#             self.weak_columns = weak_columns

#         def size(self) -> int:
#             return len(self.df)

#         def strong_text(self, element_id: int) -> str:
#             return " ".join(
#                 [self.df.iloc[element_id][col] for col in self.strong_columns]
#             )

#         def weak_text(self, element_id: int) -> str:
#             return " ".join(
#                 [self.df.iloc[element_id][col] for col in self.weak_columns]
#             )

#         def name(self) -> str:
#             return self.csv_path

#     (
#         catalog_file,
#         n_target_classes,
#     ) = demos.download_amazon_kaggle_product_catalog_sampled()

#     model = bolt.UniversalDeepTransformer(
#         data_types={
#             "QUERY": bolt.types.text(),
#             "PRODUCT_ID": bolt.types.categorical(),
#         },
#         target="PRODUCT_ID",
#         n_target_classes=n_target_classes,
#         integer_target=True,
#     )

#     data_source = docs.DocumentDataSource("PRODUCT_ID", "STRONG", "WEAK")
#     data_source.add(
#         CSV(
#             catalog_file,
#             "PRODUCT_ID",
#             ["TITLE"],
#             ["DESCRIPTION", "BULLET_POINTS", "BRAND"],
#         ),
#         start_id=0,
#     )
#     metrics = model.cold_start_on_data_source(
#         data_source=data_source,
#         strong_column_names=["STRONG"],
#         weak_column_names=["WEAK"],
#         learning_rate=0.001,
#         epochs=5,
#         batch_size=2000,
#         metrics=["categorical_accuracy"],
#     )

#     os.remove(catalog_file)

#     assert metrics["train_categorical_accuracy"][-1] > 0.5
