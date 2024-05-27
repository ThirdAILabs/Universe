import os
import shutil
from collections import defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from ndb_utils import (
    CSV_FILE,
    DOCX_FILE,
    EML_FILE,
    PDF_FILE,
    PPTX_FILE,
    PRIAXOR_PDF_FILE,
    TXT_FILE,
    URL_LINK,
    create_simple_dataset,
    docs_with_meta,
)
from thirdai import bolt, demos, neural_db
from thirdai.neural_db import documents
from thirdai.neural_db.documents import DocumentDataSource
from thirdai.neural_db.sharded_documents import DataLoadMultiplexer, shard_data_source

# We don't have a test on just the Document interface since it is just an
# interface.


@pytest.fixture(scope="session")
def prepare_documents_test():
    from thirdai.neural_db import Document, Reference

    class MockDocument(Document):
        def __init__(self, identifier: str, size: int) -> None:
            self._identifier = identifier
            self._size = size
            self._save_meta_called = 0
            self._save_meta_dir = None
            self._load_meta_called = 0
            self._load_meta_dir = None

        @property
        def size(self) -> int:
            return self._size

        @property
        def name(self) -> str:
            return self._identifier

        @property
        def matched_constraints(self):
            return {}

        def all_entity_ids(self):
            return list(range(self.size))

        # Expected strings have commas (delimiter) and double quotes to test
        # that the data source converts it to proper CSV strings.
        def expected_strong_text_for_id(doc_id: str, element_id: int):
            return f'"Strong" text from {doc_id}, with id {element_id}, plus ""'

        def expected_weak_text_for_id(doc_id: str, element_id: int):
            return f'"Weak" text from {doc_id}, with id {element_id}, plus ""'

        def expected_reference_text_for_id(doc_id: str, element_id: int):
            return f'"Reference" text from {doc_id}, with id {element_id}, plus ""'

        def expected_context_for_id_and_radius(
            doc_id: str, element_id: int, radius: int
        ):
            return (
                f'"Context" from {doc_id}, with id {element_id} and radius {radius},'
                ' plus ""'
            )

        def check_id(self, element_id: int):
            if element_id >= self._size:
                raise ValueError("Out of range")

        def strong_text(self, element_id: int) -> str:
            self.check_id(element_id)
            return MockDocument.expected_strong_text_for_id(
                self._identifier, element_id
            )

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
            assert df.iloc[i][
                strong_column
            ] == MockDocument.expected_strong_text_for_id(first_id, i)
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

    return (
        MockDocument,
        id_column,
        strong_column,
        weak_column,
        first_id,
        second_id,
        first_size,
        second_size,
        first_doc,
        second_doc,
        data_source_to_df,
        check_first_doc,
        check_second_doc,
    )


@pytest.mark.unit
def test_document_data_source(prepare_documents_test):
    (
        _,
        id_column,
        strong_column,
        weak_column,
        _,
        _,
        first_size,
        second_size,
        first_doc,
        second_doc,
        data_source_to_df,
        check_first_doc,
        check_second_doc,
    ) = prepare_documents_test

    data_source = documents.DocumentDataSource(id_column, strong_column, weak_column)

    data_source.add(first_doc, start_id=0)
    data_source.add(second_doc, start_id=first_size)

    assert data_source.size == first_size + second_size

    df = data_source_to_df(data_source)
    assert len(df) == (first_size + second_size)
    check_first_doc(df)
    check_second_doc(df, position_offset=first_size, id_offset=first_size)


@pytest.mark.unit
@pytest.mark.parametrize("number_shards", [1, 3])
def test_sharded_data_source(prepare_documents_test, number_shards):
    (
        _,
        id_column,
        strong_column,
        weak_column,
        _,
        _,
        first_size,
        second_size,
        first_doc,
        second_doc,
        data_source_to_df,
        check_first_doc,
        check_second_doc,
    ) = prepare_documents_test

    data_source = documents.DocumentDataSource(id_column, strong_column, weak_column)

    data_source.add(first_doc, start_id=0)
    data_source.add(second_doc, start_id=first_size)

    assert data_source.size == first_size + second_size

    label_to_segment_map = defaultdict(list)
    number_shards = number_shards
    sharded_data_sources = shard_data_source(
        data_source=data_source,
        number_shards=number_shards,
        label_to_segment_map=label_to_segment_map,
        update_segment_map=True,
    )
    assert len(sharded_data_sources) == number_shards
    assert sum(shard.size for shard in sharded_data_sources) == first_size + second_size
    assert all(shard.size > 0 for shard in sharded_data_sources)
    assert set(value[0] for value in label_to_segment_map.values()) == set(
        range(number_shards)
    )
    assert sorted(data_source.indices()) == sorted(label_to_segment_map.keys())

    df = pd.concat([data_source_to_df(shard) for shard in sharded_data_sources])
    df = df.sort_values(id_column)

    check_first_doc(df)
    check_second_doc(df, position_offset=first_size, id_offset=first_size)


@pytest.mark.unit
def test_document_manager_splits_intro_and_train(prepare_documents_test):
    (
        _,
        id_column,
        strong_column,
        weak_column,
        _,
        _,
        first_size,
        second_size,
        first_doc,
        second_doc,
        data_source_to_df,
        check_first_doc,
        check_second_doc,
    ) = prepare_documents_test

    doc_manager = documents.DocumentManager(id_column, strong_column, weak_column)
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
def test_document_manager_save_load(prepare_documents_test):
    (
        _,
        id_column,
        strong_column,
        weak_column,
        _,
        _,
        _,
        _,
        first_doc,
        second_doc,
        _,
        _,
        _,
    ) = prepare_documents_test

    doc_manager = documents.DocumentManager(id_column, strong_column, weak_column)
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
def test_document_manager_sources(prepare_documents_test):
    (
        _,
        id_column,
        strong_column,
        weak_column,
        _,
        _,
        _,
        _,
        first_doc,
        second_doc,
        _,
        _,
        _,
    ) = prepare_documents_test

    doc_manager = documents.DocumentManager(id_column, strong_column, weak_column)
    doc_manager.add([first_doc, second_doc])
    assert first_doc.hash in doc_manager.sources().keys()
    assert second_doc.hash in doc_manager.sources().keys()
    assert first_doc.name in [doc.name for doc in doc_manager.sources().values()]
    assert second_doc.name in [doc.name for doc in doc_manager.sources().values()]


@pytest.mark.unit
def test_document_manager_reference(prepare_documents_test):
    (
        MockDocument,
        id_column,
        strong_column,
        weak_column,
        first_id,
        second_id,
        first_size,
        _,
        first_doc,
        second_doc,
        _,
        _,
        _,
    ) = prepare_documents_test

    doc_manager = documents.DocumentManager(id_column, strong_column, weak_column)
    doc_manager.add([first_doc, second_doc])

    reference_3 = doc_manager.reference(3)
    assert reference_3.id == 3
    assert reference_3.upvote_ids[0] == 3
    assert reference_3.text == MockDocument.expected_reference_text_for_id(first_id, 3)
    assert reference_3.source == first_id

    reference_10 = doc_manager.reference(10)
    assert reference_10.id == 10
    assert reference_10.upvote_ids[0] == 10
    assert reference_10.text == MockDocument.expected_reference_text_for_id(
        second_id, 10 - first_size
    )
    assert reference_10.source == second_id


@pytest.mark.unit
def test_document_manager_context(prepare_documents_test):
    (
        MockDocument,
        id_column,
        strong_column,
        weak_column,
        first_id,
        second_id,
        first_size,
        _,
        first_doc,
        second_doc,
        _,
        _,
        _,
    ) = prepare_documents_test

    doc_manager = documents.DocumentManager(id_column, strong_column, weak_column)
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


@pytest.mark.unit
def test_udt_cold_start_on_csv_document():
    from thirdai.neural_db import CSV

    (
        catalog_file,
        n_target_classes,
    ) = demos.download_amazon_kaggle_product_catalog_sampled()

    model = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(),
            "PRODUCT_ID": bolt.types.categorical(),
        },
        target="PRODUCT_ID",
        n_target_classes=n_target_classes,
        integer_target=True,
    )

    data_source = documents.DocumentDataSource("PRODUCT_ID", "STRONG", "WEAK")
    data_source.add(
        CSV(
            catalog_file,
            "PRODUCT_ID",
            ["TITLE"],
            ["DESCRIPTION", "BULLET_POINTS", "BRAND"],
            ["TITLE"],
        ),
        start_id=0,
    )
    metrics = model.cold_start_on_data_source(
        data_source=data_source,
        strong_column_names=["STRONG"],
        weak_column_names=["WEAK"],
        learning_rate=0.001,
        epochs=5,
        batch_size=2000,
        metrics=["categorical_accuracy"],
    )

    os.remove(catalog_file)

    assert metrics["train_categorical_accuracy"][-1] > 0.5


@pytest.mark.unit
def test_csv_doc_autotuning():
    filename = "simple.csv"
    with open(filename, "w") as file:
        file.write(
            f"""col1,col2,col3,col4,col5\n
            lorem,2,3.0,How vexingly quick daft zebras jump!,2021-02-01\n
            ipsum,5,6,"Sphinx of black quartz, judge my vow.",2022-02-01\n
            dolor,8,9,The quick brown fox jumps over the lazy dog,2023-02-01\n
            """
        )

    doc = neural_db.CSV(filename)

    assert doc.id_column == "thirdai_index"
    assert doc.strong_columns == []
    assert doc.weak_columns == ["col4"]
    assert doc.reference_columns == ["col1", "col2", "col3", "col4", "col5"]

    assert "lorem" in doc.reference(0).text

    os.remove(filename)


@pytest.mark.unit
def test_data_load_multiplexer(create_simple_dataset):
    filename = create_simple_dataset

    doc = neural_db.CSV(
        filename, id_column="label", strong_columns=["text"], weak_columns=["text"]
    )
    document_data_source = DocumentDataSource(
        id_column="label", strong_column="strong", weak_column="weak"
    )
    document_data_source.add(doc, start_id=0)
    data_multiplexer = DataLoadMultiplexer(num_segments=1)
    (
        segment_filenames,
        segment_objects,
        _,
    ) = data_multiplexer.create_segments_with_data_source(
        document_data_source, defaultdict(list), update_index=True
    )
    for i, filename in enumerate(segment_filenames):
        try:
            df = pd.read_csv(filename)
            if "label" not in df.columns:
                assert False, f"column label not found!"
            if not df["label"].apply(lambda x: str(x).isdigit()).all():
                assert False, f"Some values in column 'label' are not integers"

        except Exception as e:
            assert False, f"Error reading file: {e}"
        segment_objects[i].close()


@pytest.mark.unit
def test_document_has_source_metadata():
    manager = neural_db.documents.DocumentManager(
        id_column="id", strong_column="strong", weak_column="weak"
    )
    manager.add(docs_with_meta())

    def count_matches(source):
        return len(
            manager.constraint_matcher.match({"source": neural_db.EqualTo(source)})
        )

    assert count_matches(Path(CSV_FILE).name) == 1
    assert count_matches(Path(PDF_FILE).name) == 2
    assert count_matches(Path(DOCX_FILE).name) == 2
    assert count_matches(URL_LINK) == 1
    assert count_matches(Path(PPTX_FILE).name) == 1
    assert count_matches(Path(TXT_FILE).name) == 1
    assert count_matches(Path(EML_FILE).name) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "doc_factory",
    [
        lambda: neural_db.CSV(CSV_FILE, metadata={"source": "something"}),
        lambda: neural_db.PDF(PDF_FILE, metadata={"source": "something"}),
        lambda: neural_db.DOCX(DOCX_FILE, metadata={"source": "something"}),
        lambda: neural_db.URL(URL_LINK, metadata={"source": "something"}),
        lambda: neural_db.Unstructured(PPTX_FILE, metadata={"source": "something"}),
        lambda: neural_db.Unstructured(TXT_FILE, metadata={"source": "something"}),
        lambda: neural_db.Unstructured(EML_FILE, metadata={"source": "something"}),
        lambda: neural_db.SentenceLevelPDF(PDF_FILE, metadata={"source": "something"}),
        lambda: neural_db.SentenceLevelDOCX(
            DOCX_FILE, metadata={"source": "something"}
        ),
    ],
)
def test_document_throws_when_user_passes_source_metadata(doc_factory):
    with pytest.raises(
        ValueError,
        match=r"Document metadata cannot contain the key 'source'. 'source' is a reserved key.",
    ):
        doc_factory()


@pytest.mark.unit
def test_pdf_section_titles():
    pdf = neural_db.PDF(PRIAXOR_PDF_FILE, version="v2", emphasize_section_titles=True)

    def find_strong_words(pdf, strong_words):
        found_words = False
        for _, row in pdf.table.df.iterrows():
            if strong_words in row["emphasis"]:
                found_words = True
        assert found_words

    find_strong_words(pdf, "Restrictions and Limitations")
    find_strong_words(pdf, "Groundwater Advisory")
    find_strong_words(pdf, "Information on Droplet Size")
    find_strong_words(pdf, "AGRICULTURAL USE REQUIREMENTS")
    find_strong_words(pdf, "Environmental Hazards")


@pytest.mark.unit
def test_pdf_keywords():
    pdf = neural_db.PDF(PRIAXOR_PDF_FILE, version="v2", doc_keywords="ThirdAI BOLT UDT")

    for _, row in pdf.table.df.iterrows():
        assert "ThirdAI BOLT UDT" in row["emphasis"]
