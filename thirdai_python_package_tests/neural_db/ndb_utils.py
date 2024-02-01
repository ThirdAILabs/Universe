import os
import shutil
from typing import List

import pandas as pd
import pytest
import requests
from thirdai import neural_db as ndb

from thirdai_python_package_tests.neural_db.base_connectors import base


class Equivalent_doc:
    def __init__(self, connector_doc, local_doc) -> None:
        self.connector_doc = connector_doc
        self.local_doc = local_doc


@pytest.fixture
def create_simple_dataset():
    filename = "simple.csv"
    with open(filename, "w") as file:
        file.writelines(
            [
                "text,label\n",
                "apples are green,0\n",
                "spinach is green,1\n",
                "bananas are yellow,2\n",
                "oranges are orange,3\n",
                "apples are red,4\n",
            ]
        )

    yield filename

    os.remove(filename)


@pytest.fixture
def train_simple_neural_db(create_simple_dataset):
    filename = create_simple_dataset
    db = ndb.NeuralDB()

    doc = ndb.CSV(
        filename,
        id_column="label",
        strong_columns=["text"],
        weak_columns=["text"],
        reference_columns=["label", "text"],
    )

    db.insert(sources=[doc], train=True)

    return db


BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "document_test_data"
)
CSV_FILE = os.path.join(BASE_DIR, "lorem_ipsum.csv")
URL_LINK = "https://en.wikipedia.org/wiki/Rice_University"
PDF_FILE = os.path.join(BASE_DIR, "mutual_nda.pdf")
DOCX_FILE = os.path.join(BASE_DIR, "four_english_words.docx")
PPTX_FILE = os.path.join(BASE_DIR, "quantum_mechanics.pptx")
TXT_FILE = os.path.join(BASE_DIR, "nature.txt")
EML_FILE = os.path.join(BASE_DIR, "Message.eml")


CSV_EXPLICIT_META = "csv-explicit"
PDF_META = "pdf"
DOCX_META = "docx"
URL_NO_RESPONSE_META = "url-no-response"
PPTX_META = "pptx"
TXT_META = "txt"
EML_META = "eml"
SENTENCE_PDF_META = "sentence-pdf"
SENTENCE_DOCX_META = "sentence-docx"


def meta(file_meta):
    return {"meta": file_meta}


def build_local_sharepoint_doc():
    from thirdai.neural_db.utils import DIRECTORY_CONNECTOR_SUPPORTED_EXT

    dir = os.path.join(BASE_DIR, "connector_docs", "SharePoint")
    files = os.listdir(dir)
    doc_files = []
    for file_name in files:
        file_path = os.path.join(dir, file_name)
        if (
            os.path.isfile(file_path)
            and file_name.split(sep=".")[-1] in DIRECTORY_CONNECTOR_SUPPORTED_EXT
        ):
            doc_files.append(file_path)
    doc_files = sorted(doc_files, key=lambda file_path: file_path.split(sep="/")[-1])

    ndb_docs = []
    for file_path in doc_files:
        file_name = file_path.split(sep="/")[-1]
        if file_name.endswith(".pdf"):
            temp = ndb.PDF(path=file_path)
        elif file_name.endswith(".docx"):
            temp = ndb.DOCX(path=file_path)
        else:
            temp = ndb.Unstructured(path=file_path)

        ndb_docs.append(temp)

    class CombinedDocument(ndb.Document):
        def __init__(self, ndb_docs) -> None:
            cols = [self.strong_column, self.weak_column]

            tmp_dfs = []
            for current_ndb_doc in ndb_docs:
                temp_df = pd.DataFrame(columns=cols, index=range(current_ndb_doc.size))
                temp_df["id"] = range(current_ndb_doc.size)
                temp_df[self.strong_column] = temp_df["id"].apply(
                    lambda i: current_ndb_doc.strong_text(i)
                )
                temp_df[self.weak_column] = temp_df["id"].apply(
                    lambda i: current_ndb_doc.weak_text(i)
                )
                temp_df.drop(columns=["id"], inplace=True)
                tmp_dfs.append(temp_df)

            self.df = pd.concat(tmp_dfs, ignore_index=True)

        @property
        def strong_column(self):
            return "strong_text"

        @property
        def weak_column(self):
            return "weak_text"

        @property
        def name(self):
            return "CombinedLocalSharePointDocument"

        @property
        def size(self):
            return len(self.df)

        def strong_text(self, element_id: int) -> str:
            return self.df.iloc[element_id][self.strong_column]

        def weak_text(self, element_id: int) -> str:
            return self.df.iloc[element_id][self.weak_column]

    return CombinedDocument(ndb_docs=ndb_docs)


# This is a list of getter functions that return connector_doc objects so each test can
# use fresh doc object instances.

all_connector_doc_getters = [
    Equivalent_doc(
        connector_doc=lambda: ndb.SQLDatabase(
            engine=base.get_sql_engine(),
            table_name=base.get_sql_table(),
            id_col="id",
            strong_columns=["content"],
            weak_columns=["content"],
            reference_columns=["content"],
            chunk_size=3,
        ),
        local_doc=lambda: ndb.CSV(
            path=os.path.join(BASE_DIR, "connector_docs", "SQL", "Amazon_polarity.csv"),
            id_column="id",
            strong_columns=["content"],
            weak_columns=["content"],
            reference_columns=["content"],
        ),
    ),
    Equivalent_doc(
        connector_doc=lambda: ndb.SharePoint(
            ctx=base.get_client_context(), library_path=base.get_library_path()
        ),
        local_doc=build_local_sharepoint_doc,
    ),
    Equivalent_doc(
        connector_doc=lambda: ndb.SalesForce(
            instance=base.get_salesforce_instance(),
            object_name=base.get_salesforce_object_name(),
            id_col="ID__c",
            strong_columns=["Review__c"],
            weak_columns=["Review__c"],
            reference_columns=["Review__c"],
        ),
        local_doc=lambda: ndb.CSV(
            path=os.path.join(BASE_DIR, "connector_docs", "Salesforce", "yelp.csv"),
            id_column="ID__c",
            strong_columns=["Review__c"],
            weak_columns=["Review__c"],
            reference_columns=["Review__c"],
        ),
    ),
]

# This is a list of getter functions that return doc objects so each test can
# use fresh doc object instances.
all_local_doc_getters = [
    lambda: ndb.CSV(
        CSV_FILE,
        id_column="category",
        strong_columns=["text"],
        weak_columns=["text"],
        reference_columns=["text"],
    ),
    lambda: ndb.CSV(CSV_FILE),
    lambda: ndb.PDF(PDF_FILE),
    lambda: ndb.PDF(PDF_FILE, version="v2"),
    lambda: ndb.DOCX(DOCX_FILE),
    lambda: ndb.URL(URL_LINK),
    lambda: ndb.URL(URL_LINK, requests.get(URL_LINK)),
    lambda: ndb.Unstructured(PPTX_FILE),
    lambda: ndb.Unstructured(TXT_FILE),
    lambda: ndb.Unstructured(EML_FILE),
    lambda: ndb.SentenceLevelPDF(PDF_FILE),
    lambda: ndb.SentenceLevelDOCX(DOCX_FILE),
]

# The two URL docs are different constructor invocations for the same thing.
num_duplicate_local_doc_getters = 1


def on_diskable_doc_getters(on_disk):
    return [
        # Test both CSV constructors to make sure we capture all edge cases
        # relating to how we process the id column.
        lambda: ndb.CSV(
            CSV_FILE,
            id_column="category",
            strong_columns=["text"],
            weak_columns=["text"],
            reference_columns=["text"],
            on_disk=on_disk,
        ),
        lambda: ndb.CSV(CSV_FILE, on_disk=on_disk),
        # For everything else, only test one constructor per document type.
        lambda: ndb.PDF(PDF_FILE, on_disk=on_disk),
        lambda: ndb.DOCX(DOCX_FILE, on_disk=on_disk),
        lambda: ndb.URL(URL_LINK, on_disk=on_disk),
        lambda: ndb.Unstructured(PPTX_FILE, on_disk=on_disk),
        lambda: ndb.SentenceLevelPDF(PDF_FILE, on_disk=on_disk),
        lambda: ndb.SentenceLevelDOCX(DOCX_FILE, on_disk=on_disk),
    ]


num_duplicate_on_diskable_doc_getters = 0


all_doc_getters = all_local_doc_getters + [
    eq_doc.connector_doc for eq_doc in all_connector_doc_getters
]


def docs_with_meta():
    return [
        ndb.CSV(
            CSV_FILE,
            id_column="category",
            strong_columns=["text"],
            weak_columns=["text"],
            reference_columns=["text"],
            metadata=meta(CSV_EXPLICIT_META),
        ),
        ndb.PDF(PDF_FILE, metadata=meta(PDF_META)),
        ndb.DOCX(DOCX_FILE, metadata=meta(DOCX_META)),
        ndb.URL(URL_LINK, metadata=meta(URL_NO_RESPONSE_META)),
        ndb.Unstructured(PPTX_FILE, metadata=meta(PPTX_META)),
        ndb.Unstructured(TXT_FILE, metadata=meta(TXT_META)),
        ndb.Unstructured(EML_FILE, metadata=meta(EML_META)),
        ndb.SentenceLevelPDF(PDF_FILE, metadata=meta(SENTENCE_PDF_META)),
        ndb.SentenceLevelDOCX(DOCX_FILE, metadata=meta(SENTENCE_DOCX_META)),
    ]


metadata_constraints = [
    CSV_EXPLICIT_META,
    PDF_META,
    DOCX_META,
    URL_NO_RESPONSE_META,
    PPTX_META,
    TXT_META,
    EML_META,
    SENTENCE_PDF_META,
    SENTENCE_DOCX_META,
]


def search_works(db: ndb.NeuralDB, docs: List[ndb.Document], assert_acc: bool):
    top_k = 5
    correct_result = 0
    correct_source = 0
    for doc in docs:
        if isinstance(doc, ndb.SharePoint):
            continue
        source = doc.reference(0).source
        for elem_id in range(doc.size):
            query = doc.reference(elem_id).text
            results = db.search(query, top_k)

            assert len(results) >= 1
            assert len(results) <= top_k

            for result in results:
                assert type(result.text) == str
                assert len(result.text) > 0

            correct_result += int(query in [r.text for r in results])
            correct_source += int(source in [r.source for r in results])

            batch_results = db.search_batch(
                [query, query, "SOME TOTAL RANDOM QUERY"], top_k
            )

            assert len(batch_results) == 3
            assert batch_results[0] == results
            assert batch_results[0] == batch_results[1]
            assert batch_results[0] != batch_results[2]

    assert correct_source / sum([doc.size for doc in docs]) > 0.8
    if assert_acc:
        assert correct_result / sum([doc.size for doc in docs]) > 0.8


def get_upvote_target_id(db: ndb.NeuralDB, query: str, top_k: int):
    initial_ids = [r.id for r in db.search(query, top_k)]
    target_id = 0
    while target_id in initial_ids:
        target_id += 1
    return target_id


ARBITRARY_QUERY = "This is an arbitrary search query"


# Some of the following helper functions depend on others being called before them.
# It is best to call them in the order that these helper functions are written.
# They are only written as separate functions to make it easier to read.


def insert_works(db: ndb.NeuralDB, docs: List[ndb.Document], num_duplicate_docs):
    db.insert(docs, train=False)
    assert len(db.sources()) == len(docs) - num_duplicate_docs

    initial_scores = [r.score for r in db.search(ARBITRARY_QUERY, top_k=5)]

    db.insert(docs, train=True)
    assert len(db.sources()) == len(docs) - num_duplicate_docs

    assert [r.score for r in db.search(ARBITRARY_QUERY, top_k=5)] != initial_scores

    db.insert(docs, train=True, batch_size=1, learning_rate=0.0002)
    assert len(db.sources()) == len(docs) - num_duplicate_docs

    assert [r.score for r in db.search(ARBITRARY_QUERY, top_k=5)] != initial_scores


def upvote_works(db: ndb.NeuralDB):
    # We have more than 10 indexed entities.
    target_id = get_upvote_target_id(db, ARBITRARY_QUERY, top_k=10)

    number_models = (
        db._savable_state.model.number_models
        if hasattr(db._savable_state.model, "number_models")
        else 1
    )

    # TODO(Shubh) : For mach mixture, it is not necessary that upvoting alone will
    # boost the label enough to be predicted at once. Look at a better solution than
    # upvoting multiple times.
    times_to_upvote = 3 if number_models > 1 else 5
    for i in range(times_to_upvote):
        db.text_to_result(ARBITRARY_QUERY, target_id)
    assert target_id in [r.id for r in db.search(ARBITRARY_QUERY, top_k=10)]


def upvote_batch_works(db: ndb.NeuralDB):
    queries = [
        "This query is not related to any document.",
        "Neither is this one.",
        "Wanna get some biryani so we won't have to cook dinner?",
    ]
    target_ids = [get_upvote_target_id(db, query, top_k=10) for query in queries]
    db.text_to_result_batch(list(zip(queries, target_ids)))
    for query, target_id in zip(queries, target_ids):
        assert target_id in [r.id for r in db.search(query, top_k=10)]


def associate_works(db: ndb.NeuralDB):
    # Since this is still unstable, we only check that associate() updates the
    # model in *some* way, but we don't want to make stronger assertions as it
    # would make the test flaky.
    search_results = db.search(ARBITRARY_QUERY, top_k=5)
    initial_scores = [r.score for r in search_results]
    initial_ids = [r.id for r in search_results]

    another_arbitrary_query = "Eating makes me sleepy"
    db.associate(ARBITRARY_QUERY, another_arbitrary_query)

    new_search_results = db.search(ARBITRARY_QUERY, top_k=5)
    new_scores = [r.score for r in new_search_results]
    new_ids = [r.id for r in new_search_results]

    assert (initial_scores != new_scores) or (initial_ids != new_ids)


def save_load_works(db: ndb.NeuralDB):
    search_results = [r.text for r in db.search(ARBITRARY_QUERY, top_k=5)]

    if os.path.exists("temp.ndb"):
        shutil.rmtree("temp.ndb")
    db.save("temp.ndb")

    # Change working directory to catch edge cases. E.g. if we don't properly
    # save a sqlite database, this test may still pass if the original sqlite
    # database is still in the current working directory.
    if os.path.exists("new_dir"):
        shutil.rmtree("new_dir")
    os.mkdir("new_dir")
    if os.path.exists("inner_new_dir"):
        shutil.rmtree("inner_new_dir")
    os.mkdir("inner_new_dir")
    shutil.move("temp.ndb", "inner_new_dir/temp.ndb")
    shutil.move("inner_new_dir", "new_dir/inner_new_dir")
    os.chdir("new_dir")

    # We new_dir/, and inner_new_dir/ inside of new_dir/ which contains
    # temp.ndb. By only cd-ing into new_dir/ and loading from
    # inner_new_dir/temp.ndb, we make sure that path changes are handled
    # correctly.
    new_db = ndb.NeuralDB.from_checkpoint("inner_new_dir/temp.ndb")
    new_search_results = [r.text for r in new_db.search(ARBITRARY_QUERY, top_k=5)]

    assert search_results == new_search_results
    assert db.sources().keys() == new_db.sources().keys()
    assert [doc.name for doc in db.sources().values()] == [
        doc.name for doc in new_db.sources().values()
    ]

    # Save the loaded model and test the second saved model. Some metadata is
    # updated during the loading process. If saving a loaded model and then
    # loading it again works, we can induce that the metadata will not be
    # corrupted if we save and load an arbitrary number of times.
    new_db.save("temp_2.ndb")
    new_new_db = ndb.NeuralDB.from_checkpoint("temp_2.ndb")
    new_search_results = [r.text for r in new_new_db.search(ARBITRARY_QUERY, top_k=5)]

    assert search_results == new_search_results
    assert db.sources().keys() == new_db.sources().keys()
    assert [doc.name for doc in db.sources().values()] == [
        doc.name for doc in new_db.sources().values()
    ]

    os.chdir("..")
    shutil.rmtree("new_dir")


def clear_sources_works(db: ndb.NeuralDB):
    assert len(db.sources()) > 0
    db.clear_sources()
    assert len(db.sources()) == 0


@pytest.fixture(scope="session")
def empty_neural_db():
    """Initializes an empty NeuralDB once per test session to speed up tests.
    Best used for tests that don't assert accuracy.
    """
    db = ndb.NeuralDB()
    # db.insert() initializes the mach model so this only happens once per
    # test session. Clear the sources so it's back to being empty.
    db.insert([ndb.CSV(CSV_FILE)], train=False)
    db.clear_sources()
    yield db
