import os

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
PDF_FILE = os.path.join(BASE_DIR, "mutual_nda.pdf")
DOCX_FILE = os.path.join(BASE_DIR, "four_english_words.docx")
PPTX_FILE = os.path.join(BASE_DIR, "quantum_mechanics.pptx")
TXT_FILE = os.path.join(BASE_DIR, "nature.txt")
EML_FILE = os.path.join(BASE_DIR, "Message.eml")

# connection instances for connector document
ENGINE = base.get_sql_engine()
TABLE_NAME = base.get_sql_table()

CLIENT_CONTEXT = base.get_client_context()
LIBRARY_PATH = base.get_library_path()

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
            engine=ENGINE,
            table_name=TABLE_NAME,
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
            ctx=CLIENT_CONTEXT, library_path=LIBRARY_PATH
        ),
        local_doc=build_local_sharepoint_doc,
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
    lambda: ndb.URL("https://en.wikipedia.org/wiki/Rice_University"),
    lambda: ndb.URL(
        "https://en.wikipedia.org/wiki/Rice_University",
        requests.get("https://en.wikipedia.org/wiki/Rice_University"),
    ),
    lambda: ndb.Unstructured(PPTX_FILE),
    lambda: ndb.Unstructured(TXT_FILE),
    lambda: ndb.Unstructured(EML_FILE),
    lambda: ndb.SentenceLevelPDF(PDF_FILE),
    lambda: ndb.SentenceLevelDOCX(DOCX_FILE),
]

# The two URL docs are different constructor invocationsfor the same thing.
num_duplicate_docs = 1

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
        ndb.URL(
            "https://en.wikipedia.org/wiki/Rice_University",
            metadata=meta(URL_NO_RESPONSE_META),
        ),
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
