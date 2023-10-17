import os

import pytest
import requests
from sqlalchemy import create_engine
from thirdai import neural_db as ndb


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

DB_URL = "sqlite:///" + os.path.join(BASE_DIR, "Amazon_polarity.db")
ENGINE = create_engine(DB_URL)
TABLE_NAME = "Amzn_1K"

CSV_EXPLICIT_META = "csv-explicit"
PDF_META = "pdf"
DOCX_META = "docx"
URL_NO_RESPONSE_META = "url-no-response"
PPTX_META = "pptx"
TXT_META = "txt"
EML_META = "eml"
SQL_META = "sql"
SENTENCE_PDF_META = "sentence-pdf"
SENTENCE_DOCX_META = "sentence-docx"


def meta(file_meta):
    return {"meta": file_meta}


# This is a list of getter functions that return doc objects so each test can
# use fresh doc object instances.
all_doc_getters = [
    lambda: ndb.CSV(
        CSV_FILE,
        id_column="category",
        strong_columns=["text"],
        weak_columns=["text"],
        reference_columns=["text"],
    ),
    lambda: ndb.CSV(CSV_FILE),
    lambda: ndb.PDF(PDF_FILE),
    lambda: ndb.DOCX(DOCX_FILE),
    lambda: ndb.URL("https://en.wikipedia.org/wiki/Rice_University"),
    lambda: ndb.URL(
        "https://en.wikipedia.org/wiki/Rice_University",
        requests.get("https://en.wikipedia.org/wiki/Rice_University"),
    ),
    lambda: ndb.Unstructured(PPTX_FILE),
    lambda: ndb.Unstructured(TXT_FILE),
    lambda: ndb.Unstructured(EML_FILE),
    lambda: ndb.SQLDocument(
        engine=ENGINE,
        table_name=TABLE_NAME,
        id_col="id",
        strong_columns=["content"],
        weak_columns=["content"],
        reference_columns=["content"],
        chunk_size=100,
    ),
    lambda: ndb.SentenceLevelPDF(PDF_FILE),
    lambda: ndb.SentenceLevelDOCX(DOCX_FILE),
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
        ndb.SQLDocument(
            engine=ENGINE,
            table_name=TABLE_NAME,
            id_col="id",
            strong_columns=["content"],
            weak_columns=["content"],
            reference_columns=["content"],
            chunk_size=100,
            metadata=meta(SQL_META),
        ),
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
