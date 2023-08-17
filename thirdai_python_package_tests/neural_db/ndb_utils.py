import os

import pytest
from thirdai import neural_db as ndb
from enum import Enum


@pytest.fixture
def create_simple_dataset():
    filename = "simple.csv"
    with open(filename, "w") as file:
        file.writelines(
            [
                "text,id\n",
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
        id_column="id",
        strong_columns=["text"],
        weak_columns=["text"],
        reference_columns=["id", "text"],
    )

    db.insert(sources=[doc], train=True)

    return db


class Docs(Enum):
    CSV = "csv"
    PDF = "pdf"
    DOCX = "docx"
    URL = "url"
    SENTENCE_PDF = "sentence_pdf"
    SENTENCE_DOCX = "sentence_docx"


@pytest.fixture(scope="session")
def all_docs():
    BASE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "document_test_data"
    )
    CSV_FILE = os.path.join(BASE_DIR, "lorem_ipsum.csv")
    PDF_FILE = os.path.join(BASE_DIR, "mutual_nda.pdf")
    DOCX_FILE = os.path.join(BASE_DIR, "four_english_words.docx")
    yield {
        Docs.CSV.value: ndb.CSV(
            CSV_FILE,
            id_column="category",
            strong_columns=["text"],
            weak_columns=["text"],
            reference_columns=["text"],
        ),
        Docs.PDF.value: ndb.PDF(PDF_FILE),
        Docs.DOCX.value: ndb.DOCX(DOCX_FILE),
        Docs.URL.value: ndb.URL("https://en.wikipedia.org/wiki/Rice_University"),
        Docs.SENTENCE_PDF.value: ndb.SentenceLevelPDF(PDF_FILE),
        Docs.SENTENCE_DOCX.value: ndb.SentenceLevelDOCX(DOCX_FILE),
    }


doc_choices = [doc.value for doc in Docs]
