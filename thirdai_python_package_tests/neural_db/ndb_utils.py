import os
import requests
import pytest
from thirdai import neural_db as ndb


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


def all_docs():
    BASE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "document_test_data"
    )
    CSV_FILE = os.path.join(BASE_DIR, "lorem_ipsum.csv")
    PDF_FILE = os.path.join(BASE_DIR, "mutual_nda.pdf")
    DOCX_FILE = os.path.join(BASE_DIR, "four_english_words.docx")
    return [
        ndb.CSV(
            CSV_FILE,
            id_column="category",
            strong_columns=["text"],
            weak_columns=["text"],
            reference_columns=["text"],
        ),
        ndb.CSV(CSV_FILE),
        ndb.PDF(PDF_FILE),
        ndb.DOCX(DOCX_FILE),
        ndb.URL("https://en.wikipedia.org/wiki/Rice_University"),
        ndb.URL(
            "https://en.wikipedia.org/wiki/Rice_University",
            requests.get("https://en.wikipedia.org/wiki/Rice_University"),
        ),
        ndb.SentenceLevelPDF(PDF_FILE),
        ndb.SentenceLevelDOCX(DOCX_FILE),
    ]
