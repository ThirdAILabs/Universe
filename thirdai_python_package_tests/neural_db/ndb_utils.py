import os

import pytest
import requests
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
    lambda: ndb.SentenceLevelPDF(PDF_FILE),
    lambda: ndb.SentenceLevelDOCX(DOCX_FILE),
]
