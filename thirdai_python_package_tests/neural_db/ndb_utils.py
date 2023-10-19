import os

import pytest
import requests
from pathlib import Path
from thirdai import neural_db as ndb
from thirdai.neural_db.documents import Document, Reference
from thirdai.neural_db.constraint_matcher import ConstraintValue


class MockDocument(Document):
    def __init__(self, identifier: str, size: int, metadata: dict = {}) -> None:
        self._identifier = identifier
        self._size = size
        self._save_meta_called = 0
        self._save_meta_dir = None
        self._load_meta_called = 0
        self._load_meta_dir = None
        self._doc_metadata = metadata

    # We don't implement hash to test the default implementation

    @property
    def matched_constraints(self):
        return {key: ConstraintValue(value) for key, value in self.doc_metadata.items()}

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
