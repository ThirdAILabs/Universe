import os

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def load_chunks():
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../auto_ml/python_tests/texts.csv",
    )

    return pd.read_csv(filename)


DOC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../neural_db/document_test_data"
)

CSV_FILE = os.path.join(DOC_DIR, "lorem_ipsum.csv")
DOCX_FILE = os.path.join(DOC_DIR, "four_english_words.docx")
URL_LINK = "https://en.wikipedia.org/wiki/Rice_University"
PDF_FILE = os.path.join(DOC_DIR, "mutual_nda.pdf")
PPTX_FILE = os.path.join(DOC_DIR, "quantum_mechanics.pptx")
TXT_FILE = os.path.join(DOC_DIR, "nature.txt")
EML_FILE = os.path.join(DOC_DIR, "Message.eml")
