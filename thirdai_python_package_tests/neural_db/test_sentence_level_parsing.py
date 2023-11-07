import os

import nltk
import pandas as pd
import pytest
from thirdai.neural_db import documents as docs

nltk.download("punkt")

pytestmark = [pytest.mark.unit]


class MockSentenceLevelDocument(docs.SentenceLevelExtracted):
    def __init__(self):
        # We need to write (and later remove) a dummy file since the parent
        # class expects an existing file.
        with open("temp", "w"):
            pass
        super().__init__(path="temp")
        os.remove("temp")

    def process_data(
        self,
        path: str,
    ) -> pd.DataFrame:
        paragraphs = [
            "This is a paragraph. This is the second sentence. This is the third.",
            'This paragraph has a bad "sentence". ?!*@&# (@)# . See what I\'m talking about?',
            "\n\n",
        ]

        return pd.DataFrame({"para": paragraphs, "display": paragraphs})


def test_sentence_level_document_removes_punctuation_only_sentences():
    doc = MockSentenceLevelDocument()
    assert doc.size == 5
    assert doc.reference(0).metadata["sentence"] == "This is a paragraph."
    assert doc.reference(1).metadata["sentence"] == "This is the second sentence."
    assert doc.reference(2).metadata["sentence"] == "This is the third."
    assert (
        doc.reference(3).metadata["sentence"] == 'This paragraph has a bad "sentence".'
    )
    assert doc.reference(4).metadata["sentence"] == "See what I'm talking about?"


def test_sentence_level_document_has_correct_upvote_ids():
    doc = MockSentenceLevelDocument()
    assert doc.reference(0).upvote_ids == [0, 1, 2]
    assert doc.reference(1).upvote_ids == [0, 1, 2]
    assert doc.reference(2).upvote_ids == [0, 1, 2]
    assert doc.reference(3).upvote_ids == [3, 4]
    assert doc.reference(4).upvote_ids == [3, 4]
