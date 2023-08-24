import os
import shutil
from pathlib import Path

import pytest
from ndb_utils import all_docs
from thirdai import neural_db as ndb

pytestmark = [
    pytest.mark.unit,
    pytest.mark.parametrize("doc", all_docs()),
]


# The following tests are primarily to check that the implementations have a
# consistent set of methods and properties, not to check the correctness of each
# individual implementation. This is important since Python does not enforce
# this.
def test_size_property(doc):
    assert type(doc.size) == int
    assert doc.size > 0
    for i in range(doc.size):
        # We don't make assertions since we only want to make sure nothing is out of range.
        doc.reference(i)
    with pytest.raises(IndexError):
        doc.reference(doc.size)


def test_name_property(doc):
    assert type(doc.name) == str


def test_reference_method(doc):
    for i in range(doc.size):
        reference: ndb.Reference = doc.reference(i)
        assert type(reference.id) == int
        assert reference.id == i

        assert type(reference.upvote_ids) == list
        assert len(reference.upvote_ids) >= 1
        assert type(reference.upvote_ids[0]) == int

        assert type(reference.text) == str
        assert reference.text != ""

        assert type(reference.source) == str
        assert type(reference.metadata) == dict
        assert type(reference.context(radius=0)) == str


def test_hash_property(doc):
    assert type(doc.hash) == str


def test_strong_text_method(doc):
    for i in range(doc.size):
        assert type(doc.strong_text(i)) == str


def test_weak_text_method(doc):
    for i in range(doc.size):
        assert type(doc.weak_text(i)) == str


def test_context_method(doc):
    for i in range(doc.size):
        assert type(doc.context(i, radius=0)) == str
        if doc.size > 1:
            assert type(doc.context(i, radius=1)) == str
            assert len(doc.context(i, radius=1)) > len(doc.context(i, radius=0))


def test_save_load_meta_method(doc):
    save_dir = Path("doc_save_dir")
    os.mkdir(save_dir)

    # We just want to know that it does not throw.
    doc.save_meta(save_dir)
    doc.load_meta(save_dir)
    shutil.rmtree(save_dir)
