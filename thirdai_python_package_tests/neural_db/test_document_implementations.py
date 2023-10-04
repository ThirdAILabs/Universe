import os
import shutil
from pathlib import Path

import nltk

nltk.download("punkt")

import pytest
from ndb_utils import all_doc_getters
from thirdai import neural_db as ndb

pytestmark = [
    pytest.mark.unit,
    pytest.mark.parametrize("get_doc", all_doc_getters),
]


# The following tests are primarily to check that the implementations have a
# consistent set of methods and properties, not to check the correctness of each
# individual implementation. This is important since Python does not enforce
# this.
# The tests are parametrized with a getter function that returns a doc object
# to ensure that each test gets a new instance of the doc object. Some of the
# tested methods may mutate the doc object, which then affects other tests if
# they share the same doc instances.
def test_size_property(get_doc):
    doc = get_doc()
    assert type(doc.size) == int
    assert doc.size > 0
    for i in range(doc.size):
        # We don't make assertions since we only want to make sure nothing is out of range.
        doc.reference(i)
    with pytest.raises(ValueError):
        doc.reference(doc.size)


def test_name_property(get_doc):
    doc = get_doc()
    assert type(doc.name) == str


def test_reference_method(get_doc):
    doc = get_doc()
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


def test_hash_property(get_doc):
    doc = get_doc()
    assert type(doc.hash) == str


def test_strong_text_method(get_doc):
    doc = get_doc()
    for i in range(doc.size):
        assert type(doc.strong_text(i)) == str


def test_weak_text_method(get_doc):
    doc = get_doc()
    for i in range(doc.size):
        assert type(doc.weak_text(i)) == str


def test_context_method(get_doc):
    doc = get_doc()
    for i in range(doc.size):
        assert type(doc.context(i, radius=0)) == str
        if doc.size > 1:
            assert type(doc.context(i, radius=1)) == str
            assert len(doc.context(i, radius=1)) > len(doc.context(i, radius=0))


def test_save_load_meta_method(get_doc):
    doc = get_doc()
    save_dir = Path("doc_save_dir")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    # We just want to know that it does not throw.
    doc.save_meta(save_dir)
    doc.load_meta(save_dir)
    shutil.rmtree(save_dir)


def test_doc_save_load_method(get_doc):
    doc = get_doc()
    doc.save("doc_save_dir")
    # This calls a static method.
    loaded_doc = doc.load("doc_save_dir")

    assert loaded_doc.name == doc.name
    assert loaded_doc.hash == doc.hash
    assert loaded_doc.size == doc.size
    for i in range(doc.size):
        assert loaded_doc.reference(i).text == doc.reference(i).text
