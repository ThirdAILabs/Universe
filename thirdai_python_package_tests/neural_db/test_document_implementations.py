import os
import shutil
from pathlib import Path

import nltk

nltk.download("punkt")

import pytest
from common_document_tests import assess_doc_methods_properties
from ndb_utils import all_local_doc_getters
from thirdai import neural_db as ndb

pytestmark = [
    pytest.mark.unit,
    pytest.mark.parametrize("get_doc", all_local_doc_getters),
]


# The following tests are primarily to check that the implementations have a
# consistent set of methods and properties, not to check the correctness of each
# individual implementation. This is important since Python does not enforce
# this.
# The tests are parametrized with a getter function that returns a doc object
# to ensure that each test gets a new instance of the doc object. Some of the
# tested methods may mutate the doc object, which then affects other tests if
# they share the same doc instances.


def test_doc_methods_properties(get_doc):
    doc = get_doc()
    assess_doc_methods_properties(doc)


def test_doc_save_load_method(get_doc):
    doc = get_doc()
    save_to = "doc_save_dir"
    doc.save(save_to)
    # This calls a static method.
    loaded_doc = doc.load(save_to)
    shutil.rmtree(save_to)

    assert loaded_doc.name == doc.name
    assert loaded_doc.hash == doc.hash
    assert loaded_doc.size == doc.size
    for i in range(doc.size):
        assert loaded_doc.reference(i).text == doc.reference(i).text
