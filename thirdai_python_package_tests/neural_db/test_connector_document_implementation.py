import os
import shutil
from pathlib import Path

import pytest
from base_connectors.base import get_base_connectors
from document_common_tests import assess_doc_methods_properties
from ndb_utils import all_connector_doc_getters
from thirdai import neural_db as ndb

pytestmark = [
    pytest.mark.unit,
    pytest.mark.parametrize("get_connector_doc", all_connector_doc_getters),
]


def test_all_batch_fetchable(get_connector_doc):
    doc = get_connector_doc.connector_doc()
    rows_fetched = 0
    for current_chunk in doc.chunk_iterator():
        rows_fetched += len(current_chunk)

    assert rows_fetched == doc.size


def test_document_equivalence(get_connector_doc):
    connector_doc, local_doc = (
        get_connector_doc.connector_doc(),
        get_connector_doc.local_doc(),
    )
    assert connector_doc.size == local_doc.size
    for row in connector_doc.row_iterator():
        assert row.strong == local_doc.strong_text(row.id)
        assert row.weak == local_doc.weak_text(row.id)


def test_connector_reattach(get_connector_doc):
    connector_doc = get_connector_doc.connector_doc()
    saved_path = "doc_save_dir"
    connector_doc.save(saved_path)

    loaded_doc = ndb.Document.load(saved_path)
    shutil.rmtree(saved_path)
    with pytest.raises(AttributeError):
        connector = getattr(loaded_doc, loaded_doc._get_connector_object_name())

    # Setup phase should not throw any error
    connector_instance = get_base_connectors(loaded_doc)
    loaded_doc.setup_connection(connector_instance)


def test_doc_property(get_connector_doc):
    connector_doc = get_connector_doc.connector_doc()
    assess_doc_methods_properties(doc=connector_doc)
