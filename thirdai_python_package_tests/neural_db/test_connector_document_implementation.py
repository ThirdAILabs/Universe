import os

import pytest
from ndb_utils import all_connector_doc_getter
from thirdai import neural_db as ndb

pytestmark = [
    pytest.mark.unit,
    pytest.mark.parametrize("get_connector_doc", all_connector_doc_getter),
]

DOC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "document_test_data", "connector_docs"
)


def test_all_batch_fetchable(get_connector_doc):
    doc = get_connector_doc.connector_doc()
    rows_fetched = 0
    for current_chunk in doc.next_chunk():
        rows_fetched += len(current_chunk)

    assert rows_fetched == doc.size


def test_document_equivalence(get_connector_doc):
    connector_doc, local_doc = (
        get_connector_doc.connector_doc(),
        get_connector_doc.local_doc(),
    )
    assert connector_doc.size == local_doc.size
    for row in connector_doc.row_iterator():
        assert row.strong == local_doc.strong_text(row.element_id)
        assert row.weak == local_doc.weak_text(row.element_id)
