import os
import sys
import time

import pandas as pd
import pytest
from document_common_tests import assess_doc_methods_properties
from ndb_utils import BASE_DIR, on_diskable_doc_getters
from thirdai import neural_db as ndb
from thirdai.neural_db.table import SQLiteTable


def get_size(obj, seen=None):
    """Recursively finds size of objects
    From https://goshippo.com/blog/measure-real-size-any-python-object
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


@pytest.mark.unit
@pytest.mark.parametrize(
    "on_disk_doc, in_memory_doc_1, in_memory_doc_2",
    zip(
        [get_doc() for get_doc in on_diskable_doc_getters(on_disk=True)],
        [get_doc() for get_doc in on_diskable_doc_getters(on_disk=False)],
        [get_doc() for get_doc in on_diskable_doc_getters(on_disk=False)],
    ),
)
def test_tables_on_disk_table_uses_less_memory(
    on_disk_doc, in_memory_doc_1, in_memory_doc_2
):
    assert get_size(in_memory_doc_1) == get_size(in_memory_doc_2)
    assert get_size(on_disk_doc) < get_size(in_memory_doc_1)


@pytest.mark.unit
@pytest.mark.parametrize(
    "DocClass, save_dir",
    [
        (ndb.CSV, "csv"),
        (ndb.URL, "url"),
        # Backwards compatibility code for both DOCX and PDF is handled by a single
        # parent class so we only need to test one. Same is true for
        # SentenceLevelDOCX and SentenceLevelPDF.
        (ndb.PDF, "pdf"),
        (ndb.SentenceLevelPDF, "sentence_pdf"),
    ],
)
def test_tables_dataframe_table_backwards_compatibility(DocClass, save_dir):
    load_from = f"{BASE_DIR}/v0.7.26_doc_checkpoints/{save_dir}"
    doc = DocClass.load(load_from)
    assess_doc_methods_properties(doc)


@pytest.mark.unit
def test_row_id_is_primary_key():
    df = pd.DataFrame(
        {
            "id": range(100000),
            "other": range(100000),
        }
    )

    df = df.set_index("id")
    table = SQLiteTable(df)

    s = time.time()
    for i in range(1000):
        table.select_with_constraint(column="id", value=i)
    id_duration = time.time() - s

    s = time.time()
    for i in range(1000):
        table.select_with_constraint(column="other", value=i)
    other_duration = time.time() - s

    # Querying by primary key is about 6.6X faster on mac.
    # Use factor of 3 for testing so it's not flaky.
    assert other_duration > (3 * id_duration)

    os.remove(table.db_path)
