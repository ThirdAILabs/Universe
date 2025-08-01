import os
import random
import sys
import time

import pandas as pd
import pytest
from document_common_tests import assess_doc_methods_properties
from ndb_utils import BASE_DIR, on_diskable_doc_getters
from thirdai import neural_db as ndb
from thirdai.neural_db.table import DataFrameTable, SQLiteTable


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
def test_tables_on_disk_table_uses_less_memory():
    df = pd.DataFrame(
        {
            "id": range(100000),
            "text": [f"this is the {i}-th line" for i in range(100000)],
        }
    )
    df = df.set_index("id")
    assert get_size(SQLiteTable(df)) < get_size(DataFrameTable(df))


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
def test_sqlitetable_select_by_row_id_is_fast():
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
    # Use factor of 1.5 for testing so it's not flaky.
    assert other_duration > (1.5 * id_duration)

    os.remove(table.db_path)


@pytest.mark.unit
def test_table_methods_consistent_across_implementations():
    df = pd.DataFrame(
        {
            "id": range(100),
            "other": range(100),
        }
    )

    df = df.set_index("id")

    sqlite = SQLiteTable(df)
    pandas = DataFrameTable(df)

    assert all(sqlite.columns == pandas.columns)
    assert sqlite.size == pandas.size
    assert all(sqlite.ids == pandas.ids)

    sqlite_iter = sqlite.iter_rows_as_dicts()
    pandas_iter = sqlite.iter_rows_as_dicts()
    for sqlite_row, pandas_row in zip(sqlite_iter, pandas_iter):
        assert sqlite_row == pandas_row

    for i in range(10, 20):
        assert sqlite.field(i, "other") == pandas.field(i, "other")
        assert sqlite.row_as_dict(i) == pandas.row_as_dict(i)
        for sqlite_row, pandas_row in zip(
            sqlite.range_rows_as_dicts(i, i + 10),
            pandas.range_rows_as_dicts(i, i + 10),
        ):
            assert sqlite_row == pandas_row
