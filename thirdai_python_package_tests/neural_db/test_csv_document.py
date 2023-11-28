import os

import pandas as pd
import pytest
from thirdai import neural_db as ndb

pytestmark = [pytest.mark.unit]


@pytest.fixture(scope="module")
def make_csv_doc(request):
    explicit_columns = request.param.get("explicit_columns", None)
    doc_id_column = request.param.get("doc_id_column", None)

    doc_ids = list(range(100))
    strongs = [f"This is strong text {doc_id}" for doc_id in doc_ids]
    weaks = [f"This is weak text {doc_id}" for doc_id in doc_ids]
    # We reverse the doc IDs so we can easily tell if the IDs are consistent
    # with the doc ID column.
    doc_ids.reverse()
    path = "test_csv_document.csv"
    pd.DataFrame({"doc_id": doc_ids, "strong": strongs, "weak": weaks}).to_csv(
        path, index=False
    )
    if not explicit_columns:
        ndb_doc = ndb.CSV(path)
    else:
        ndb_doc = ndb.CSV(
            path,
            id_column="doc_id" if doc_id_column else None,
            strong_columns=["strong"],
            weak_columns=["weak"],
            reference_columns=["strong", "weak"],
        )

    yield ndb_doc
    os.remove(path)


def strong_columns_empty(doc: ndb.Document):
    for i in range(doc.size):
        if doc.strong_text(i) != "":
            return False
    return True


def valid_inferred_weak_columns(doc: ndb.Document):
    """Asserts that the weak text of each entry is a concatenation of all text
    columns in the CSV. Does not check order of rows.
    """
    for i in range(doc.size):
        # Weak text is a concatenation of all text columns
        if "strong" not in doc.weak_text(i):
            return False
        if "weak" not in doc.weak_text(i):
            return False
    return True


def valid_explicit_strong_columns(doc):
    """Asserts that the strong text of each entry is from the "strong" column of
    the CSV. Does not check order of rows.
    """
    for i in range(doc.size):
        if "strong" not in doc.strong_text(i):
            return False
        if "weak" in doc.strong_text(i):
            return False
    return True


def valid_explicit_weak_columns(doc):
    """Asserts that the weak text of each entry is from the "weak" column of
    the CSV. Does not check order of rows.
    """
    for i in range(doc.size):
        if "weak" not in doc.weak_text(i):
            return False
        if "strong" in doc.weak_text(i):
            return False
    return True


def ids_are_row_numbers(doc: ndb.Document):
    for i in range(doc.size):
        if not str(i) in doc.weak_text(i):
            return False
    return True


def ids_consistent_with_doc_id_column(doc: ndb.Document):
    # The doc ID column is reversed, so our CSV file looks something like:
    # doc_id,strong,weak
    # N - 1,strong 0,weak 0
    # N - 2,strong 1,weak 1
    # N - 3,strong 2,weak 2
    for i in range(doc.size):
        if not str(doc.size - 1 - i) in doc.strong_text(i):
            return False
    return True


@pytest.mark.parametrize(
    "make_csv_doc", [{"explicit_columns": False, "doc_id_column": False}], indirect=True
)
def test_csv_with_inferred_columns(make_csv_doc):
    doc = make_csv_doc
    assert strong_columns_empty(doc)
    assert valid_inferred_weak_columns(doc)
    assert ids_are_row_numbers(doc)


@pytest.mark.parametrize(
    "make_csv_doc", [{"explicit_columns": True, "doc_id_column": True}], indirect=True
)
def test_csv_with_explicit_columns_with_doc_id_column(make_csv_doc):
    doc = make_csv_doc
    assert valid_explicit_strong_columns(doc)
    assert valid_explicit_weak_columns(doc)
    assert ids_consistent_with_doc_id_column(doc)


@pytest.mark.parametrize(
    "make_csv_doc", [{"explicit_columns": True, "doc_id_column": False}], indirect=True
)
def test_csv_with_explicit_columns_without_doc_id_column(make_csv_doc):
    doc = make_csv_doc
    assert valid_explicit_strong_columns(doc)
    assert valid_explicit_weak_columns(doc)
    assert ids_are_row_numbers(doc)
