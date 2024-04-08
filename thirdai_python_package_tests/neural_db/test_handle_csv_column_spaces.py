import pytest
from thirdai import neural_db as ndb


@pytest.mark.unit
@pytest.mark.parametrize("explicit_cols", [False, True])
def test_handle_csv_column_spaces(explicit_cols):
    filename = "spaced_columns.csv"
    with open(filename, "w") as o:
        # We have "doc query" and "doc_query" columns to ensure that we properly
        # handle column name collisions.
        o.write("doc id,doc query,doc_query\n")
        o.write("0,query 0,query_0\n")
        o.write("1,query 1,query_1\n")
        o.write("2,query 2,query_2\n")

    csv = (
        ndb.CSV(
            filename,
            id_column="doc id",
            strong_columns=["doc query"],
            weak_columns=["doc_query"],
            reference_columns=["doc query", "doc_query"],
        )
        if explicit_cols
        else ndb.CSV(filename)
    )

    db = ndb.NeuralDB()

    # If this doesn't raise an error, then itertuples works
    db.insert([csv])

    # Ensure that constraints work with original column names
    assert len(db.search("query", 1, constraints={"doc query": "query 0"})) == 1
    assert len(db.search("query", 1, constraints={"doc_query": "query_1"})) == 1
