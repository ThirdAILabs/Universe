import pytest
from thirdai import neural_db as ndb


@pytest.mark.unit
@pytest.mark.parametrize("explicit_cols", [False, True])
def test_handle_csv_column_spaces(explicit_cols):
    filename = "spaced_columns.csv"
    with open(filename, "w") as o:
        o.write("doc id,doc query\n")
        o.write("0,query 0\n")
        o.write("1,query 1\n")
        o.write("2,query 2\n")

    csv = (
        ndb.CSV(
            filename,
            id_column="doc id",
            strong_columns=["doc query"],
            weak_columns=["doc query"],
            reference_columns=["doc query"],
        )
        if explicit_cols
        else ndb.CSV(filename)
    )

    db = ndb.NeuralDB()
    db.insert([csv])

    # Make sure that the column names are actually converted
    assert len(db.search("query", 1, constraints={"doc query": "query 0"})) == 0
    assert len(db.search("query", 1, constraints={"doc_query": "query 0"})) == 1
