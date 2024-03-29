from thirdai import neural_db as ndb


def test_handle_csv_column_spaces():
    filename = "spaced_columns.csv"
    with open(filename, "w") as o:
        o.write("doc id,doc query\n")
        o.write("0,query 0\n")
        o.write("1,query 1\n")
        o.write("2,query 2\n")

    db = ndb.NeuralDB()
    db.insert([ndb.CSV(filename)])

    # Make sure that the column names are actually converted
    assert len(db.search("query", 1, constraints={"doc id": 0})) == 0
    assert len(db.search("query", 1, constraints={"doc_id": 0})) == 1
    assert len(db.search("query", 1, constraints={"doc query": "query 0"})) == 0
    assert len(db.search("query", 1, constraints={"doc_query": "query 0"})) == 1
