import pytest
from thirdai import neural_db as ndb
from thirdai.neural_db.neural_db import DocumentManager, SupDataSource

pytestmark = [pytest.mark.unit, pytest.mark.release]


def expected_rows(queries, labels, delimiter):
    if delimiter:
        return [
            delimiter.join(map(str, label_row)) + "," + query
            for query, label_row in zip(queries, labels)
        ]
    return [
        str(label) + "," + query
        for query, label_row in zip(queries, labels)
        for label in label_row
    ]


TARGET_BATCH_SIZE = 1000  # Just something big


@pytest.mark.parametrize("model_id_delimiter", [" ", None])
def test_sup_data_source(model_id_delimiter):
    with open("mock_unsup.csv", "w") as out:
        out.write("id,strong\n")
        out.write("0,this is the first query\n")
        out.write("1,this is the second query\n")

    doc_manager = DocumentManager(
        id_column="id", strong_column="strong", weak_column="weak"
    )

    source_id = doc_manager.add(
        [ndb.CSV("mock_unsup.csv", id_column="id", strong_columns=["strong"])]
    )[1][0]

    # Test multi label case (with id delimiter)
    with open("mock_sup.csv", "w") as out:
        out.write("id,query\n")
        # make sure that single label rows are also handled correctly in a
        # multilabel dataset.
        out.write("0,this is the first query\n")
        out.write("0:1,this is the second query\n")
        out.write("0:1:,trailing label delimiter\n")
    sup_doc = ndb.Sup(
        "mock_sup.csv",
        query_column="query",
        id_column="id",
        id_delimiter=":",
        source_id=source_id,
    )
    data_source = SupDataSource(
        doc_manager, query_col="query", data=[sup_doc], id_delimiter=model_id_delimiter
    )
    assert data_source.next_batch(TARGET_BATCH_SIZE) == [
        "id,query",
        *expected_rows(
            queries=[
                "this is the first query",
                "this is the second query",
                "trailing label delimiter",
            ],
            labels=[[0], [0, 1], [0, 1]],
            delimiter=model_id_delimiter,
        ),
    ]

    # Test single label case (without id delimiter)
    with open("mock_sup.csv", "w") as out:
        out.write("id,query\n")
        out.write("0,this is the first query\n")
        out.write("1,this is the second query\n")
    sup_doc = ndb.Sup(
        "mock_sup.csv",
        query_column="query",
        id_column="id",
        id_delimiter=":",
        source_id=source_id,
    )
    data_source = SupDataSource(
        doc_manager, query_col="query", data=[sup_doc], id_delimiter=model_id_delimiter
    )
    assert data_source.next_batch(TARGET_BATCH_SIZE) == [
        "id,query",
        "0,this is the first query",
        "1,this is the second query",
    ]

    data_source = SupDataSource(
        doc_manager,
        query_col="query",
        data=[
            ndb.Sup(
                "mock_sup.csv",
                queries=["this is the first query", "this is the second query"],
                labels=[[0], [0, 1]],
                source_id=source_id,
            )
        ],
        id_delimiter=model_id_delimiter,
    )
    assert data_source.next_batch(TARGET_BATCH_SIZE) == [
        "id,query",
        *expected_rows(
            queries=[
                "this is the first query",
                "this is the second query",
            ],
            labels=[[0], [0, 1]],
            delimiter=model_id_delimiter,
        ),
    ]


def test_sup_data_source_with_id_map():
    with open("mock_unsup.csv", "w") as out:
        out.write("id,strong\n")
        out.write("one,this is the first query\n")
        out.write("two,this is the second query\n")

    doc_manager = DocumentManager(
        id_column="model_id", strong_column="strong", weak_column="weak"
    )

    _, source_ids = doc_manager.add(
        [ndb.CSV("mock_unsup.csv", id_column="id", strong_columns=["strong"])]
    )
    source_id = source_ids[0]

    # Test multi label case (with id delimiter)
    with open("mock_sup.csv", "w") as out:
        out.write("id,query\n")
        # make sure that single label rows are also handled correctly in a
        # multilabel dataset.
        out.write("one,this is the first query\n")
        out.write("one:two,this is the second query\n")
        out.write("one:two:,trailing label delimiter\n")

    sup_doc = ndb.Sup(
        "mock_sup.csv",
        query_column="query",
        id_column="id",
        id_delimiter=":",
        source_id=source_id,
    )

    mock_model_query_col = "model_query"
    mock_model_id_delimiter = " "
    data_source = SupDataSource(
        doc_manager,
        query_col=mock_model_query_col,
        data=[sup_doc],
        id_delimiter=mock_model_id_delimiter,
    )

    assert data_source.next_batch(TARGET_BATCH_SIZE) == [
        "model_id,model_query",
        "0,this is the first query",
        "0 1,this is the second query",
        "0 1,trailing label delimiter",
    ]
