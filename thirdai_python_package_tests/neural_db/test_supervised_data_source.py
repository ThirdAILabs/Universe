import pytest
from thirdai.neural_db.neural_db import SupDataSource, DocumentManager
from thirdai import neural_db as ndb

# input uses sequence
# single label csv input
# multi lable csv input with trailing delimiter
# All cases with and without model id delimiter


pytestmark = [pytest.mark.unit, pytest.mark.release]


def multilabel_str(labels, delimiter):
    if not delimiter:
        return str(labels[0])
    return delimiter.join(map(str, labels))


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
    assert data_source.next_batch(target_batch_size=5) == [
        "query,id",
        "this is the first query,0",
        "this is the second query," + multilabel_str([0, 1], model_id_delimiter),
        "trailing label delimiter," + multilabel_str([0, 1], model_id_delimiter),
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
    assert data_source.next_batch(target_batch_size=5) == [
        "query,id",
        "this is the first query,0",
        "this is the second query,1",
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
    assert data_source.next_batch(target_batch_size=5) == [
        "query,id",
        "this is the first query,0",
        "this is the second query," + multilabel_str([0, 1], model_id_delimiter),
    ]
