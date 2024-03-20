from collections import defaultdict
from copy import deepcopy

import pytest
from thirdai import neural_db as ndb
from thirdai.neural_db.neural_db import DocumentManager
from thirdai.neural_db.sharded_documents import shard_data_source
from thirdai.neural_db.supervised_datasource import SupDataSource

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


def make_csvs_and_sups_and_add_to_document_manager(
    document_manager: DocumentManager, number_docs: int, queries_per_doc: int
):
    csv_docs = []
    sup_docs = []

    for doc_number in range(number_docs):
        with open(f"mock_sup_{doc_number}.csv", "w") as f:
            f.write("id,query\n")
            for id in range(queries_per_doc):
                f.write(f"{id},this is sup {doc_number} query number {id}\n")

        csv_doc = ndb.CSV(
            f"mock_sup_{doc_number}.csv", id_column="id", strong_columns=["query"]
        )
        csv_docs.append(csv_doc)
        _, source_ids = document_manager.add(documents=[csv_doc])
        sup_docs.append(
            ndb.Sup(
                f"mock_sup_{doc_number}.csv",
                query_column="query",
                id_column="id",
                id_delimiter=":",
                source_id=source_ids[0],
            )
        )
    return csv_docs, sup_docs


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
        'id,"query"',
        *expected_rows(
            queries=[
                '"this is the first query"',
                '"this is the second query"',
                '"trailing label delimiter"',
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
        'id,"query"',
        '0,"this is the first query"',
        '1,"this is the second query"',
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
        'id,"query"',
        *expected_rows(
            queries=[
                '"this is the first query"',
                '"this is the second query"',
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
        'model_id,"model_query"',
        '0,"this is the first query"',
        '0 1,"this is the second query"',
        '0 1,"trailing label delimiter"',
    ]

@pytest.mark.parametrize("number_shards", [1,2])
def test_sup_data_source_sharding(number_shards):
    doc_manager = DocumentManager(
        id_column="model_id", strong_column="strong", weak_column="weak"
    )

    _, sup_docs = make_csvs_and_sups_and_add_to_document_manager(
        document_manager=doc_manager, number_docs=3, queries_per_doc=20
    )

    document_data_source = doc_manager.get_data_source()
    sup_data_source = SupDataSource(
        doc_manager=doc_manager, query_col="query", data=sup_docs, id_delimiter=None
    )
    label_to_segment_map = defaultdict(list)

    document_data_source_shards = shard_data_source(
        data_source=document_data_source,
        label_to_segment_map=label_to_segment_map,
        number_shards=number_shards,
        update_segment_map=True,
    )

    copied_label_to_segment_map = deepcopy(label_to_segment_map)

    sup_data_source_shards = shard_data_source(
        sup_data_source,
        label_to_segment_map=label_to_segment_map,
        number_shards=number_shards,
        update_segment_map=False,
    )

    assert copied_label_to_segment_map == label_to_segment_map

    for document_shard, sup_shard in zip(
        document_data_source_shards, sup_data_source_shards
    ):
        lines1 = list(document_shard._get_line_iterator())
        lines2 = list(sup_shard._get_line_iterator())

        assert len(lines1) == len(lines2)
        for index in range(1, len(lines1)):
            elements1 = [x.strip('"') for x in lines1[index].split(",")]
            elements2 = [x.strip('"') for x in lines2[index].split(",")]
            assert elements1[0] == elements2[0]
            assert elements1[0] == elements2[0]


def test_sup_data_source_sharding_multilabel():
    """
    This test verifies that sharding works for SupDataSource even when there are multiple labels. Another unintended check of this test is that even when when all labels in label_to_segment_map maps to 1 shard, sharding is still successful.
    """
    doc_manager = DocumentManager(
        id_column="model_id", strong_column="strong", weak_column="weak"
    )
    queries = [
        "query_one",
        "query_two_three",
    ]
    labels = [[1], [2, 3]]

    sup = ndb.Sup(queries=queries, labels=labels, uses_db_id=True)
    sup_source = SupDataSource(
        doc_manager=doc_manager, query_col="query", data=[sup], id_delimiter=","
    )

    label_to_segment_map = defaultdict(list)
    for i in range(1, 4):
        label_to_segment_map[i] = [0]

    sup_shards = shard_data_source(
        data_source=sup_source,
        label_to_segment_map=label_to_segment_map,
        number_shards=2,
        update_segment_map=False,
    )
    for index, line in enumerate(list(sup_shards[0]._get_line_iterator())[1:]):
        assert index + 1 == int(line.split(",")[0])
