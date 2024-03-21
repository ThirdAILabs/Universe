import pandas as pd
import pytest
from download_dataset_fixtures import download_scifact_dataset
from thirdai.neural_db.documents import CSV, DocumentDataSource
from thirdai.neural_db.inverted_index import InvertedIndex

pytestmark = [pytest.mark.unit, pytest.mark.release]


@pytest.mark.parametrize("shard_size, expected_num_shards", [(2000, 3), (6000, 1)])
def test_ndb_inverted_index(download_scifact_dataset, shard_size, expected_num_shards):
    doc_file, _, query_file, _ = download_scifact_dataset

    data_source = DocumentDataSource(id_column="", strong_column="", weak_column="")
    data_source.add(
        CSV(
            doc_file,
            id_column="DOC_ID",
            strong_columns=["TITLE"],
            weak_columns=["TEXT"],
        ),
        start_id=0,
    )

    index = InvertedIndex(max_shard_size=shard_size)

    index.insert(data_source)

    # Check all data is indexed and the expected number of shards are created.
    assert len(index.indexes) == expected_num_shards
    assert sum(i.size() for i in index.indexes) == 5183

    queries = pd.read_csv(query_file)
    queries["DOC_ID"] = queries["DOC_ID"].map(lambda x: list(map(int, x.split(":"))))

    correct = 0
    for _, row in queries.iterrows():
        pred = index.query([row["QUERY"]], k=4)[0][0][0]

        if pred in row["DOC_ID"]:
            correct += 1

    acc = correct / len(queries)
    print(f"{acc=}")
    assert acc >= 0.52  # Should be ~0.5367 for 3 shards and 0.53 for 1 shard
