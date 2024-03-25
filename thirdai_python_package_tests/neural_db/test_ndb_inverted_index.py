import pandas as pd
import pytest
from download_dataset_fixtures import download_scifact_dataset
from thirdai.neural_db.documents import CSV, DocumentDataSource
from thirdai.neural_db.inverted_index import InvertedIndex
from thirdai.neural_db.supervised_datasource import Sup, SupDataSource
from thirdai.neural_db.documents import DocumentManager

pytestmark = [pytest.mark.unit, pytest.mark.release]


@pytest.mark.parametrize("shard_size, expected_num_shards", [(2000, 3), (6000, 1)])
def test_ndb_inverted_index(download_scifact_dataset, shard_size, expected_num_shards):
    doc_file, supervised_trn, query_file, _ = download_scifact_dataset

    doc_manager = DocumentManager(
        id_column="DOC_ID", strong_column="TITLE", weak_column="TEXT"
    )
    doc_manager.add(
        [
            CSV(
                doc_file,
                id_column="DOC_ID",
                strong_columns=["TITLE"],
                weak_columns=["TEXT"],
            )
        ]
    )

    index = InvertedIndex(max_shard_size=shard_size)

    index.insert(doc_manager.get_data_source())

    # Check all data is indexed and the expected number of shards are created.
    assert len(index.indexes) == expected_num_shards
    assert sum(i.size() for i in index.indexes) == 5183

    queries = pd.read_csv(query_file)
    queries["DOC_ID"] = queries["DOC_ID"].map(lambda x: list(map(int, x.split(":"))))

    def accuracy():
        correct = 0
        for _, row in queries.iterrows():
            pred = index.query([row["QUERY"]], k=4)[0][0][0]
            if pred in row["DOC_ID"]:
                correct += 1
        return correct / len(queries)

    acc = accuracy()

    print(f"{acc=}")
    assert acc >= 0.52  # Should be ~0.5367 for 3 shards and 0.53 for 1 shard

    supervised_data = SupDataSource(
        doc_manager=doc_manager,
        query_col="QUERY",
        data=[
            Sup(
                csv=supervised_trn,
                query_column="QUERY",
                id_column="DOC_ID",
                id_delimiter=":",
            )
        ],
        id_delimiter=":",
    )
    index.supervised_train(supervised_data)

    acc = accuracy()

    print(f"{acc=}")
    assert acc >= 0.72  # Should be ~0.7267 for 3 shards and 0.73 for 1 shard
