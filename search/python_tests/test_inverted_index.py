import os

import pandas as pd
import pytest
from download_dataset_fixtures import download_scifact_dataset
from thirdai import search


def evaluate(index, test_set):
    results = index.query(test_set["QUERY"].to_list(), k=1)

    correct = 0
    for result, labels in zip(results, test_set["DOC_ID"]):
        if result[0][0] in labels:
            correct += 1

    return correct / len(test_set)


def load_supervised_data(filename):
    df = pd.read_csv(filename)
    df["DOC_ID"] = df["DOC_ID"].map(lambda x: list(map(int, x.split(":"))))

    return df


@pytest.mark.unit
def test_inverted_index(download_scifact_dataset):
    doc_file, trn_supervised, query_file, _ = download_scifact_dataset

    doc_df = pd.read_csv(doc_file)

    doc_df["TEXT"] = doc_df["TITLE"] + " " + doc_df["TEXT"]

    index = search.Finetunabl(shard_size=1000)

    index.index(ids=doc_df["DOC_ID"].to_list(), docs=doc_df["TEXT"].to_list())

    query_df = load_supervised_data(query_file)
    unsupervised_acc = evaluate(index, query_df)

    print("unsupervised_acc=", unsupervised_acc)
    assert unsupervised_acc >= 0.54  # Should be 0.543 (should be deterministic)

    path = "./scifact.index"
    index.save(path)

    index = search.InvertedIndex.load(path)
    os.remove(path)

    after_load_acc = evaluate(index, query_df)
    print("after_load_acc=", after_load_acc)
    assert after_load_acc == unsupervised_acc
