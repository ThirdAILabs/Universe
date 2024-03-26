import pandas as pd
import pytest
from download_dataset_fixtures import download_scifact_dataset
from nltk.tokenize import word_tokenize
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
    df["QUERY"] = df["QUERY"].map(word_tokenize)
    df["DOC_ID"] = df["DOC_ID"].map(lambda x: list(map(int, x.split(":"))))

    return df


@pytest.mark.unit
def test_inverted_index(download_scifact_dataset):
    doc_file, trn_supervised, query_file, _ = download_scifact_dataset

    doc_df = pd.read_csv(doc_file)

    doc_df["TEXT"] = doc_df["TITLE"] + " " + doc_df["TEXT"]
    doc_df["TEXT"] = doc_df["TEXT"].map(word_tokenize)

    index = search.InvertedIndex()

    index.index(ids=doc_df["DOC_ID"].to_list(), docs=doc_df["TEXT"].to_list())

    query_df = load_supervised_data(query_file)
    unsupervised_acc = evaluate(index, query_df)

    print("unsupervised_acc=", unsupervised_acc)
    assert unsupervised_acc >= 0.52  # Should be 0.53 (should be deterministic)

    supervised_samples = load_supervised_data(trn_supervised)

    updates_ids, update_texts = [], []
    for _, row in supervised_samples.iterrows():
        for doc_id in row["DOC_ID"]:
            updates_ids.append(doc_id)
            update_texts.append(row["QUERY"])

    index.update(updates_ids, update_texts)

    supervised_acc = evaluate(index, query_df)

    print("supervised_acc=", supervised_acc)
    assert supervised_acc >= 0.71  # Should be 0.72 (should be deterministic)
