from thirdai import search

from download_dataset_fixtures import download_scifact_dataset
import pandas as pd
from nltk.tokenize import word_tokenize


def test_inverted_index(download_scifact_dataset):
    doc_file, _, query_file, _ = download_scifact_dataset

    doc_df = pd.read_csv(doc_file)

    doc_df["TEXT"] = doc_df["TITLE"] + " " + doc_df["TEXT"]
    doc_df.drop("TITLE", inplace=True, axis=1)
    doc_df["TEXT"] = doc_df["TEXT"].map(word_tokenize)

    docs = [(row["DOC_ID"], row["TEXT"]) for _, row in doc_df.iterrows()]

    index = search.InvertedIndex()

    index.index(docs)

    query_df = pd.read_csv(query_file)
    query_df["QUERY"] = query_df["QUERY"].map(word_tokenize)
    query_df["DOC_ID"] = query_df["DOC_ID"].map(lambda x: list(map(int, x.split(":"))))

    results = index.query(query_df["QUERY"].to_list(), k=1)

    correct = 0
    for result, labels in zip(results, query_df["DOC_ID"]):
        if result[0][0] in labels:
            correct += 1

    acc = correct / len(query_df)
    assert acc >= 0.4
