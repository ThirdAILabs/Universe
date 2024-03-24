from download_dataset_fixtures import download_scifact_dataset
from thirdai import bolt, data
import pandas as pd
import os
from mach_retriever_utils import train_simple_mach_retriever, QUERY_FILE
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.release]


def check_search_accuracy(model, supervised_tst):
    df = pd.read_csv(supervised_tst)
    correct = 0
    for _, row in df.iterrows():
        labels = set(map(int, row["DOC_ID"].split(":")))
        preds = model.search(row["QUERY"], top_k=1)

        if preds[0][0] in labels:
            correct += 1

    assert correct / len(df) >= 0.55


def test_mach_retriever_scifact(download_scifact_dataset):
    unsupervised_file, supervised_trn, supervised_tst, _ = download_scifact_dataset

    model = (
        bolt.Mach()
        .tokenizer("words")
        .emb_dim(1024)
        .n_buckets(1000)
        .emb_bias()
        .output_bias()
        .output_activation("sigmoid")
        .build()
    )

    model.coldstart(
        unsupervised_file,
        strong_cols=["TITLE"],
        weak_cols=["TEXT"],
        learning_rate=1e-3,
        epochs=5,
        metrics=["hash_precision@5"],
    )

    model.train(
        supervised_trn,
        learning_rate=1e-3,
        epochs=10,
        metrics=["hash_precision@5"],
    )

    metrics = model.evaluate(
        supervised_tst,
        metrics=["precision@1", "recall@5"],
    )

    assert metrics["val_precision@1"][-1] >= 0.58

    check_search_accuracy(model, supervised_tst)

    save_path = "./mach_retriever"
    model.save(save_path)
    model = bolt.MachRetriever.load(save_path)
    os.remove(save_path)

    check_search_accuracy(model, supervised_tst)

    model.train(
        supervised_trn,
        learning_rate=1e-3,
        epochs=1,
        metrics=["hash_precision@5"],
    )

    metrics = model.evaluate(
        supervised_tst,
        metrics=["precision@1", "recall@5"],
    )

    assert metrics["val_precision@1"][-1] >= 0.58


def test_mach_retriever_introduce_delete():
    model = train_simple_mach_retriever()

    model.erase(list(range(20)))

    assert model.index.num_entities() == 80

    model.clear()

    assert model.index.num_entities() == 0

    model.introduce(
        data.TransformedIterator(
            data.CsvIterator(QUERY_FILE),
            data.transformations.ToTokens("id", "id"),
        ),
        strong_cols=["text"],
        weak_cols=[],
        text_augmentation=False,
        load_balancing=False,
    )

    assert model.index.num_entities() == 100

    df = pd.read_csv(QUERY_FILE)

    correct = 0
    for i, query in enumerate(df["text"]):
        if model.search(query, top_k=1)[0][0] == i:
            correct += 1

    assert correct / len(df) >= 0.95


def test_mach_retriever_ranking():
    model = train_simple_mach_retriever()

    df = pd.read_csv(QUERY_FILE)

    for query in df["text"]:
        search_results = model.search(query, top_k=8)

        expected_rank_results = [
            search_results[i] for i in range(0, len(search_results), 2)
        ]
        candidates = set([x[0] for x in expected_rank_results])

        rank_results = model.rank(query, candidates, top_k=8)
        assert len(rank_results) == 4
        assert rank_results == expected_rank_results
