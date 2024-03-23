from download_dataset_fixtures import download_scifact_dataset
from thirdai import bolt
import pandas as pd


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

    df = pd.read_csv(supervised_tst)
    correct = 0
    for _, row in df.iterrows():
        labels = set(map(int, row["DOC_ID"].split(":")))
        preds = model.search(row["QUERY"], top_k=1)

        if preds[0][0] in labels:
            correct += 1

    assert correct / len(df) >= 0.55
