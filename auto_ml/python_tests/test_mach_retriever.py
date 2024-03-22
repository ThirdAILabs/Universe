from download_dataset_fixtures import download_scifact_dataset
from thirdai import bolt, data


def test_mach_retriever_scifact(download_scifact_dataset):
    unsupervised_file, supervised_trn, supervised_tst, _ = download_scifact_dataset

    def get_iterator(path):
        return data.TransformedIterator(
            data.CsvIterator(path),
            data.transformations.ToTokenArrays("DOC_ID", "DOC_ID", ":", 0xFFFFFFFF),
        )

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
        get_iterator(unsupervised_file),
        strong_cols=["TITLE"],
        weak_cols=["TEXT"],
        learning_rate=1e-3,
        epochs=5,
        metrics=["hash_precision@5"],
    )

    model.train(
        get_iterator(supervised_trn),
        learning_rate=1e-3,
        epochs=10,
        metrics=["hash_precision@5"],
    )

    metrics = model.evaluate(
        get_iterator(supervised_tst),
        metrics=["precision@1", "recall@5"],
    )

    assert metrics["val_precision@1"][-1] >= 0.58
