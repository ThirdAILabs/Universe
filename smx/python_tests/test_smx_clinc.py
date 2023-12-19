from thirdai import smx, data, dataset
import time
import numpy as np
from download_dataset_fixtures import download_clinc_dataset
import pytest


def load_data(filename, shuffle):
    transformations = (
        data.transformations.Pipeline()
        .then(
            data.transformations.Text(
                input_column="text", output_indices="indices", output_values="values"
            )
        )
        .then(data.transformations.ToTokens("category", "category_id", dim=150))
    )

    data_iter = data.CsvIterator(
        data_source=dataset.FileDataSource(filename), delimiter=","
    )

    loader = data.Loader(
        data_iterator=data_iter,
        transformation=transformations,
        state=None,
        input_columns=[data.OutputColumns("indices", "values")],
        output_columns=[data.OutputColumns("category_id")],
        batch_size=2048,
        shuffle=shuffle,
    )

    return loader.all_smx()


@pytest.mark.unit
def test_smx_clinc(download_clinc_dataset):
    train_file, test_file, _ = download_clinc_dataset

    train_x, train_y = load_data(train_file, shuffle=True)
    val_x, val_y = load_data(test_file, shuffle=False)

    model = smx.Sequential(
        [
            smx.Embedding(n_embs=100000, emb_dim=512, reduce_mean=False),
            smx.Activation("relu"),
            smx.Linear(dim=150, input_dim=512),
        ]
    )

    optimizer = smx.optimizers.Adam(model.parameters(), lr=0.01)

    for epoch in range(5):
        s = time.perf_counter()
        for x, y in zip(train_x, train_y):
            out = model(x[0])

            loss = smx.cross_entropy(out, y[0])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        e = time.perf_counter()
        correct, total = 0, 0
        for x, y in zip(val_x, val_y):
            out = model(x[0]).tensor.numpy()

            np_y = y[0].numpy()
            correct += np.sum(np.argmax(out, axis=1) == np_y)
            total += len(np_y)

        accuracy = correct / total
        print(f"epoch {epoch} time={e-s}s val_accuracy={accuracy}\n")

    assert accuracy >= 0.85
