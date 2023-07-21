import pytest
from download_dataset_fixtures import download_clinc_dataset
from thirdai import bolt_v2 as bolt
from thirdai import data, dataset


def load_data(filename):
    transformations = data.transformations.TransformationList(
        [
            data.transformations.ToTokens("category", "category_id", dim=150),
            data.transformations.Text(input_column="text", output_column="tokens"),
        ]
    )

    data_iter = data.ColumnMapIterator(
        data_source=dataset.FileDataSource(filename), delimiter=","
    )

    loader = data.Loader(
        data_iterator=data_iter,
        transformation=transformations,
        input_columns=[("tokens", None)],
        output_columns=[("category_id", None)],
        batch_size=2048,
    )

    return loader.next()


def build_model():
    input_layer = bolt.nn.Input(dim=100000)

    hidden = bolt.nn.Embedding(dim=512, input_dim=100000, activation="relu")(
        input_layer
    )

    output = bolt.nn.FullyConnected(
        dim=150, input_dim=hidden.dim(), activation="softmax"
    )(hidden)

    loss = bolt.nn.losses.CategoricalCrossEntropy(output, labels=bolt.nn.Input(dim=150))

    return bolt.nn.Model(inputs=[input_layer], outputs=[output], losses=[loss])


@pytest.mark.unit
def test_data_loader_clinc(download_clinc_dataset):
    train_file, test_file, _ = download_clinc_dataset

    train_data = load_data(train_file)
    val_data = load_data(test_file)

    model = build_model()

    trainer = bolt.train.Trainer(model)

    metrics = trainer.train(
        train_data=train_data,
        epochs=5,
        learning_rate=0.01,
        validation_data=val_data,
        validation_metrics=["categorical_accuracy"],
    )

    assert metrics["val_categorical_accuracy"][-1] >= 0.85
