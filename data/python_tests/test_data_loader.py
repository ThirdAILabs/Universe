import numpy as np
import pytest
from download_dataset_fixtures import download_clinc_dataset
from thirdai import bolt, data, dataset


def text_transformation():
    return data.transformations.Text(
        input_column="text", output_indices="indices", output_values="values"
    )


def load_data(filename):
    transformations = (
        data.transformations.Pipeline()
        .then(data.transformations.ToTokens("category", "category_id", dim=150))
        .then(text_transformation())
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
        shuffle=True,
        shuffle_buffer_size=1000,
    )

    return loader.all()


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


@pytest.fixture(scope="session")
def train_bolt_on_clinc(download_clinc_dataset):
    train_file, test_file, inference_samples = download_clinc_dataset

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

    return model, metrics, inference_samples


@pytest.mark.unit
def test_data_loader_clinc(train_bolt_on_clinc):
    _, metrics, _ = train_bolt_on_clinc

    assert metrics["val_categorical_accuracy"][-1] >= 0.85


# Checks that the features produced by the data loader are consistent with the
# features produced by directly transforming the input samples. This is accomplished
# by asserting that a model trained with the data loader performs well on inference
# samples that have been transformed directly.
@pytest.mark.unit
def test_single_sample_featurization(train_bolt_on_clinc):
    model, metrics, inference_samples = train_bolt_on_clinc

    tokenizer = text_transformation()

    correct = 0
    for sample, label in inference_samples:
        inputs = data.to_tensors(
            column_map=tokenizer(data.ColumnMap(sample)),
            columns_to_convert=[data.OutputColumns("indices", "values")],
            batch_size=100,
        )[0]

        output = model.forward(inputs)[0]
        pred = np.argmax(output.activations, axis=1)[0]
        if pred == label:
            correct += 1

    acc = correct / len(inference_samples)
    assert acc >= 0.85


# Checks that the features produced by the data loader are consistent with the
# features produced by directly transforming the input samples. This is accomplished
# by asserting that a model trained with the data loader performs well on inference
# samples that have been transformed directly.
@pytest.mark.unit
def test_batch_featurization(train_bolt_on_clinc):
    model, metrics, inference_samples = train_bolt_on_clinc

    tokenizer = text_transformation()

    batch = [x[0] for x in inference_samples]
    labels = np.array([x[1] for x in inference_samples])
    batch = data.to_tensors(
        column_map=tokenizer(data.ColumnMap(batch)),
        columns_to_convert=[data.OutputColumns("indices", "values")],
        batch_size=10000,
    )[0]

    output = model.forward(batch)[0]
    preds = np.argmax(output.activations, axis=1)

    acc = np.mean(preds == labels)
    assert np.mean(preds == labels) >= 0.85
