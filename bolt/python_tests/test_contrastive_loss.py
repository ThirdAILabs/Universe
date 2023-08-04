import numpy as np
import pytest
from sklearn.metrics.pairwise import euclidean_distances
from thirdai import bolt, dataset

pytestmark = [pytest.mark.unit]


def get_contrastive_and_embedding_models(
    input_dim, embedding_dim, dissimilar_cutoff_distance
):
    input_1 = bolt.nn.Input(dim=input_dim)

    input_2 = bolt.nn.Input(dim=input_dim)

    output_op = bolt.nn.FullyConnected(
        dim=embedding_dim, input_dim=input_dim, activation="softmax"
    )

    output_1 = output_op(input_1)
    output_2 = output_op(input_2)

    labels = bolt.nn.Input(dim=1)

    contrastive_loss = bolt.nn.losses.EuclideanContrastive(
        output_1=output_1,
        output_2=output_2,
        labels=labels,
        dissimilar_cutoff_distance=dissimilar_cutoff_distance,
    )

    contrastive_model = bolt.nn.Model(
        inputs=[input_1, input_2],
        outputs=[output_1, output_2],
        losses=[contrastive_loss],
    )

    # TODO(Nick): Replace with noop loss
    emb_input = bolt.nn.Input(dim=input_dim)
    labels = bolt.nn.Input(dim=embedding_dim)
    embedding_output = output_op(emb_input)
    loss = bolt.nn.losses.CategoricalCrossEntropy(embedding_output, labels)
    embedding_model = bolt.nn.Model(
        inputs=[emb_input], outputs=[embedding_output], losses=[loss]
    )

    return (
        contrastive_model,
        embedding_model,
        contrastive_loss,
    )


def test_contrastive_number_embeddings():
    """
    This test sets up a task where the input to the network are one hot encoded
    integers. A number's embedding should be close to another number's embedding
    if they are in the same group. This is an easy test because none of the
    items are in the same group, so the model can just make items in the same
    group have the same embedding.
    """
    max_integer = 20
    groups = [0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 6, 6, 6, 7, 8, 8]
    inputs_are_similar = lambda i, j: groups[i] == groups[j]
    dissimilar_cutoff_distance = 1
    batch_size = 64
    embedding_dim = 50

    inputs_1 = np.zeros((max_integer * max_integer, max_integer), dtype="float32")
    inputs_2 = np.zeros((max_integer * max_integer, max_integer), dtype="float32")
    labels = np.zeros(max_integer * max_integer, dtype="float32")

    for i in range(max_integer):
        for j in range(max_integer):
            inputs_1[max_integer * i + j][i] = 1
            inputs_2[max_integer * i + j][j] = 1
            labels[max_integer * i + j] = inputs_are_similar(i, j)

    inputs_1 = dataset.from_numpy(inputs_1, batch_size=batch_size)
    inputs_2 = dataset.from_numpy(inputs_2, batch_size=batch_size)
    labels = dataset.from_numpy(labels, batch_size=batch_size)

    inputs = bolt.train.convert_datasets(
        [inputs_1, inputs_2], dims=[max_integer, max_integer]
    )
    labels = bolt.train.convert_dataset(labels, dim=1)

    (
        contrastive_model,
        embedding_model,
        contrastive_loss,
    ) = get_contrastive_and_embedding_models(
        max_integer,
        embedding_dim=embedding_dim,
        dissimilar_cutoff_distance=dissimilar_cutoff_distance,
    )

    trainer = bolt.train.Trainer(contrastive_model)

    history = trainer.train(
        train_data=[inputs, labels],
        learning_rate=0.1,
        epochs=200,
        train_metrics={
            "train_loss": bolt.train.metrics.LossMetric(contrastive_loss),
        },
    )

    identity = bolt.train.convert_dataset(
        dataset.from_numpy(
            np.identity(max_integer, dtype="float32"),
            batch_size=max_integer * max_integer,
        ),
        dim=max_integer,
    )[0]

    result = np.array(
        [r.activations for r in embedding_model.forward(identity, use_sparsity=False)]
    )[0]

    pairwise_distances = euclidean_distances(result, result)

    total_similar_distance = 0
    total_dissimilar_distance = 0
    num_similar = 0
    num_dissimilar = 0
    for i in range(max_integer):
        for j in range(max_integer):
            if i != j:
                if inputs_are_similar(i, j):
                    total_similar_distance += pairwise_distances[i][j]
                    num_similar += 1
                else:
                    total_dissimilar_distance += pairwise_distances[i][j]
                    num_dissimilar += 1

    average_similar_distance = total_similar_distance / num_similar
    average_dissimilar_distance = total_dissimilar_distance / num_dissimilar

    print(average_similar_distance, average_dissimilar_distance)
    assert average_similar_distance < average_dissimilar_distance
