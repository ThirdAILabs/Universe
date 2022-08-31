from thirdai import bolt, dataset
import numpy as np
import pytest
import time

pytestmark = [pytest.mark.unit]

def build_train_and_predict(values_np, indices_np, offsets_np, labels_np, input_output_dim, output_sparsity=0.15, batch_size=64):
    print(output_sparsity)
    data = dataset.from_numpy((indices_np, values_np, offsets_np), batch_size=batch_size)
    labels = dataset.from_numpy(labels_np, batch_size=batch_size)

    input_layer = bolt.graph.Input(dim=input_output_dim)
    output_layer = bolt.graph.FullyConnected(
        dim=input_output_dim, activation="softmax", sparsity=output_sparsity,
        sampling_config=bolt.DWTASamplingConfig(
            hashes_per_table=3,
            num_tables=64,
            reservoir_size=8
    )
    )(input_layer)
    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.CategoricalCrossEntropyLoss())

    model.summary(detailed=True)

    train_config = bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=3).silence()

    start = time.time()
    model.train(data, labels, train_config)
    print("Training time: ", time.time() - start)

    predict_config = (
        bolt.graph.PredictConfig.make()
        .with_metrics(["categorical_accuracy"])
        .return_activations()
        .silence()
    )
    return model.predict(data, labels, predict_config)

def test_sparse_sparse_above_threshold():
    input_output_dim = 1000
    input_num_nonzeros = 1
    num_examples = 10000
    values_np = np.ones(input_num_nonzeros * num_examples).astype("float32")
    indices_np = np.random.randint(input_output_dim, size=input_num_nonzeros * num_examples).astype("uint32")
    offsets_np = np.arange(0, (num_examples + 1) * input_num_nonzeros, input_num_nonzeros).astype("uint32")
    labels_np = np.reshape(indices_np, (-1, 1))
    res = build_train_and_predict(values_np, indices_np, offsets_np, labels_np, input_output_dim)

    print(res[0]['categorical_accuracy'])



test_sparse_sparse_above_threshold()