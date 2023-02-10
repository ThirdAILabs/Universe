from thirdai import bolt, dataset
import numpy as np
import sklearn.metrics
import sys


class DLRM:
    def __init__(self, num_int_features: int, num_cat_features: int, chunk_size: int):
        int_input = bolt.nn.Input(dim=num_int_features)
        hidden1 = bolt.nn.FullyConnected(dim=32, activation="relu")(int_input)

        cat_input = bolt.nn.TokenInput(
            dim=4294967295, num_tokens_range=(num_cat_features, num_cat_features)
        )

        embedding = bolt.nn.Embedding(
            num_embedding_lookups=8,
            lookup_size=4,
            log_embedding_block_size=29,
            chunk_size=chunk_size,
            reduction="concat",
            num_tokens_per_input=num_cat_features,
        )(cat_input)

        feature_interaction = bolt.nn.DlrmAttention()(
            fc_layer=hidden1, embedding_layer=embedding
        )

        concat = bolt.nn.Concatenate()([hidden1, feature_interaction])

        hidden_output = concat
        for _ in range(3):
            hidden_output = bolt.nn.FullyConnected(
                dim=500,
                sparsity=0.4,
                activation="relu",
                sampling_config=bolt.nn.RandomSamplingConfig(),
            )(hidden_output)

        output = bolt.nn.FullyConnected(dim=1, activation="sigmoid")(hidden_output)

        self.model = bolt.nn.Model(inputs=[int_input, cat_input], output=output)
        self.model.compile(bolt.nn.losses.BinaryCrossEntropy())

    def train(
        self,
        x_int: np.ndarray,
        x_cat: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        epochs: int = 1,
        learning_rate: float = 0.0001,
    ):
        assert x_int.dtype == np.float32
        assert x_cat.dtype == np.uint32
        assert y.dtype == np.float32
        assert len(y.shape) == 1 or y.shape[1] == 1

        x_int_dataset = dataset.from_numpy(x_int, batch_size=batch_size)
        x_cat_dataset = dataset.from_numpy(x_cat, batch_size=batch_size)
        y_dataset = dataset.from_numpy(y, batch_size=batch_size)

        train_cfg = bolt.TrainConfig(learning_rate=learning_rate, epochs=epochs)

        metrics = self.model.train(
            train_data=[x_int_dataset, x_cat_dataset],
            train_labels=y_dataset,
            train_config=train_cfg,
        )

        return metrics["epoch_times"][0]

    def predict(self, x_int: np.ndarray, x_cat: np.ndarray) -> np.ndarray:
        assert x_int.dtype == np.float32
        assert x_cat.dtype == np.uint32

        x_int_dataset = dataset.from_numpy(x_int, batch_size=2048)
        x_cat_dataset = dataset.from_numpy(x_cat, batch_size=2048)

        eval_cfg = bolt.EvalConfig().return_activations()

        _, activations = self.model.evaluate(
            test_data=[x_int_dataset, x_cat_dataset],
            test_labels=None,
            predict_config=eval_cfg,
        )

        assert activations.shape == (len(x_int), 1)

        return activations[:, 0]


def featurize_int_data(x_int: np.ndarray) -> np.ndarray:
    return np.log(x_int + 1)


def load_data(filename):
    data = np.load(filename)

    X_cat = data["X_cat"].astype(np.uint32)
    X_int = data["X_int"].astype(np.float32)
    y = data["y"].astype(np.float32)
    counts = data["counts"]
    print("Data loaded", flush=True)

    temp = np.zeros(len(counts) + 1, dtype=np.uint32)
    temp[1:] = np.cumsum(counts).astype(np.uint32)
    temp += X_int.shape[1]
    counts = temp[:-1]

    idxs = np.arange(y.shape[0])
    np.random.shuffle(idxs)

    n_train = int(len(idxs) * 0.8)

    X_int = featurize_int_data(X_int)
    X_cat += counts

    print("Data featurized", flush=True)

    train_idxs = idxs[:n_train]
    test_idxs = idxs[n_train:]

    X_cat_train = X_cat[train_idxs]
    X_cat_test = X_cat[test_idxs]

    X_int_train = X_int[train_idxs]
    X_int_test = X_int[test_idxs]

    y_train = y[train_idxs]
    y_test = y[test_idxs]

    print("Train/test split created", flush=True)

    return ((X_int_train, X_cat_train, y_train), (X_int_test, X_cat_test, y_test))


results = []

def main():
    (X_int_train, X_cat_train, y_train), (X_int_test, X_cat_test, y_test) = load_data(
        sys.argv[1]
    )

    print("X_int_train: ", X_int_train.shape)
    print("X_cat_train: ", X_cat_train.shape)
    print("y_train: ", y_train.shape)
    print("X_int_test: ", X_int_test.shape)
    print("X_cat_test: ", X_cat_test.shape)
    print("y_test: ", y_test.shape)


    # for cs in [1] + list(range(2, 51, 2)):
    for cs in [1, 2, 4, 8, 16, 32, 64, 128]:
        model = DLRM(
            num_int_features=X_int_test.shape[1],
            num_cat_features=X_cat_test.shape[1],
            chunk_size=cs,
        )

        time = model.train(x_int=X_int_train, x_cat=X_cat_train, y=y_train, batch_size=512)

        scores = model.predict(x_int=X_int_test, x_cat=X_cat_test)

        roc_auc = sklearn.metrics.roc_auc_score(y_test, scores)

        print(f"Chunk size={cs}, time={time}, roc auc={roc_auc}")

        results.append((cs, time, roc_auc))


if __name__ == "__main__":

    main()

    print(results)
