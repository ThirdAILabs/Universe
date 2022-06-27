import math
import numpy as np
from thirdai._thirdai import bolt
from tqdm import tqdm


class Mach:
    def __init__(
        self,
        max_label,
        num_classifiers,
        input_dim,
        hidden_layer_dim,
        hidden_layer_sparsity,
        last_layer_dim,
        last_layer_sparsity,
        use_softmax,
    ):
        self.num_classifiers = num_classifiers
        self.max_label = max_label
        self.label_to_group = np.random.randint(
            0, last_layer_dim, size=(num_classifiers, max_label)
        )
        self.use_softmax = use_softmax
        self.group_to_labels = [
            [[] for _ in range(last_layer_dim)] for _ in range(num_classifiers)
        ]
        for classifier_id in range(num_classifiers):
            for label, group in enumerate(self.label_to_group[classifier_id]):
                self.group_to_labels[classifier_id][group].append(label)

        self.classifiers = [
            self._create_single_dense_classifier(
                use_softmax=use_softmax,
                input_dim=input_dim,
                hidden_layer_dim=hidden_layer_dim,
                hidden_layer_sparsity=hidden_layer_sparsity,
                last_layer_dim=last_layer_dim,
                last_layer_sparsity=last_layer_sparsity,
            )
            for _ in range(num_classifiers)
        ]

    def map_labels_to_groups(self, labels, classifier_id):
        if len(labels) != 3:
            raise ValueError(
                "Labels need to be in a sparse format (indices, values, offsets)"
            )
        labels_as_list = list(labels)
        group_mapper = np.vectorize(
            lambda label: self.label_to_group[classifier_id][label]
        )
        labels_as_list[0] = group_mapper(labels_as_list[0]).astype("uint32")
        return tuple(labels_as_list)

    def train(
        self,
        train_x,
        train_y,
        num_epochs=5,
        batch_size=512,
        learning_rate=0.001,
    ):

        for epoch in range(num_epochs):
            for classifier_id, classifier in enumerate(self.classifiers):

                mapped_train_y = self.map_labels_to_groups(train_y, classifier_id)

                classifier.train(
                    train_data=train_x,
                    train_labels=mapped_train_y,
                    loss_fn=(
                        bolt.CategoricalCrossEntropyLoss()
                        if self.use_softmax
                        else bolt.BinaryCrossEntropyLoss()
                    ),
                    learning_rate=learning_rate,
                    epochs=1,
                    batch_size=batch_size,
                    verbose=True,
                )

    def print_current_metrics(metrics=["categorical_accuracy"]):
        for classifier in self.classifiers:
            mapped_test_y = self.map_labels_to_groups(test_y, classifier_id)
            classifier.predict(
                test_data=test_x,
                test_labels=mapped_test_y,
                metrics=metrics,
            )

    def query_slow(self, batch):
        results = np.array(
            [
                classifier.predict(batch, None, 2048, verbose=True)[1]
                for classifier in self.classifiers
            ]
        )
        scores = np.zeros(shape=(len(batch[2]) - 1, self.max_label))
        for vec_id in tqdm(list(range(len(scores)))):
            for label in range(self.max_label):
                for classifier_id in range(self.num_classifiers):
                    scores[vec_id, label] += results[
                        classifier_id, vec_id, self.label_to_group[classifier_id, label]
                    ]
        return np.argmax(scores, axis=1)

    # TODO(josh): Can implement in C++ for way more speed
    # TODO(josh): Use better inference, this is equivalent to threshold = 1
    # TODO(josh): Allow returning top k instead of just top 1
    def query_fast(self, batch, m=10):
        num_elements = len(batch[2]) - 1
        results = np.array(
            [
                classifier.predict(batch, None, 2048, verbose=True)[1]
                for classifier in self.classifiers
            ]
        )
        top_m_groups = np.array([self._top_k_indices(arr, m) for arr in results])

        scores = np.zeros(shape=(num_elements, self.max_label))
        for vec_id in list(range(len(scores))):
            label_set = []
            for classifier_id in range(self.num_classifiers):
                for group in top_m_groups[classifier_id, vec_id]:
                    for label in self.group_to_labels[classifier_id][group]:
                        label_set.append(label)
            label_set = set(label_set)
            for classifier_id in range(self.num_classifiers):
                for label in label_set:
                    scores[vec_id, label] += results[
                        classifier_id, vec_id, self.label_to_group[classifier_id, label]
                    ]

        return np.argmax(scores, axis=1)

    def _top_k_indices(self, numpy_array, top_k):
        return np.argpartition(numpy_array, -top_k, axis=1)[:, -top_k:]

    def _create_single_dense_classifier(
        self,
        use_softmax,
        input_dim,
        last_layer_dim,
        last_layer_sparsity,
        hidden_layer_dim,
        hidden_layer_sparsity,
    ):
        layers = [
            bolt.FullyConnected(
                dim=hidden_layer_dim,
                sparsity=hidden_layer_sparsity,
                activation_function=bolt.ActivationFunctions.ReLU,
            ),
            bolt.FullyConnected(
                dim=last_layer_dim,
                sparsity=last_layer_sparsity,
                activation_function=(
                    bolt.ActivationFunctions.Softmax
                    if use_softmax
                    else bolt.ActivationFunctions.Sigmoid
                ),
            ),
        ]
        network = bolt.Network(layers=layers, input_dim=input_dim)
        return network
