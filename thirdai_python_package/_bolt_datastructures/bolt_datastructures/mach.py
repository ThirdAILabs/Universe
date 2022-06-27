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
                    epochs=num_epochs_per_iteration,
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

    # def query_fast(self, batch, threshold=-1):
    #     if threshold == -1:
    #         threshold = self.num_classifiers
    #     num_vectors = len(batch[2]) - 1
    #     results = np.array(
    #         [
    #             classifier.predict(batch, None, 2048, verbose=True)[1]
    #             for classifier in self.classifiers
    #         ]
    #     )
    #     concatenated_results = np.concatenate(results)
    #     sorted_groups = np.argsort(concatenated_results, axis=1)[::-1]
    #     results = np.zeros(shape=(self.max_label,))
    #     count = np.zeros(shape=(num_vectors, self.max_label))
    #     for vec_id in tqdm(list(range(num_vectors))):
    #         for group in sorted_groups[vec_id]:
    #             for point in self.group_to_labels[group // self.max_label][
    #                 group % self.max_label
    #             ]:
    #                 count[vec_id, point] += 1
    #                 if count[vec_id, point] == threshold:
    #                     results[vec_id] = point
    #                     break
    #     return results

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
