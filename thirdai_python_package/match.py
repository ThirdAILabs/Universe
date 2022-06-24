import math
import numpy as np
from thirdai._thirdai import bolt
from tqdm import tqdm


class Match:
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
        # self.group_to_label = [
        #     [[] for _ in range(last_layer_dim)] for _ in range(num_classifiers)
        # ]
        # for classifier_id in range(num_classifiers):
        #     for label, group in enumerate(self.label_to_group[classifier_id]):
        #         self.group_to_label[classifier_id][group].append(label)

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

    def index(
        self,
        train_x,
        train_y,
        test_x,
        test_y,
        num_iterations=10,
        num_epochs_per_iteration=5,
        batch_size=2048,
        learning_rate=0.01,
    ):

        for iteration in range(num_iterations):
            for classifier_id, classifier in enumerate(self.classifiers):

                mapped_train_y = self.map_labels_to_groups(train_y, classifier_id)
                mapped_test_y = self.map_labels_to_groups(test_y, classifier_id)

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

                classifier.predict(
                    test_data=test_x,
                    test_labels=mapped_test_y,
                    metrics=["categorical_accuracy"]
                )

    def query(self, batch):
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
