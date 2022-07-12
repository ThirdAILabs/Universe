import math
import numpy as np
from thirdai._thirdai import bolt
import os
import pickle


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
        seed_for_group_assigments=0,
    ):

        self.num_classifiers = num_classifiers
        self.max_label = max_label
        self.input_dim = input_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.hidden_layer_sparsity = hidden_layer_sparsity
        self.last_layer_dim = last_layer_dim
        self.last_layer_sparsity = last_layer_sparsity
        self.use_softmax = use_softmax

        # setting a random seed
        np.random.seed(seed_for_group_assigments)

        self.label_to_group = np.random.randint(
            0, last_layer_dim, size=(num_classifiers, max_label)
        )

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

    def freeze_hash_tables(self):
        for classifiers in self.classifiers:
            classifiers.freeze_hash_tables()

    def save(self, folder, save_for_inference):

        if not os.path.exists(folder):
            os.mkdir(folder)

        if not os.path.isdir(folder):
            raise FileNotFoundError(f"{folder} is not a valid path to a directory")

        metadata = {
            "max_label": self.max_label,
            "num_classifiers": self.num_classifiers,
            "use_softmax": self.use_softmax,
            "input_dim": self.input_dim,
            "hidden_layer_dim": self.hidden_layer_dim,
            "hidden_layer_sparsity": self.hidden_layer_sparsity,
            "last_layer_dim": self.last_layer_dim,
            "last_layer_sparsity": self.last_layer_sparsity,
            "seed_for_group_assigments": self.seed_for_group_assigments,
        }

        with open(folder + "/metadata_mach", "wb") as f:
            pickle.dump(metadata, f)

        for classifiers_id in range(self.num_classifiers):
            if save_for_inference:
                self.classifiers[classifiers_id].save_for_inference(
                    f"{folder}/classifier_{classifiers_id}"
                )
            else:
                self.classifiers[classifiers_id].checkpoint(
                    f"{folder}/classifier_{classifiers_id}"
                )

    def checkpoint(self, folder):
        self.save(folder, save_for_inference=False)

    def save_for_inference(self, folder):
        self.save(folder, save_for_inference=True)

    def load(folder):

        if not os.path.exists(folder):
            raise FileNotFoundError(f"The passed in path {folder} does not exist")

        if not os.path.isdir(folder):
            raise Exception(
                f"{folder} is not a path to a folder that contains metadata and classifiers"
            )

        if not os.path.exists(folder + "/metadata_mach"):
            raise Exception("Metadata not found for the mach model")

        with open(folder + "/metadata_mach", "rb") as f:
            metadata = pickle.load(f)

        newMach = Mach(
            max_label=metadata["max_label"],
            num_classifiers=metadata["num_classifiers"],
            input_dim=metadata["input_dim"],
            hidden_layer_dim=metadata["hidden_layer_dim"],
            hidden_layer_sparsity=metadata["hidden_layer_sparsity"],
            last_layer_dim=metadata["last_layer_dim"],
            last_layer_sparsity=metadata["last_layer_sparsity"],
            use_softmax=metadata["use_softmax"],
            seed_for_group_assigments=metadata["seed_for_group_assigments"],
        )

        newMach.classifiers = []
        for i in range(newMach.num_classifiers):
            if not os.path.exists(folder + f"/classifier_{i}"):
                raise Exception(
                    f"Could not find the {i}th classifier for the mach model inside the folder {folder}"
                )
            newMach.classifiers.append(bolt.Network.load(folder + f"/classifier_{i}"))

        return newMach

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
        learning_rate,
        num_epochs,
        batch_size,
    ):

        loss_func = (
            bolt.CategoricalCrossEntropyLoss()
            if self.use_softmax
            else bolt.BinaryCrossEntropyLoss()
        )

        for epoch in range(num_epochs):
            for classifier_id, classifier in enumerate(self.classifiers):

                mapped_train_y = self.map_labels_to_groups(train_y, classifier_id)

                classifier.train(
                    train_data=train_x,
                    train_labels=mapped_train_y,
                    loss_fn=loss_func,
                    learning_rate=learning_rate,
                    epochs=1,
                    batch_size=batch_size,
                    verbose=True,
                )

    def query_slow(self, batch):
        results = np.array(
            [
                classifier.predict(batch, test_labels=None, verbose=False)[1]
                for classifier in self.classifiers
            ]
        )
        num_query_results = results.shape[1]
        scores = np.zeros(shape=(num_query_results, self.max_label))
        for vec_id in range(num_query_results):
            for label in range(self.max_label):
                for classifier_id in range(self.num_classifiers):
                    scores[vec_id, label] += results[
                        classifier_id, vec_id, self.label_to_group[classifier_id, label]
                    ]
        return np.argmax(scores, axis=1)

    # TODO(josh): Can implement in C++ for way more speed
    # TODO(josh): Use better inference, this is equivalent to threshold = 1
    # TODO(josh): Allow returning top k instead of just top 1
    def query_fast(self, batch, num_groups_to_check_per_classifier=10):
        results = np.array(
            [
                classifier.predict(batch, test_labels=None, verbose=False)[1]
                for classifier in self.classifiers
            ]
        )
        top_m_groups = np.array(
            [
                self._top_k_indices(arr, num_groups_to_check_per_classifier)
                for arr in results
            ]
        )
        num_query_results = results.shape[1]

        scores = np.zeros(shape=(num_query_results, self.max_label))
        for vec_id in range(num_query_results):

            label_set = set()
            for classifier_id in range(self.num_classifiers):
                for group in top_m_groups[classifier_id, vec_id]:
                    for label in self.group_to_labels[classifier_id][group]:
                        label_set.add(label)

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
        last_layer_act_func = (
            bolt.ActivationFunctions.Softmax
            if use_softmax
            else bolt.ActivationFunctions.Sigmoid
        )
        layers = [
            bolt.FullyConnected(
                dim=hidden_layer_dim,
                sparsity=hidden_layer_sparsity,
                activation_function=bolt.ActivationFunctions.ReLU,
            ),
            bolt.FullyConnected(
                dim=last_layer_dim,
                sparsity=last_layer_sparsity,
                activation_function=last_layer_act_func,
            ),
        ]
        network = bolt.Network(layers=layers, input_dim=input_dim)
        return network
