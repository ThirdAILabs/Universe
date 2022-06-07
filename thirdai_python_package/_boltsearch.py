import math
import numpy as np
from thirdai._thirdai import bolt

# For now works on dense numpy vectors
class BoltSearch:
    def __init__(
        self,
        estimated_dataset_size,
        num_classifiers,
        input_dim,
        num_groups_to_consider=5,
    ):
        self.num_classes = int(math.sqrt(estimated_dataset_size))
        self.num_classifiers = num_classifiers
        self.num_groups_to_consider = num_groups_to_consider

        self.classifiers = [
            self._create_single_dense_classifier(
                last_layer_dim=self.num_classes, input_dim=input_dim
            )
            for _ in range(num_classifiers)
        ]
        for classifier in self.classifiers:
            classifier.enable_sparse_inference(remember_mistakes=False)

    # TODO: Change num_epochs to just train until not converged
    def index(
        self,
        dataset,
        fraction_to_use_for_training=0.1,
        num_epochs=100,
        batch_size=2048,
        learning_rate=0.01,
    ):
        train_dataset = dataset[: int(fraction_to_use_for_training * len(dataset))]
        current_assignments = [
            self._get_random_group_assignments(
                num_items_in_dataset=len(train_dataset), num_groups=self.num_classes
            )
            for _ in range(self.num_classifiers)
        ]

        for epoch in range(num_epochs):
            for i, classifier in enumerate(self.classifiers):
                classifier.train(
                    train_data=train_dataset,
                    train_labels=current_assignments[i],
                    loss_fn=bolt.CategoricalCrossEntropyLoss(),
                    learning_rate=learning_rate,
                    epochs=1,
                    batch_size=batch_size
                    # verbose=False,
                )

                predictions = classifier.predict(
                    train_dataset,
                    current_assignments[i],
                    1024,
                    ["categorical_accuracy"],
                )

                new_group_sizes = [0 for _ in range(self.num_classes)]
                new_group_assignments = []

                all_indices = predictions[1]
                all_values = predictions[2]
                for indices, values in zip(all_indices, all_values):

                    groups_to_consider = indices[values.argsort()[::-1]][
                        : self.num_groups_to_consider
                    ]
                    group_counts = [
                        new_group_sizes[group] for group in groups_to_consider
                    ]
                    min_group_index = groups_to_consider[np.argmin(group_counts)]

                    new_group_sizes[min_group_index] += 1
                    new_group_assignments.append(min_group_index)

                current_assignments[i] = np.array(new_group_assignments)

                print(
                    min(new_group_sizes),
                    max(new_group_sizes),
                    np.std(new_group_sizes),
                    len(train_dataset) / self.num_classes,
                )

    def query(self, batch):
        pass

    def _create_single_dense_classifier(
        self,
        input_dim,
        last_layer_dim,
        last_layer_sparsity=0.01,
        hidden_layer_dim=10000,
        hidden_layer_sparsity=0.01,
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
                activation_function=bolt.ActivationFunctions.Softmax,
            ),
        ]
        return bolt.Network(layers=layers, input_dim=input_dim)

    def _get_random_group_assignments(self, num_items_in_dataset, num_groups):
        return np.random.randint(low=0, high=num_groups, size=(num_items_in_dataset,))
        # groups = [[] for _ in range(num_groups)]
        # for i in range(num_items_in_dataset):
        #     groups[group_assignments[i]].append(i)
        # return group_assignments, groups
