import math
import numpy as np
from thirdai._thirdai import bolt

# For now works on dense numpy vectors
class BoltSearch:
    def __init__(self, estimated_dataset_size, num_classifiers, input_dim):
        self.num_classes = int(math.sqrt(estimated_dataset_size))
        self.num_classifiers = num_classifiers
        self._classifiers = [
            self._create_single_dense_classifier(
                last_layer_dim=self.num_classes, input_dim=input_dim
            )
            for _ in range(num_classifiers)
        ]

    # TODO: Change num_epochs to just train until not converged
    def index(
        self, dataset, fraction_to_use_for_training=0.1, num_epochs=10, batch_size=2048, learning_rate=0.01
    ):
        train_dataset = dataset[: int(fraction_to_use_for_training * len(dataset))]
        current_assignments = [self._get_random_group_assignments(
            num_items_in_dataset=len(train_dataset), num_groups=self.num_classes
        ) for _ in range(self.num_classifiers)]

        for epoch in range(num_epochs):
            for i, classifier in enumerate(self._classifiers):
                classifier.train(
                    train_data=train_dataset,
                    train_labels=current_assignments[i],
                    loss_fn=bolt.CategoricalCrossEntropyLoss(),
                    learning_rate=learning_rate,
                    epochs=1,
                    batch_size=batch_size
                    # verbose=False,
                )
                predictions = classifier.predict(train_dataset, current_assignments[i], 1024, ["categorical_accuracy"])
                print(predictions)
        pass

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
