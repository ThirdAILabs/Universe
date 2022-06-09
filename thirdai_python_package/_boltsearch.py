import math
import numpy as np
from thirdai._thirdai import bolt
from tqdm import tqdm

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
        num_epochs=10,
        batch_size=2048,
        learning_rate=0.01,
    ):
        self.groups = []
        self.num_points = len(dataset)
        train_dataset = dataset[: int(fraction_to_use_for_training * len(dataset))]
        current_assignments = [
            self._get_random_group_assignments(
                num_items_in_dataset=len(train_dataset), num_groups=self.num_classes
            )
            for _ in range(self.num_classifiers)
        ]

        for classifier_id, classifier in enumerate(self.classifiers):
            for epoch in range(num_epochs):
                classifier.train(
                    train_data=train_dataset,
                    train_labels=current_assignments[classifier_id],
                    loss_fn=bolt.CategoricalCrossEntropyLoss(),
                    learning_rate=learning_rate,
                    epochs=1,
                    batch_size=batch_size
                    # verbose=False,
                )

                predictions = classifier.predict(
                    train_dataset,
                    current_assignments[classifier_id],
                    2048,
                    ["categorical_accuracy"],
                )

                current_assignments[classifier_id] = self._get_new_group_assignments(
                    predicted_group_ids=predictions[1],
                    predicted_activations=predictions[2],
                    num_groups_to_consider=self.num_groups_to_consider
                )

            all_predictions = classifier.predict(dataset, None, 2048)
            all_assignments = self._get_new_group_assignments(
                predicted_group_ids=all_predictions[1],
                predicted_activations=all_predictions[2],
                num_groups_to_consider=1
            )
            print(all_assignments)
            print(all_predictions)
            group_memberships = [[] for _ in range(self.num_classes)]
            group_lens = [0 for _ in range(self.num_classes)]
            for vec_id, group_id in enumerate(all_assignments):
                group_memberships[group_id].append(vec_id)
                group_lens[group_id] += 1
            self.groups += group_memberships

            print(
                f"Min group size {min(group_lens)}, max group size {max(group_lens)}, std group size {np.std(group_lens)}"
            )

    def query(self, batch, top_k, threshold_to_return=-1):
        if threshold_to_return == -1:
            threshold_to_return = self.num_classifiers
        all_predictions = [
            classifier.predict(batch, None, 2048, verbose=False)
            for classifier in self.classifiers
        ]
        results = []
        for vec_id in tqdm(range(len(batch))):
            group_ids = np.concatenate(
                [
                    [self.num_classes * i + p for p in prediction[1][vec_id]]
                    for i, prediction in enumerate(all_predictions)
                ]
            )
            activations = np.concatenate(
                [prediction[2][vec_id] for prediction in all_predictions]
            )
            group_ids = group_ids[activations.argsort()[::-1]]
            results.append(
                self._group_testing_inference(group_ids, top_k, threshold_to_return)
            )
        return results

    def _group_testing_inference(self, group_ids, top_k, threshold):
        result = []
        point_counts = np.zeros(shape=(self.num_points,))
        for group_id in group_ids:
            print(group_id, self.groups[group_id])
            for point_id in self.groups[group_id]:
                point_counts[point_id] += 1
                if point_counts[point_id] == threshold:
                    result.append(point_id)
                    if len(result) == top_k:
                        return result
        return result

    def _create_single_dense_classifier(
        self,
        input_dim,
        last_layer_dim,
        last_layer_sparsity=0.025,
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

    def _get_new_group_assignments(self, predicted_group_ids, predicted_activations, num_groups_to_consider):
        new_group_sizes = [0 for _ in range(self.num_classes)]
        new_group_assignments = []

        for group_ids, activations in zip(predicted_group_ids, predicted_activations):

            groups_to_consider = group_ids[activations.argsort()[::-1]][
                : num_groups_to_consider
            ]
            group_counts = [new_group_sizes[group] for group in groups_to_consider]
            selected_group_id = groups_to_consider[np.argmin(group_counts)]

            new_group_sizes[selected_group_id] += 1
            new_group_assignments.append(selected_group_id)

        print(
            f"Min group size {min(new_group_sizes)}, max group size {max(new_group_sizes)}, std group size {np.std(new_group_sizes)}"
        )

        return np.array(new_group_assignments)
