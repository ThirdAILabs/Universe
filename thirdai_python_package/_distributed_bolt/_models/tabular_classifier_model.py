from typing import Dict, List
from thirdai._thirdai import bolt, dataset
import pandas as pd


class TabularClassifierModel:
    """This class implements the APIs to create, train and predict on a network
    which workers are running. Currently, It only supports Tabular Classifier.
    However, It could easily be extended to other models too. The functions
    defined here run on each of the node while distributing.
    """

    def __init__(
        self, config: Dict, total_nodes: int, num_layers: int, id: int, column_datatypes
    ):
        """Initailizes Model

        Args:
            config (Dict): Configuration File for the network
            total_nodes (int): Total number of workers
            layers (List[int]): array containing dimensions for each layer
            id (int): Model Id

        Raises:
            ValueError: Loading Dataset
        """
        self.num_layers = num_layers
        self.total_nodes = total_nodes

        self.train_file = config["dataset"]["train_file"][id]
        self.test_file = config["dataset"]["test_file"]
        self.prediction_file = config["dataset"]["prediction_file"]

        self.model_size = config["params"]["model_size"]
        self.n_classes = config["params"]["n_classes"]

        self.classifier = bolt.TabularClassifier(
            model_size=self.model_size, n_classes=self.n_classes
        )
        self.classifier.init_classifier_distributed_training(
            train_file=self.train_file,
            column_datatypes=column_datatypes,
            epochs=1,
            learning_rate=0.01,
        )
        self.distributed_training_context = (
            self.classifier.get_distributed_training_context()
        )
        self.classifier_model = self.classifier.get_bolt_graph_model()
        self.num_of_training_batches = self.distributed_training_context.numTrainingBatch()

    def calculate_gradients(self, batch_no: int):
        """This function trains the network and calculate gradients for the
            network of the model for the batch id, batch_no

        Args:
            batch_no (int): This function trains the network and calculate gradients for the
                network of the model for the batch id, batch_no
        """
        self.distributed_training_context.calculateGradientSingleNode(batch_no)

    def get_calculated_gradients(self):
        """Returns the calculated gradients.

        Returns:
            _type_: tuple of weight and bias gradients.
        """
        w_gradients = []
        b_gradients = []
        w_gradients.append(
            self.classifier_model.get_layer("fc_1").weight_gradients.get()
        )
        w_gradients.append(
            self.classifier_model.get_layer("fc_2").weight_gradients.get()
        )
        b_gradients.append(self.classifier_model.get_layer("fc_1").bias_gradients.get())
        b_gradients.append(self.classifier_model.get_layer("fc_2").bias_gradients.get())
        return (w_gradients, b_gradients)

    def set_gradients(self, w_gradients, b_gradients):
        """This function set the gradient in the current network with the updated
            gradients provided.

        Args:
            w_gradients __type__: weight gradients to update the network with
            b_gradients __type__: bias gradients to update the network with
        """
        for layer in range(len(w_gradients)):
            if layer == 0:
                self.classifier_model.get_layer("fc_1").weight_gradients.set(
                    w_gradients[layer]
                )
                self.classifier_model.get_layer("fc_1").bias_gradients.set(
                    b_gradients[layer]
                )
            elif layer == 1:
                self.classifier_model.get_layer("fc_2").weight_gradients.set(
                    w_gradients[layer]
                )
                self.classifier_model.get_layer("fc_2").bias_gradients.set(
                    b_gradients[layer]
                )
            else:
                raise ValueError(
                    "There should be only two layers for Tabular Classifier."
                )

    def get_parameters(self):
        """This function returns the weight and bias parameters from the network

        Returns:
            __type__: returns a tuple of weight and bias parameters
        """
        weights = []
        biases = []
        weights.append(self.classifier_model.get_layer("fc_1").weights.get())
        weights.append(self.classifier_model.get_layer("fc_2").weights.get())
        biases.append(self.classifier_model.get_layer("fc_1").biases.get())
        biases.append(self.classifier_model.get_layer("fc_2").biases.get())
        return weights, biases

    def set_parameters(self, weights, biases):
        """This function set the weight and bias parameter in the current network with
            the updated weights provided.

        Args:
            weights: weights parameter to update the network with
            biases: bias parameter to update the gradient with
        """
        for layer in range(len(weights)):
            if layer == 0:
                self.classifier_model.get_layer("fc_1").weights.set(weights[layer])
                self.classifier_model.get_layer("fc_1").biases.set(biases[layer])
            elif layer == 1:
                self.classifier_model.get_layer("fc_2").weights.set(weights[layer])
                self.classifier_model.get_layer("fc_2").biases.set(biases[layer])
            else:
                raise ValueError("Tabular Classifier should only have 2 layers")

    def update_parameters(self, learning_rate: float):
        """This function update the network parameters using the gradients stored and
            learning rate provided.

        Args:
            learning_rate (float): Learning Rate for the network
        """
        self.distributed_training_context.updateParametersSingleNode()

    def num_of_batches(self) -> int:
        """return the number of training batches present for this particular network

        Returns:
            int: number of batches
        """
        return self.num_of_training_batches

    def predict(self):
        """return the prediction for this particular network

        Returns:
            InferenceMetricData: tuple of matric and activations
        """
        self.distributed_training_context.finishTraining()
        self.classifier.predict(
            test_file=self.test_file, output_file=self.prediction_file
        )

        df = pd.read_csv(self.test_file)
        test_labels = list(df[df.columns[-1]])

        with open(self.prediction_file) as pred:
            pred_lines = pred.readlines()

        predictions = [x[:-1] for x in pred_lines]

        assert len(predictions) == len(test_labels)
        return sum(
            (prediction == answer)
            for (prediction, answer) in zip(predictions, test_labels)
        ) / len(predictions)
