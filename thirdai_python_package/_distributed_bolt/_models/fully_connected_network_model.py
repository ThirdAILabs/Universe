from typing import Dict, List
from thirdai._thirdai import bolt, dataset
from thirdai._distributed_bolt.utils import load_dataset, make_layers_from_config


class FullyConnectedNetworkModel:
    """This class implements the APIs to create, train and predict on a network
    which workers are running. Currently, It only supports FullyConnectedNetwork.
    However, It could easily be extended to other models too. The functions
    defined here run on each of the node while distributing.
    """

    def __init__(self, config: Dict, total_nodes: int, num_layers: int, id: int):
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

        # getting training and testing data
        data = load_dataset(config, total_nodes, id)
        if data is None:
            raise ValueError("Unable to load a dataset. Please check the config")

        self.train_data, self.train_label, self.test_data, self.test_label = data

        # initializing Distributed Network
        self.bolt_layers = make_layers_from_config(config["layers"])
        self.input_dim = config["dataset"]["input_dim"]
        self.network = bolt.DistributedNetwork(
            layers=self.bolt_layers, input_dim=self.input_dim
        )

        # get variables for initializing training
        self.rehash = config["params"]["rehash"]
        self.rebuild = config["params"]["rebuild"]
        if config["params"]["loss_fn"].lower() == "categoricalcrossentropyloss":
            self.loss = bolt.CategoricalCrossEntropyLoss()
        elif config["params"]["loss_fn"].lower() == "meansquarederror":
            self.loss = bolt.MeanSquaredError()
        else:
            print(
                "'{}' is not a valid loss function".format(config["params"]["loss_fn"])
            )

        # prepare node for training
        self.num_of_training_batches = self.network.prepareNodeForDistributedTraining(
            self.train_data,
            self.train_label,
            rehash=self.rehash,
            rebuild=self.rebuild,
            verbose=False,
        )

        self.test_metrics = config["params"]["test_metrics"]

    def calculate_gradients(self, batch_no: int):
        """This function trains the network and calculate gradients for the
            network of the model for the batch id, batch_no

        Args:
            batch_no (int): This function trains the network and calculate gradients for the
                network of the model for the batch id, batch_no
        """
        self.network.calculateGradientSingleNode(batch_no, self.loss)

    def get_calculated_gradients(self):
        """Returns the calculated gradients.

        Returns:
            _type_: tuple of weight and bias gradients.
        """
        w_gradients = []
        b_gradients = []
        for layer in range(self.num_layers - 1):
            x = self.network.get_weights_gradients(layer)
            y = self.network.get_biases_gradients(layer)
            w_gradients.append(x)
            b_gradients.append(y)
        return (w_gradients, b_gradients)

    def set_gradients(self, w_gradients, b_gradients):
        """This function set the gradient in the current network with the updated
            gradients provided.

        Args:
            w_gradients __type__: weight gradients to update the network with
            b_gradients __type__: bias gradients to update the network with
        """
        for layer in range(len(w_gradients)):
            self.network.set_weights_gradients(layer, w_gradients[layer])
            self.network.set_biases_gradients(layer, b_gradients[layer])

    def get_parameters(self):
        """This function returns the weight and bias parameters from the network

        Returns:
            __type__: returns a tuple of weight and bias parameters
        """
        weights = []
        biases = []
        for layer in range(self.num_layers - 1):
            x = self.network.get_weights(layer)
            y = self.network.get_biases(layer)
            weights.append(x)
            biases.append(y)
        return weights, biases

    def set_parameters(self, weights, biases):
        """This function set the weight and bias parameter in the current network with
            the updated weights provided.

        Args:
            weights: weights parameter to update the network with
            biases: bias parameter to update the gradient with
        """
        for layer in range(len(weights)):
            self.network.set_weights(layer, weights[layer])
            self.network.set_biases(layer, biases[layer])

    def update_parameters(self, learning_rate: float):
        """This function update the network parameters using the gradients stored and
            learning rate provided.

        Args:
            learning_rate (float): Learning Rate for the network
        """
        self.network.updateParametersSingleNode(learning_rate)

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
        acc = self.network.predictSingleNode(
            self.test_data,
            self.test_label,
            False,
            self.test_metrics,
            verbose=False,
        )
        return acc
