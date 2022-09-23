from typing import Dict, List, Optional
from thirdai._thirdai import bolt, dataset
from thirdai._distributed_bolt.utils import load_train_test_data
from ..utils import contruct_dag_model


class FullyConnectedNetworkSingleNode:
    """
    This class implements the APIs to create, train and predict on a network
    which workers are running. Currently, It only supports FullyConnectedNetwork.
    However, It could easily be extended to other models too. The functions
    defined here run on each of the node while distributing.

    """

    def __init__(self, config: Dict, total_nodes: int, layer_dims: List[int], id: int):
        """

        :param config: Configuration File for the network
        :type config: Dict
        :param total_nodes: Total number of workers
        :type total_nodes: int
        :param layer_dims: array containing dimensions for each layer
        :type layer_dims: List[int]
        :param id: Model Id
        :type id: int
        """
        self.layer_dims = layer_dims

        (
            self.train_data,
            self.train_label,
            self.test_data,
            self.test_label,
        ) = load_train_test_data(config, total_nodes, id)

        self.rehash = config["params"]["rehash"]
        self.rebuild = config["params"]["rebuild"]
        self.learning_rate = config["params"]["learning_rate"]
        self.epochs = config["params"]["epochs"]

        if config["params"]["loss_fn"].lower() == "categoricalcrossentropyloss":
            self.loss = bolt.CategoricalCrossEntropyLoss()
        elif config["params"]["loss_fn"].lower() == "meansquarederror":
            self.loss = bolt.MeanSquaredError()
        else:
            print(
                "'{}' is not a valid loss function".format(config["params"]["loss_fn"])
            )

        train_config = (
            bolt.graph.TrainConfig.make(
                learning_rate=self.learning_rate, epochs=self.epochs
            )
            .silence()
            .with_rebuild_hash_tables(self.rehash)
            .with_reconstruct_hash_functions(self.rebuild)
        )

        inputs, output_node = contruct_dag_model(config)
        self.network = bolt.graph.DistributedModel(
            inputs=inputs,
            output=output_node,
            train_data=[self.train_data],
            train_labels=self.train_label,
            train_config=train_config,
            loss=self.loss,
        )
        self.test_metrics = config["params"]["test_metrics"]
        self.node_name_list = []
        for i in range(len(self.layer_dims) - 1):
            self.node_name_list.append("fc_" + str(i + 1))

    def calculate_gradients(self, batch_no: int):
        """
        This function trains the network and calculate gradients for the
        network of the model for the batch id, batch_no

        :param batch_no: This function trains the network and calculate gradients for the
                network of the model for the batch id, batch_no
        :type batch_no: int
        """
        self.network.calculateGradientSingleNode(batch_no)

    def get_calculated_gradients(self):
        """
        Returns the calculated gradients.

        :return: tuple of weight and bias gradients.
        :rtype: Tuple
        """
        w_gradients = []
        b_gradients = []
        for node_id in range(len(self.node_name_list)):
            x = self.network.get_layer(
                self.node_name_list[node_id]
            ).weight_gradients.copy()
            y = self.network.get_layer(
                self.node_name_list[node_id]
            ).bias_gradients.copy()
            w_gradients.append(x)
            b_gradients.append(y)
        return (w_gradients, b_gradients)

    def set_gradients(self, w_gradients, b_gradients):
        """
        This function set the gradient in the current network with the updated
            gradients provided.

        :param w_gradients: weight gradients to update the network with
        :type w_gradients: numpy.ndarray
        :param b_gradients: bias gradients to update the network with
        :type b_gradients: numpy.ndarray
        """
        for layer_num in range(len(w_gradients)):
            self.network.get_layer(self.node_name_list[layer_num]).weight_gradients.set(
                w_gradients[layer_num]
            )
            self.network.get_layer(self.node_name_list[layer_num]).bias_gradients.set(
                b_gradients[layer_num]
            )

    def get_parameters(self):
        """
        This function returns the weight and bias parameters from the network

        :return: returns a tuple of weight and bias parameters
        :rtype: Tuple(numpy.ndarray, numpy.ndarray)
        """
        weights = []
        biases = []
        for node_id in range(len(self.node_name_list)):
            x = self.network.get_layer(self.node_name_list[node_id]).weights.copy()
            y = self.network.get_layer(self.node_name_list[node_id]).biases.copy()
            weights.append(x)
            biases.append(y)
        return weights, biases

    def set_parameters(self, weights, biases):
        """
        This function set the weight and bias parameter in the current network with
        the updated weights provided.

        :param weights: weights parameter to update the network with
        :type weights: numpy.ndarray
        :param biases: bias parameter to update the gradient with
        :type biases: numpy.ndarray
        """
        for layer_num in range(len(weights)):
            self.network.get_layer(self.node_name_list[layer_num]).weights.set(
                weights[layer_num]
            )
            self.network.get_layer(self.node_name_list[layer_num]).biases.set(
                biases[layer_num]
            )

    def update_parameters(self, learning_rate: float):
        """
        This function update the network parameters using the gradients stored and
            learning rate provided.

        :param learning_rate: Learning Rate for the network
        :type learning_rate: float
        """
        self.network.updateParametersSingleNode()

    def num_of_batches(self) -> int:
        """
        return the number of training batches present for this particular network

        :return: number of batches
        :rtype: int
        """
        return self.network.numTrainingBatch()

    def finish_training(self):
        self.network.finishTraining()

    def predict(self):
        """
        return the prediction for this particular network

        :return: tuple of metric and activations
        :rtype: _type_
        """
        predict_config = (
            bolt.graph.PredictConfig.make().with_metrics(self.test_metrics).silence()
        )
        return self.network.predict(
            test_data=self.test_data,
            test_labels=self.test_label,
            predict_config=predict_config,
        )
