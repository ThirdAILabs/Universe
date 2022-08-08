from thirdai._thirdai import bolt, dataset
from .utils import create_fully_connected_layer_configs, load_dataset, initLogging


class Model:
    """
    This class implements the APIs to create, train and predict on a network
    which workers are running. Currently, It only supports FullyConnectedNetwork.
    However, It could easily be extended to other models too.

    Arguments:
        config: Configuration File for the network
        total_nodes: Total number of workers
        layers: array containing dimensions for each layer
    """

    def __init__(self, config, total_nodes, layers, id):
        self.layers = layers

        # getting training and testing data
        data = load_dataset(config, total_nodes, id)
        if data is None:
            raise ValueError("Unable to load a dataset. Please check the config")

        self.train_data, self.train_label, self.test_data, self.test_label = data

        # initializing Distributed Network
        self.bolt_layers = create_fully_connected_layer_configs(config["layers"])
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

    def calculateGradients(self, batch_no):
        """
        This function trains the network and calculate gradients for the
        network of the model for the batch id, batch_no

        Argument:
            batch_no: batch number of data to train the model
        """
        self.network.calculateGradientSingleNode(batch_no, self.loss)

    def getCalculatedGradients(self):
        """
        This function returns the gradients from the network.
        """
        w_gradients = []
        b_gradients = []
        for layer in range(len(self.layers) - 1):
            x = self.network.get_weights_gradients(layer)
            y = self.network.get_biases_gradients(layer)
            w_gradients.append(x)
            b_gradients.append(y)
        return (w_gradients, b_gradients)

    def setGradients(self, w_gradients, b_gradients):
        """
        This function set the gradient in the current network with the updated
        gradients provided.

        Arguments:
            w_gradients: weight gradients to update the network with
            b_gradients: bias gradients to update the network with
        """
        for layer in range(len(w_gradients)):
            self.network.set_weights_gradients(layer, w_gradients[layer])
            self.network.set_biases_gradients(layer, b_gradients[layer])

    def getParameters(self):
        """
        This function returns the weight and bias parameters from the network
        """
        weights = []
        biases = []
        for layer in range(len(self.layers) - 1):
            x = self.network.get_weights(layer)
            y = self.network.get_biases(layer)
            weights.append(x)
            biases.append(y)
        return weights, biases

    def setParameters(self, weights, biases):
        """
        This function set the weight and bias parameter in the current network with
        the updated weights provided.

        Arguments:
            weights: weights parameter to update the network with
            biases: bias parameter to update the gradient with
        """
        for layer in range(len(weights)):
            self.network.set_weights(layer, weights[layer])
            self.network.set_biases(layer, biases[layer])

    def updateParameters(self, learning_rate):
        """
        This function update the network parameters using the gradients stored and
        learning rate provided.

        Argument:
            learning_rate: Learning Rate for the network
        """
        self.network.updateParametersSingleNode(learning_rate)

    def num_of_batches(self):
        """
        return the number of training batches present for this particular network
        """

        return self.num_of_training_batches

    def predict(self):
        """
        return the prediction for this particular network
        """
        acc = self.network.predictSingleNode(
            self.test_data,
            self.test_label,
            False,
            ["categorical_accuracy"],
            verbose=False,
        )
        return acc
