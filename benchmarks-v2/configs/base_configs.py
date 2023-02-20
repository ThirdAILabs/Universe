from abc import ABC, abstractmethod


class UDTBenchmarkConfig:
    """
    Default config for UDT models.
    """

    learning_rate = None
    num_epochs = None
    target = "label"
    n_target_classes = None
    delimiter = ","
    metric_type = "categorical_accuracy"
    model_config = None
    model_config_path = None
    callbacks = []


class BoltBenchmarkConfig(ABC):
    """
    Default config for two-layered Bolt models.
    """

    loss_fn = "CategoricalCrossEntropyLoss"
    metric_type = "categorical_accuracy"
    num_epochs = None
    learning_rate = None
    input_dim = None
    hidden_node = {}
    output_node = {}
    callbacks = []

    @abstractmethod
    def load_datasets():
        pass


class DLRMConfig(ABC):
    """
    Default config for DLRM models.
    """

    loss_fn = "CategoricalCrossEntropyLoss"
    metric_type = "categorical_accuracy"
    num_epochs = None
    learning_rate = None
    delimiter = None
    callbacks = []

    input_dim = None
    token_input = {}
    first_hidden_node = {}
    second_hidden_node = {}
    embedding_node = {}
    third_hidden_node = {}
    output_node = {}

    @abstractmethod
    def load_datasets():
        pass
