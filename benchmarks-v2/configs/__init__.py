# These base classes are declared here so that it is
# easier to enumerate all their subclasses
# (e.g. BoltBenchmarkConfig.__subclasses__()) prior
# to launching benchmark runs.


class UDTBenchmarkConfig:
    """
    Default config for UDT models.
    """

    learning_rate = 0.01
    num_epochs = 1
    target = "label"
    n_target_classes = 2
    delimiter = ","
    metric_type = "categorical_accuracy"
    model_config = None
    model_config_path = None
    callbacks = []


class BoltBenchmarkConfig:
    """
    Default config for two-layered Bolt models. This sets some
    default parameters for every layer with the assumption that
    every layer is dense unless otherwise specified.
    """

    loss_fn = "CategoricalCrossEntropyLoss"
    metric_type = "categorical_accuracy"
    num_epochs = 5
    hidden_sparsity = 1.0
    output_spasity = 1.0
    learning_rate = 1e-04
    hidden_activation = "ReLU"
    hidden_sampling_config = None
    output_activation = "Softmax"
    output_sampling_config = None
    callbacks = []


class DLRMConfig:
    """
    Default config for DLRM models.
    """

    loss_fn = "CategoricalCrossEntropyLoss"
    metric_type = "categorical_accuracy"
    num_epochs = 1
    learning_rate = 1e-04
    delimiter = " "
    max_num_numerical_features = 13
    max_num_categorical_features = 26
    compute_roc_auc = False
    callbacks = []

    num_embedding_lookups = 8
    embedding_lookup_size = 16
    log_embedding_block_size = 20
    num_tokens_per_input = 26
