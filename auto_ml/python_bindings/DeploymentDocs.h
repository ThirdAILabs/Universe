#pragma once

namespace thirdai::automl::deployment::python::docs {

const char* const UINT_HYPERPARAMETER = R"pbdoc(
Represents a integer parameter in the DeploymentConfig.

Notes:
    This class cannot be constructed directly. It should be constructed through 
    the functions ConstantParameter, OptionMappedParameter, or UserSpecifiedParameter.

Examples:
    >>> param = deployment.ConstantParameter(10)
    >>> param = deployment.OptionMappedParameter(option_name="size", values={"small": 10, "large": 20})
    >>> param = deployment.UserSpecifiedParameter(name="n_classes", type=int)

)pbdoc";

const char* const FLOAT_HYPERPARAMETER = R"pbdoc(
Represents a float parameter in the DeploymentConfig.

Notes:
    This class cannot be constructed directly. It should be constructed through 
    the functions ConstantParameter, OptionMappedParameter, or UserSpecifiedParameter.

Examples:
    >>> param = deployment.ConstantParameter(1.4)
    >>> param = deployment.OptionMappedParameter(option_name="sparsity", values={"sparse": 0.1, "dense": 1.0})
    >>> param = deployment.UserSpecifiedParameter(name="sparsity", type=float)

)pbdoc";

const char* const STR_HYPERPARAMETER = R"pbdoc(
Represents a string parameter in the DeploymentConfig.

Notes:
    This class cannot be constructed directly. It should be constructed through 
    the functions ConstantParameter, OptionMappedParameter, or UserSepcifiedParameter.

Examples:
    >>> param = deployment.ConstantParameter("relu")
    >>> param = deployment.OptionMappedParameter(option_name="task", values={"single_class": "softmax", "multi_class": "sigmoid"})
    >>> param = deployment.UserSpecifiedParameter(name="activation", type=str)

)pbdoc";

const char* const BOOL_HYPERPARAMETER = R"pbdoc(
Represents a boolean parameter in the DeploymentConfig.

Notes:
    This class cannot be constructed directly. It should be constructed through 
    the functions ConstantParameter, OptionMappedParameter, or UserSepcifiedParameter.

Examples:
    >>> param = deployment.ConstantParameter(True)
    >>> param = deployment.OptionMappedParameter(option_name="use_pairgrams", values={"small": False, "large": True})
    >>> param = deployment.UserSpecifiedParameter(name="use_sparse_inference", type=bool)

)pbdoc";

const char* const SAMPLING_CONFIG_HYPERPARAMETER = R"pbdoc(
Represents a sampling config parameter in the DeploymentConfig.

Notes:
    This class cannot be constructed directly. It should be constructed through 
    the functions ConstantParameter or OptionMappedParameter. Note that a sampling 
    config hyper parameter cannot be constructed as a UserSpecifiedParameter as 
    it is part of the complexity that should be abstracted away from the user API.

Examples:
    >>> param = deployment.ConstantParameter(bolt.DWTASamplingConfig(...))
    >>> param = deployment.OptionMappedParameter(
            option_name="size", values={"small": bolt.DWTASamplingConfig(...), "large": bolt.DWTASamplingConfig(...)}
        )

)pbdoc";

const char* const CONSTANT_PARAMETER = R"pbdoc(
Constructs a ConstantHyperParameter which is a HyperParameter with a fixed value 
that cannot be impacted by user inputed parameters.

Args:
    value (bool, int, float, str, or bolt.SamplingConfig): The constant value that 
        the constant parameter will take. 

Returns:
    (BoolHyperParameter, UintHyperParameter, FloatHyperParameter, StrHyperParameter, or SamplingConfigHyperParameter):
        This function is overloaded and hence will construct the appropriate hyper 
        parameter for the type of the input value. 
        
Notes:
    This function will actually return an instance of the ConstantParameter<T> class 
    which is a subclass of HyperParameter<T> which is the underlying class for 
    UintHyperParameter, FloatHyperParameter, etc.

Examples:
    >>> param = deployment.ConstantParameter(10)
    >>> param = deployment.ConstantParameter("relu")

)pbdoc";

const char* const OPTION_MAPPED_PARAMETER = R"pbdoc(
Constructs an OptionMappedHyperParameter which is a HyperParameter which maps 
user specified options (given as strings) to different possible values. Which 
option is specified by the user is determined by parameters mapping passed into 
the ModelPipeline constructor. The option_name argument is used to search this map, 
and the corresponding value is used to query the the values mapping.

Args:
    option_name (str): The name of the option mapped parameter. This is used to 
        search the parameters mapping to find the value of the option the user 
        specified. 
    values (Dict[str, bool], Dict[str, int], Dict[str, float], Dict[str, str], or Dict[str, bolt.SamplingConfig)]: 
        A mapping between the different user specified options and their corresponding 
        values.

Returns:
    (BoolHyperParameter, UintHyperParameter, FloatHyperParameter, StrHyperParameter, or SamplingConfigHyperParameter):
        This function is overloaded and hence will construct the appropriate hyper 
        parameter based on the type of the values of the dictionary. 
        
Notes:
    This function will actually return an instance of the OptionMappedParameter<T> 
    class which is a subclass of HyperParameter<T> which is the underlying class 
    for UintHyperParameter, FloatHyperParameter, etc.

Examples:
    >>> param = deployment.OptionMappedParameter(option_name="size", values={"small": 10, "large": 20})
    >>> model = deployment.ModelPipeline(config, parameters={"size": "small"})
)pbdoc";

const char* const USER_SPECIFIED_PARAMETER = R"pbdoc(
Constructs a UserSpecifiedHyperParameter which is a HyperParameter whose value is 
determined by the key in the parameters map (passed to the constructor
for the ModelPipeline) for the given parameter name. 

Args:
    name (str): The name of the parameter, this is used to search the parameter map.
    type (type): This should be one of bool, int, float, or str. This is the type 
    of the parameter, this is used to determine which staticly typed C++ class to 
    instantiate. 

Returns:
    (BoolHyperParameter, UintHyperParameter, FloatHyperParameter, StrHyperParameter, or SamplingConfigHyperParameter):
        This function is construct the appropriate hyper parameter based on the 
        type argument.
        
Notes:
    This function will actually return an instance of the OptionMappedParameter<T> 
    class which is a subclass of HyperParameter<T> which is the underlying class 
    for UintHyperParameter, FloatHyperParameter, etc.

Examples:
    >>> param = deployment.UserSpecifiedParameter(name="n_classes", type=int)
    >>> model = deployment.ModelPipeline(config, parameters={"n_classes": 32})
 
)pbdoc";

const char* const AUTOTUNED_SPARSITY_PARAMETER_INIT = R"pbdoc(
Constructs an AutoTunedSparsityParameter, which is a float HyperParameter whose 
value is determined by a user input dimension. This is intended to be used to autotune 
sparsity for a layer whose size is a user input. 

Args:
    dimension_parameter_name (str): The name of the user specified dimension parameter. 
        This name should to an UserSpecifiedParameter of type int. 

Returns:
    AutotunedSparsityParameter:
    A hyper parameter which can determine a value of the sparsity based off the 
    supplied dimension.

Notes:
    This HyperParameter is intended to be used for sparsity in the output layer.
    The intended use case is that the output dimension may be user specified, and
    we may want to use sparsity in this layer if the number of neurons is large
    enough, but we don't want the user to be responsible for inputing a
    reasonable sparsity value. Hence this class allows you to specify that the
    sparsity in a given layer is auto-tuned based off of a user specified
    dimension. Note that using an OptionMappedParameter is not sufficient because
    it would require enumerating the possible dimensions. Also note that it is
    best practice to use OptionMappedParameters for hidden layer dimensions to
    ensure reasonable architectures, and so this should really only be used in
    the output layer.

Examples:
    >>> param = deployment.AutotunedSparsityParameter(dimension_parameter_name="n_classes")
    >>> model = deployment.ModelPipeline(config, parameters={"n_classes": 1000})

)pbdoc";

const char* const NODE_CONFIG = R"pbdoc(
This class connot be represented directly. It is the base class for the configs for 
different nodes/layers in the model of the ModelConfig for the ModelPipeline.

)pbdoc";

const char* const FULLY_CONNECTED_CONFIG_INIT_WITH_SPARSITY = R"pbdoc(
Constructs a FullyConnectedNodeConfig which represents a FullyConnected node in 
the final bolt dag model.

Args:
    name (str): The name of the node. This is used by subsequent nodes to reference
        this node as a predecessor.
    dim (UintHyperParameter): The HyperParameter representing the dimension.
    sparsity (FloatHyperParameter): The HyperParameter representing the sparsity.
    activation (StrHyperParameter): The HyperParameter representing the activation 
        function.
    predecessor (str): The name of the node which should be used as this node's 
        predecessor. This must be the name of another node in the config.
    sampling_config (Optional[bolt.SamplingConfig]): This is an optional parameter 
        which represents the sampling config to be used. If not specified this is 
        autotuned if the sparsity ends up as less than 1.0. 

Returns: 
    FullyConnectedNodeConfig: 

Examples:
    >>> layer_config = deployment.FullyConnectedNodeConfig(
            name="hidden",
            dim=deployment.OptionMappedParameter(
                option_name="size", values={"small": 100, "large": 200}
            ),
            sparsity=deployment.UserSpecifiedParameter("sparsity"),
            activation=deployment.ConstantParameter("relu"),
            predecessor="input",
        )

)pbdoc";

const char* const FULLY_CONNECTED_CONFIG_INIT_DENSE = R"pbdoc(
Constructs a FullyConnectedNodeConfig which represents a dense FullyConnected node
in the final bolt dag model. This differs from the other constructor as it does 
not take in sparsity or sampling config parameters.

Args:
    name (str): The name of the node. This is used by subsequent nodes to reference
        this node as a predecessor.
    dim (UintHyperParameter): The HyperParameter representing the dimension.
    activation (StrHyperParameter): The HyperParameter representing the activation 
        function.
    predecessor (str): The name of the node which should be used as this node's 
        predecessor. This must be the name of another node in the config. 

Returns: 
    FullyConnectedNodeConfig: 

Examples:
    >>> layer_config = deployment.FullyConnectedNodeConfig(
            name="hidden",
            dim=deployment.OptionMappedParameter(
                option_name="size", values={"small": 100, "large": 200}
            ),
            activation=deployment.ConstantParameter("relu"),
            predecessor="input",
        )

)pbdoc";

const char* const MODEL_CONFIG_INIT = R"pbdoc(
Constructs a ModelConfig which represents the complete bolt dag model for a ModelPipeline.

Args:
    input_names (List[str]): The names of the inputs to the model. The actual input nodes are 
        constructed by the DatasetFactory in the ModelPipeline since it will have 
        knowledge of the output of the dataset loaders. This list of names is simply
        to refer to those inputs. It must be the same length as the number of inputs 
        returned from the dataset loaders.
    nodes (List[NodeConfig]): The list of nodes to be used in the dag model. Note 
        that the last node in the list is assumed to be the output node. 
    loss (bolt.LossFunction): The loss function to use to compile the model. 

Returns:
    ModelConfig:

Examples:
    >>> model_config = deployment.ModelConfig(
            input_names=["input"],
            nodes=[
                deployment.FullyConnectedNodeConfig(
                    name="hidden",
                    dim=deployment.OptionMappedParameter(
                        option_name="size", values={"small": 100, "large": 200}
                    ),
                    activation=deployment.ConstantParameter("relu"),
                    predecessor="input",
                ),
                deployment.FullyConnectedNodeConfig(
                    name="output",
                    dim=deployment.UserSpecifiedParameter("output_dim", type=int),
                    sparsity=deployment.ConstantParameter(1.0),
                    activation=deployment.ConstantParameter("softmax"),
                    predecessor="hidden",
                ),
            ],
            loss=bolt.CategoricalCrossEntropyLoss(),
        )
           
)pbdoc";

const char* const BLOCK_CONFIG = R"pbdoc(
This class connot be represented directly. It is the base class for the configs for 
different blocks in the dataset pipeline.

)pbdoc";

const char* const NUMERICAL_CATEGORICAL_BLOCK_CONFIG_INIT = R"pbdoc(
Constructs a config representing a NumericalCategoricalBlock in the dataset pipeline.

Args:
    n_classes (UintHyperParameter): The number of classes (categories) that could 
        occur in the dataset at this column.
    delimiter (StrHyperParameter): A character which will delineate the different 
        labels in the column if it is multi-class.

Returns:
    NumericalCategoricalBlockConfig:

Notes:
    The column index will be passed passed to this config to construct the final
    block for the dataset pipeline. The column index will be infered by the dataset
    config which contains this block config.

Example:
    >>> label_block = deployment.NumericalCategoricalBlockConfig(
            n_classes=deployment.UserSpecifiedParameter("output_dim", type=int),
            delimiter=deployment.ConstantParameter(","),
        )

)pbdoc";

const char* const DENSE_ARRAY_BLOCK_CONFIG_INIT = R"pbdoc(
Constructs a config representing a DenseArrayBlock in the dataset pipeline.

Args:
    dim (UintHyperParameter): The number of columns to be used to construct the array. 

Returns:
    DenseArrayBlockConfig:

Notes:
    The column index will be passed passed to this config to construct the final
    block for the dataset pipeline. The column index will be infered by the dataset
    config which contains this block config.

Example:
    >>> array_block = deployment.DenseArrayBlockConfig(
            dim=deployment.ConstantParameter(8)
        )

)pbdoc";

const char* const TEXT_BLOCK_CONFIG_INIT_WITH_RANGE = R"pbdoc(
Constructs a config representing a PairGramTextBlock or UniGramTextBlock in the 
dataset pipeline.

Args:
    use_pairgrams (bool): Whether or not to use pairgrams on the text. This is not 
        a hyperparameter because we want to abstract this decision away from the 
        user.
    range (UintHyperParameter): The range of the unigrams or pairgrams.

Returns:
    TextBlockConfig:

Notes:
    The column index will be passed passed to this config to construct the final
    block for the dataset pipeline. The column index will be infered by the dataset
    config which contains this block config.

Example:
    >>> text_block = deployment.TextBlockConfig(use_pairgrams=True)

)pbdoc";

const char* const TEXT_BLOCK_CONFIG_INIT = R"pbdoc(
Constructs a config representing a PairGramTextBlock or UniGramTextBlock in the 
dataset pipeline.

Args:
    use_pairgrams (bool): Whether or not to use pairgrams on the text. This is not 
        a hyperparameter because we want to abstract this decision away from the 
        user.

Returns:
    TextBlockConfig:

Notes:
    The column index will be passed passed to this config to construct the final
    block for the dataset pipeline. The column index will be infered by the dataset
    config which contains this block config.

Example:
    >>> data_block = deployment.TextBlockConfig(use_pairgrams=True)

)pbdoc";

const char* const DATASET_LOADER_FACTORY_CONFIG = R"pbdoc(
This class cannot be constructed directly, and is the base class for all of the 
dataset loader factory configs which are responsible for constructing the dataset 
factories for the ModelPipeline. 

)pbdoc";

const char* const SINGLE_BLOCK_DATASET_FACTORY_CONFIG_INIT = R"pbdoc(
This is a config for a simple dataset factory which has a single data block and 
a single label block. It expects that the dataset will be a csv with two columns, 
<label>,<data>. In the future we will want to add more complicated dataset factory
configs, but this should be sufficient for our initial use cases.

Args:
    data_block (BlockConfig): The config for the data block.
    label_block (BlockConfig): The config for the label block.
    shuffle (bool): Whether or not to shuffle the data on loading.
    delimiter (str): A character representing the delimiter of the columns.

Returns:
    SingleBlockDatasetFactory:

Examples:
    >>> dataset_config = deployment.SingleBlockDatasetFactory(
            data_block=deployment.TextBlockConfig(use_pairgrams=True),
            label_block=deployment.NumericalCategoricalBlockConfig(
                n_classes=deployment.UserSpecifiedParameter("output_dim", type=int),
                delimiter=deployment.ConstantParameter(","),
            ),
            shuffle=deployment.ConstantParameter(False),
            delimiter=deployment.UserSpecifiedParameter("delimiter", type=str),
        )

)pbdoc";

const char* const TRAIN_EVAL_PARAMETERS_CONFIG_INIT = R"pbdoc(
This class represents additional parameters that are required for training, evaluation,
or inference, but that we want to abstract away from the user.

Args:
    rebuild_hash_tables_interval (Option[int]): Interval (in number of samples) 
        for rebuilding the hash tabels in sparse layers. This is autotuned if not 
        specified.
    reconstruct_hash_functions_interval (Option[int]): Interval (in number of samples)
        for reconstructing the hash functions in sparse layers. This is autotuned
        if not specified.
    default_batch_size (int): The default batch size the ModelPipeline should use
        for training if the user does not specify it. 
    use_sparse_inference (bool): If sparse inference should be used for the model.
    evaluation_metrics (List[str]): Any metrics that should be computed during 
        evaluation.
    prediction_threshold (Option[float]): Optional parameter, if specified the model
        will ensure that the largest activation is always at least this threshold.
        This is used for multi-class classification tasks that use a theshold to 
        determine predictions.

Returns:
    TrainEvalParameters:

Examples:
    >>> train_eval_params = deployment.TrainEvalParameters(
            rebuild_hash_tables_interval=None,
            reconstruct_hash_functions_interval=None,
            default_batch_size=256,
            use_sparse_inference=True,
            evaluation_metrics=["categorical_accuracy"],
        )

)pbdoc";

const char* const DEPLOYMENT_CONFIG_INIT = R"pbdoc(
Constructs a DeploymentConfig which specifies a ModelPipeline.

Args:
    dataset_config (DatasetConfig): The config for the dataset loaders.
    model_config (ModelConfig): The config for the model.
    train_eval_parameters (TrainEvalParameters): The additional train and eval 
        parameters.

Returns:
    DeploymentConfig:

Examples:
    >>> config = deployment.DeploymentConfig(
            dataset_config=dataset_config,
            model_config=model_config,
            train_eval_parameters=train_eval_params,
        )

)pbdoc";

const char* const DEPLOYMENT_CONFIG_SAVE = R"pbdoc(
Saves a serialized version of the deployment config. This can be used to provide 
a ModelPipeline architecture to a customer, as the ModelPipeline has a constructor 
that allows it to be constructed directly from the serialized config.

Args:
    filename (str): The file to save the serialized config in.

Returns:
    None

)pbdoc";

const char* const DEPLOYMENT_CONFIG_LOAD = R"pbdoc(
Loads a saved deployment config. 

Args:
    filename (str): The file which contains the serialized config.

Returns:
    DeploymentConfig:

)pbdoc";

const char* const MODEL_PIPELINE_INIT_FROM_CONFIG = R"pbdoc(
Constructs a ModelPipeline from a deployment config and a set of input parameters.

Args:
    deployment_config (DeploymentConfig): A config for the ModelPipeline.
    parameters (Dict[str, Union[bool, int, float, str]]): A mapping from parameter 
        names to values. This is used to pass in values to UserSpecifiedParameters, 
        or provide the name of the option to use for OptionMappedParameters.

Returns
    ModelPipeline:

Examples:
    >>> model = deployment.ModelPipeline(
            deployment_config=deployment.DeploymentConfig(...),
            parameters={"size": "large", "output_dim": num_classes, "delimiter": ","},
        )

)pbdoc";

const char* const MODEL_PIPELINE_INIT_FROM_SAVED_CONFIG = R"pbdoc(
Constructs a ModelPipeline from a serialized deployment config and a set of input 
parameters.

Args:
    config_path (str): A path to a serialized deployment config for the ModelPipeline.
    parameters (Dict[str, Union[bool, int, float, str]]): A mapping from parameter 
        names to values. This is used to pass in values to UserSpecifiedParameters, 
        or provide the name of the option to use for OptionMappedParameters.

Returns
    ModelPipeline:

Examples:
    >>> model = deployment.ModelPipeline(
            config_path="path_to_a_config",
            parameters={"size": "large", "output_dim": num_classes, "delimiter": ","},
        )

)pbdoc";

const char* const MODEL_PIPELINE_TRAIN_FILE = R"pbdoc(
Trains a ModelPipeline on a given dataset using a file on disk.

Args:
    filename (str): Path to the dataset file.
    epochs (int): Number of epochs to train for.
    learning_rate (float): The learning rate to use for training.
    batch_size (Option[int]): This is an optional parameter indicating which batch
        size to use for training. If not specified the default batch size from the 
        TrainEvalParameters is used.
    max_in_memory_batches (Option[int]): The maximum number of batches to load in
        memory at a given time. If this is specified then the dataset will be processed
        in a streaming fashion.

Returns:
    None

Examples:
    >>> model.train(
            filename=TRAIN_FILE, epochs=5, learning_rate=0.01, max_in_memory_batches=12
        )

)pbdoc";

const char* const MODEL_PIPELINE_TRAIN_DATA_LOADER = R"pbdoc(
Trains a ModelPipeline on a given dataset using any DataLoader.

Args:
    data_source (dataset.DataLoader): A data loader for the given dataset.
    epochs (int): Number of epochs to train for.
    learning_rate (float): The learning rate to use for training.
    max_in_memory_batches (Option[int]): The maximum number of batches to load in
        memory at a given time. If this is specified then the dataset will be processed
        in a streaming fashion.

Returns:
    None

Examples:
    >>> model.train(
            data_source=dataset.S3DataLoader(...), epochs=5, learning_rate=0.01, max_in_memory_batches=12
        )

)pbdoc";

const char* const MODEL_PIPELINE_EVALUATE_FILE = R"pbdoc(
Evaluates the ModelPipeline on the given dataset and returns a numpy array of the 
activations.

Args:
    filename (str): Path to the dataset file.

Returns:
    (np.ndarray or Tuple[np.ndarray, np.ndarray]): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (dataset_length, num_nonzeros_in_output).

Examples:
    >>> activations = model.evaluate(filename=TEST_FILE)

)pbdoc";

const char* const MODEL_PIPELINE_EVALUATE_DATA_LOADER = R"pbdoc(
Evaluates the ModelPipeline on the given dataset and returns a numpy array of the 
activations.

Args:
    data_source (dataset.DataLoader): A data loader for the given dataset.

Returns:
    (np.ndarray or Tuple[np.ndarray, np.ndarray]): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (dataset_length, num_nonzeros_in_output).

Examples:
    >>> (active_neurons, activations) = model.evaluate(data_source=dataset.S3DataLoader(...))

)pbdoc";

const char* const MODEL_PIPELINE_PREDICT = R"pbdoc(
Performs inference on a single sample.

Args:
    input_sample (str): A str representing the input. This will be processed in the 
        same way as the dataset, and thus should be the same format as a line in 
        the dataset, except with the label columns removed.

Returns: 
    (np.ndarray or Tuple[np.ndarray, np.ndarray]): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (num_nonzeros_in_output, ).

Examples:
    >>> activations = model.predict("The blue cat jumped")

)pbdoc";

const char* const MODEL_PIPELINE_PREDICT_TOKENS = R"pbdoc(
Performs inference on a single sample represented as bert tokens

Args:
    tokens (List[int]): A list of integers representing bert tokens.

Returns: 
    (np.ndarray or Tuple[np.ndarray, np.ndarray]): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (num_nonzeros_in_output, ).

Examples:
    >>> activations = model.predict_tokens(tokens=[9, 42, 19, 71, 33])

)pbdoc";

const char* const MODEL_PIPELINE_PREDICT_BATCH = R"pbdoc(
Performs inference on a batch of samples samples in parallel.

Args:
    input_samples (List[str]): A list of strings representing each sample. Each 
        input will be processed in the same way as the dataset, and thus should 
        be the same format as a line in the dataset, except with the label columns 
        removed.

Returns: 
    (np.ndarray or Tuple[np.ndarray, np.ndarray]): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (batch_size, num_nonzeros_in_output).

Examples:
    >>> activations = model.predict_batch([
            "The cat ran",
            "The dog sat", 
            "The cow ate grass"
        ])

)pbdoc";

const char* const MODEL_PIPELINE_SAVE = R"pbdoc(
Saves a serialized version of the ModelPipeline.

Args:
    filename (str): The file to save the serialized ModelPipeline in.

Returns:
    None

)pbdoc";

const char* const MODEL_PIPELINE_LOAD = R"pbdoc(
Loads a saved deployment config. 

Args:
    filename (str): The file which contains the serialized config.

Returns:
    DeploymentConfig:

)pbdoc";

}  // namespace thirdai::automl::deployment::python::docs