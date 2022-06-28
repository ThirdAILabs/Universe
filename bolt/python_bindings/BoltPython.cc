#include "BoltPython.h"
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/text_classifier/TextClassifier.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <limits>
#include <optional>
#include <sstream>
#include <string>

namespace thirdai::bolt::python {

void createBoltSubmodule(py::module_& module) {
  auto bolt_submodule = module.def_submodule("bolt");

#if THIRDAI_EXPOSE_ALL
#pragma message("THIRDAI_EXPOSE_ALL is defined")  // NOLINT
  py::class_<thirdai::bolt::SamplingConfig>(
      bolt_submodule, "SamplingConfig",
      "SamplingConfig represents a layer's sampling hyperparameters.")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, std::string>(),
           py::arg("hashes_per_table"), py::arg("num_tables"),
           py::arg("range_pow"), py::arg("reservoir_size"),
           py::arg("hash_function") = "DWTA",
           "Builds a SamplingConfig object. \n\n"
           "Arguments:\n"
           " * hashes_per_table: Int - number of hashes to be concatenated in "
           "each table."
           " * num_tables: Int - number of hash tables."
           " * range_pow: Int - hash range as a power of 2. E.g. if hash range "
           "is 8, range_pow = 3. "
           " Note that the correct range_pow differs for each hash function. "
           "For DWTA, range_pow = 3 * hashes_per_table."
           " For SRP or FastSRP, range_pow = hashes_per_table."
           " * reservoir_size: Int - maximum number of elements stored in each "
           "hash bucket."
           " * hash_function: Pass hash function as string (Optional) - The "
           "hash function "
           "used for sparse training and inference. One of DWTA, SRP, or "
           "FastSRP. Defaults to DWTA.")
      .def(py::init<>(), "Builds a default SamplingConfig object.")
      .def_readonly("hashes_per_table", &SamplingConfig::hashes_per_table)
      .def_readonly("num_tables", &SamplingConfig::num_tables)
      .def_readonly("range_pow", &SamplingConfig::range_pow)
      .def_readonly("reservoir_size", &SamplingConfig::reservoir_size)
      .def_readonly("hash_function", &SamplingConfig::_hash_function);

#endif

  py::enum_<ActivationFunction>(
      bolt_submodule, "ActivationFunctions",
      "An enum of all available activation functions. To use it, pass it to "
      "the 'activation_function' parameter of a LayerConfig object.")
      .value("ReLU", ActivationFunction::ReLU,
             "Rectified Linear Units (ReLU) activation function; "
             "introduces non-linearity to the neural network.")
      .value("Linear", ActivationFunction::Linear,
             "Returns the outputs of a layer as-is.")
      .value("Tanh", ActivationFunction::Tanh,
             "Hyperbolic tangent activation function; "
             "maps the outputs of a layer to the range [-1, 1].")
      .value("Softmax", ActivationFunction::Softmax,
             "Softmax activation function; converts logits to classification "
             "probabilities. Currently, this activation function can only be "
             "applied to the final layer in the neural network.")
      .value("Sigmoid", ActivationFunction::Sigmoid,
             "Sigmoid activation function; converts logits to indepedent"
             "probabilities. Currently, this activation function can only be "
             "applied to the final layer in the neural network.");

  bolt_submodule.def("getActivationFunction", &getActivationFunction,
                     py::arg("name"),
                     "Converts an activation function name to "
                     "the corresponding enum.");

  // TODO(Geordie, Nicholas): put loss functions in its own submodule
  py::class_<LossFunction, std::shared_ptr<LossFunction>>(  // NOLINT
      bolt_submodule, "LossFunction", "Base class for all loss functions");

  py::class_<CategoricalCrossEntropyLoss,
             std::shared_ptr<CategoricalCrossEntropyLoss>, LossFunction>(
      bolt_submodule, "CategoricalCrossEntropyLoss",
      "A loss function for multi-class (one label per sample) classification "
      "tasks.")
      .def(py::init<>(), "Constructs a CategoricalCrossEntropyLoss object.");

  py::class_<BinaryCrossEntropyLoss, std::shared_ptr<BinaryCrossEntropyLoss>,
             LossFunction>(
      bolt_submodule, "BinaryCrossEntropyLoss",
      "A loss function for multi-label (multiple class labels per each sample) "
      "classification tasks.")
      .def(py::init<>(), "Constructs a BinaryCrossEntropyLoss object.");

  py::class_<MeanSquaredError, std::shared_ptr<MeanSquaredError>, LossFunction>(
      bolt_submodule, "MeanSquaredError",
      "A loss function that minimizes mean squared error (MSE) for regression "
      "tasks. "
      "MSE = sum( (actual - prediction)^2 )")
      .def(py::init<>(), "Constructs a MeanSquaredError object.");

  py::class_<WeightedMeanAbsolutePercentageErrorLoss,
             std::shared_ptr<WeightedMeanAbsolutePercentageErrorLoss>,
             LossFunction>(
      bolt_submodule, "WeightedMeanAbsolutePercentageError",
      "A loss function to minimize weighted mean absolute percentage error "
      "(WMAPE) "
      "for regression tasks. WMAPE = 100% * sum(|actual - prediction|) / "
      "sum(|actual|)")
      .def(py::init<>(),
           "Constructs a WeightedMeanAbsolutePercentageError object.");

  py::class_<thirdai::bolt::SequentialLayerConfig,  // NOLINT
             std::shared_ptr<thirdai::bolt::SequentialLayerConfig>>(
      bolt_submodule, "Sequential");

  py::class_<thirdai::bolt::FullyConnectedLayerConfig,
             std::shared_ptr<thirdai::bolt::FullyConnectedLayerConfig>,
             thirdai::bolt::SequentialLayerConfig>(
      bolt_submodule, "FullyConnected", "Defines a fully-connected layer.\n")
#if THIRDAI_EXPOSE_ALL
      .def(py::init<uint64_t, float, ActivationFunction,
                    thirdai::bolt::SamplingConfig>(),
           py::arg("dim"), py::arg("sparsity"), py::arg("activation_function"),
           py::arg("sampling_config"),
           "Constructs the FullyConnectedLayerConfig object.\n"
           "Arguments:\n"
           " * dim: Int - The dimension of the layer.\n"
           " * sparsity: Float - The fraction of neurons to use during "
           "sparse training "
           "and sparse inference. For example, sparsity=0.05 means the "
           "layer uses 5% of "
           "its neurons when processing an individual sample.\n"
           " * activation_function: ActivationFunctions enum - We support five "
           "activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n"
           " * sampling_config: SamplingConfig - Sampling configuration.")
#endif
      .def(py::init<uint64_t, ActivationFunction>(), py::arg("dim"),
           py::arg("activation_function"),
           "Constructs a FullyConnectedLayerConfig object.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * activation_function: ActivationFunctions enum - We support five "
           "activation functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n"
           "Also accepts `getActivationFunction(function_name), e.g. "
           "`getActivationFunction('ReLU')`")
      .def(py::init<uint64_t, float, ActivationFunction>(), py::arg("dim"),
           py::arg("sparsity"), py::arg("activation_function"),
           "Constructs a FullyConnectedLayerConfig object.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * activation_function: ActivationFunctions enum - We support five "
           "activation functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n"
           " * sparsity: Float - The fraction of neurons to use during "
           "sparse training "
           "and sparse inference. For example, sparsity=0.05 means the "
           "layer uses 5% of "
           "its neurons when processing an individual sample.\n"
           "Also accepts `getActivationFunction(function_name), e.g. "
           "`getActivationFunction('ReLU')`")
      .def(py::init<uint64_t, float, std::string>(), py::arg("dim"),
           py::arg("sparsity"), py::arg("activation_function"),
           "Constructs a FullyConnectedLayerConfig object.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * activation_function: String specifying the activation function "
           "to use, no restrictions on case - We support five activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n"
           " * sparsity: Float - The fraction of neurons to use during "
           "sparse training "
           "and sparse inference. For example, sparsity=0.05 means the "
           "layer uses 5% of "
           "its neurons when processing an individual sample.\n")
      .def(py::init<uint64_t, std::string>(), py::arg("dim"),
           py::arg("activation_function"),
           "Constructs a FullyConnectedLayerConfig object.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * activation_function: String specifying the activation function "
           "to use, no restrictions on case - We support five activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n"
           "Eg. relu or Relu ,Softmax or softMax, Linear or lineaR.");

#if THIRDAI_EXPOSE_ALL
  py::class_<thirdai::bolt::ConvLayerConfig,
             std::shared_ptr<thirdai::bolt::ConvLayerConfig>,
             thirdai::bolt::SequentialLayerConfig>(
      bolt_submodule, "Conv",
      "Defines a 2D convolutional layer that convolves over "
      "non-overlapping patches.")
      .def(py::init<uint64_t, float, ActivationFunction,
                    thirdai::bolt::SamplingConfig,
                    std::pair<uint32_t, uint32_t>, uint32_t>(),
           py::arg("num_filters"), py::arg("sparsity"),
           py::arg("activation_function"), py::arg("sampling_config"),
           py::arg("kernel_size"), py::arg("num_patches"),
           "Constructs the ConvLayerConfig object.\n"
           "Arguments:\n"
           " * num_filters: Int - Number of convolutional filters.\n"
           " * sparsity: Float - The fraction of filters to use during "
           "sparse training and sparse inference. For example, "
           "sparsity=0.05 means the layer uses 5% of the filters "
           "when processing each patch.\n"
           " * activation_function: ActivationFunctions enum - We support five "
           "activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n"
           " * sampling_config: SamplingConfig - Sampling configuration.\n"
           " * kernel_size: Pair of ints - 2D dimensions of each patch.\n"
           " * num_patches: Int - Number of patches.")
      .def(py::init<uint64_t, float, ActivationFunction,
                    std::pair<uint32_t, uint32_t>, uint32_t>(),
           py::arg("num_filters"), py::arg("sparsity"),
           py::arg("activation_function"), py::arg("kernel_size"),
           py::arg("num_patches"),
           "Constructs a ConvLayerConfig object.\n"
           "Arguments:\n"
           " * num_filters: Int (positive) - Number of convolutional filters.\n"
           " * sparsity: Float (positive) - The fraction of filters to use "
           "during "
           "sparse training and sparse inference. For example, "
           "sparsity=0.05 means the layer uses 5% of the filters "
           "when processing each patch.\n"
           " * activation_function: ActivationFunctions enum, e.g. ReLU, "
           "Softmax, "
           "Sigmoid, "
           "Tanh, "
           "Linear. "
           "Also accepts `getActivationFunction(function_name), e.g. "
           "`getActivationFunction('ReLU')`\n"
           " * kernel_size: Pair of ints - 2D dimensions of each patch.\n"
           " * num_patches: Int (positive) - Number of patches.");
  py::class_<thirdai::bolt::EmbeddingLayerConfig>(
      bolt_submodule, "Embedding",
      "Defines a space-efficient embedding table lookup layer.")
      .def(py::init<uint32_t, uint32_t, uint32_t>(),
           py::arg("num_embedding_lookups"), py::arg("lookup_size"),
           py::arg("log_embedding_block_size"),
           "Constructs an embedding layer.\n"
           "Arguments:\n"
           " * num_embedding_lookups: Int (positive) - The number of embedding "
           "table "
           "lookups per categorical feature.\n"
           " * lookup_size: Int (positive) - Embedding dimension.\n"
           " * log_embedding_block_size: Int (positive) - log (base 2) of the "
           "size of the "
           "embedding table.");
#endif

  py::class_<PyNetwork>(bolt_submodule, "Network",
                        "Fully connected neural network.")
      .def(py::init<std::vector<
                        std::shared_ptr<thirdai::bolt::SequentialLayerConfig>>,
                    uint64_t>(),
           py::arg("layers"), py::arg("input_dim"),
           "Constructs a neural network.\n"
           "Arguments:\n"
           " * layers: List of SequentialLayerConfig - Configurations for the "
           "sequence of "
           "layers in the neural network.\n"
           " * input_dim: Int (positive) - Dimension of input vectors in the "
           "dataset.")
      .def("__str__",
           [](const PyNetwork& network) {
             std::stringstream summary;
             network.buildNetworkSummary(summary);
             return summary.str();
           })
      .def(
          "summary", &PyNetwork::printSummary, py::arg("detailed") = false,
          "Prints a summary of the network.\n"
          "Arguments:\n"
          " * detailed: boolean. Optional. When specified to \"True\", "
          "summary will additionally print layer config details for each layer "
          "in the network.")
      .def("train", &PyNetwork::train, py::arg("train_data"),
           py::arg("train_labels"), py::arg("loss_fn"),
           py::arg("learning_rate"), py::arg("epochs"),
           py::arg("batch_size") = 0, py::arg("rehash") = 0,
           py::arg("rebuild") = 0,
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           "Trains the network on the given training data.\n"
           "Arguments:\n"
           " * train_data: BoltDataset - Training data. This can be one of "
           "three things. First it can be a BoltDataset as loaded by "
           "thirdai.dataset.load_bolt_svm_dataset or "
           "thirdai.dataset.load_bolt_csv_dataset. It can be a dense numpy "
           "array of float32 where each row in the array is interpreted as a "
           "vector. Finally it can be a set of sparse vectors represented as "
           "three numpy arrays (indices, values, offsets) where indices and "
           "offsets are uint32 and values are float32. In this case indices is "
           "a 1D array of all the nonzero indices concatenated, values is a 1D "
           "array of all the nonzero values concatenated, and offsets are the "
           "start positions in the indices and values array of each vector "
           "plus one extra element at the end of the array representing the "
           "total number of nonzeros. This is so that indices[offsets[i], "
           "offsets[i + 1]] contains the indices of the ith vector and "
           "values[offsets[i], offsets[i+1] contains the values of the ith "
           "vector.For example if we have the vectors {0.0, 1.5, 0.0, 9.0} and "
           "{0.0, 0.0, 0.0, 4.0} then the indices array is {1, 3, 3}, the "
           "values array is {1.5, 9.0, 4.0} and the offsets array is {0, 2, "
           "3}.\n"
           " * train_labels: BoltDataset - Training labels. This can be one of "
           "three things. First it can be a BoltDataset as loaded by "
           "thirdai.dataset.load_bolt_svm_dataset or "
           "thirdai.dataset.load_bolt_csv_dataset. It can be a dense numpy "
           "array of float32 where each row in the array is interpreted as a "
           "label vector. Finally it can be a set of sparse vectors (each "
           "vector is a label vector) represented as three numpy arrays "
           "(indices, values, offsets) where indices and offsets are uint32 "
           "and values are float32. In this case indices is a 1D array of all "
           "the nonzero indices concatenated, values is a 1D array of all the "
           "nonzero values concatenated, and offsets are the start positions "
           "in the indices and values array of each vector plus one extra "
           "element at the end of the array representing the total number of "
           "nonzeros. This is so that indices[offsets[i], offsets[i + 1]] "
           "contains the indices of the ith vector and values[offsets[i], "
           "offsets[i+1] contains the values of the ith vector.For example if "
           "we have the vectors {0.0, 1.5, 0.0, 9.0} and {0.0, 0.0, 0.0, 4.0} "
           "then the indices array is {1, 3, 3}, the values array is {1.5, "
           "9.0, 4.0} and the offsets array is {0, 2, 3}.\n"
           " * loss_fn: LossFunction - The loss function to minimize.\n"
           " * learning_rate: Float (positive) - Learning rate.\n"
           " * epochs: Int (positive) - Number of training epochs over the "
           "training data.\n"
           " * rehash: Int (positive) - Optional. Number of training samples "
           "before "
           "rehashing neurons. "
           "If not provided, BOLT will autotune this parameter.\n\n"
           "\t\tBOLT's sparse training works by applying smart hash functions "
           "to "
           "all neurons in the "
           "network, and they have to be rehashed periodically. This parameter "
           "sets the frequency "
           "of rehashing.\n"
           " * rebuild: Int (positive) - Optional. Number of training samples "
           "before "
           "rebuilding hash tables and generating new smart hash functions. "
           "This is typically around 5 times the value of `rehash`."
           "If not provided, BOLT will autotune this parameter.\n"
           " * metrics: List of str - Optional. The metrics to keep track of "
           "during training. "
           "See the section on metrics.\n"
           " * verbose: Boolean - Optional. If set to False, only displays "
           "progress bar. "
           "If set to True, prints additional information such as metrics and "
           "epoch times. "
           "Set to True by default.\n\n"

           "Returns a mapping from metric names to an array their values for "
           "every epoch.")
      .def("predict", &PyNetwork::predict, py::arg("test_data"),
           py::arg("test_labels"), py::arg("batch_size") = 2048,
           py::arg("sparse_inference") = false,
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           py::arg("batch_limit") = std::numeric_limits<uint32_t>::max(),
           "Predicts the output given the input vectors and evaluates the "
           "predictions based on the given metrics.\n"
           "Arguments:\n"
           " * test_data: BoltDataset - Test data. This can be one of "
           "three things. First it can be a BoltDataset as loaded by "
           "thirdai.dataset.load_bolt_svm_dataset or "
           "thirdai.dataset.load_bolt_csv_dataset. It can be a dense numpy "
           "array of float32 where each row in the array is interpreted as a "
           "vector. Finally it can be a set of sparse vectors represented as "
           "three numpy arrays (indices, values, offsets) where indices and "
           "offsets are uint32 and values are float32. In this case indices is "
           "a 1D array of all the nonzero indices concatenated, values is a 1D "
           "array of all the nonzero values concatenated, and offsets are the "
           "start positions in the indices and values array of each vector "
           "plus one extra element at the end of the array representing the "
           "total number of nonzeros. This is so that indices[offsets[i], "
           "offsets[i + 1]] contains the indices of the ith vector and "
           "values[offsets[i], offsets[i+1] contains the values of the ith "
           "vector.For example if we have the vectors {0.0, 1.5, 0.0, 9.0} and "
           "{0.0, 0.0, 0.0, 4.0} then the indices array is {1, 3, 3}, the "
           "values array is {1.5, 9.0, 4.0} and the offsets array is {0, 2, "
           "3}.\n"
           " * test_labels: BoltDataset - Test labels. This can be one of "
           "four things. First it can be a BoltDataset as loaded by "
           "thirdai.dataset.load_bolt_svm_dataset or "
           "thirdai.dataset.load_bolt_csv_dataset. It can be a dense numpy "
           "array of float32 where each row in the array is interpreted as a "
           "label vector. Finally it can be a set of sparse vectors (each "
           "vector is a label vector) represented as three numpy arrays "
           "(indices, values, offsets) where indices and offsets are uint32 "
           "and values are float32. In this case indices is a 1D array of all "
           "the nonzero indices concatenated, values is a 1D array of all the "
           "nonzero values concatenated, and offsets are the start positions "
           "in the indices and values array of each vector plus one extra "
           "element at the end of the array representing the total number of "
           "nonzeros. This is so that indices[offsets[i], offsets[i + 1]] "
           "contains the indices of the ith vector and values[offsets[i], "
           "offsets[i+1] contains the values of the ith vector.For example if "
           "we have the vectors {0.0, 1.5, 0.0, 9.0} and {0.0, 0.0, 0.0, 4.0} "
           "then the indices array is {1, 3, 3}, the values array is {1.5, "
           "9.0, 4.0} and the offsets array is {0, 2, 3}. Finally, the test "
           "labels can be passed in as test_labels=None, in which case they "
           "will be ignored. If labels are not supplied then no metrics will "
           "be computed but activations will still be returned.\n"
           " * sparse_inference: (Bool) - When this is true the model will use "
           "sparsity in inference. This will lead to faster inference but can "
           "cause a slight loss in accuracy. This option defaults to false.\n"
           " * metrics: List of str - Optional. The metrics to keep track of "
           "during training. "
           "See the section on metrics.\n"
           " * verbose: Boolean - Optional. If set to False, only displays "
           "progress bar. "
           "If set to True, prints additional information such as metrics and "
           "inference times. "
           "Set to True by default.\n\n"

           "Returns a tuple consisting of (0) a mapping from metric names to "
           "their values "
           "and (1) output vectors (predictions) from the network in the form "
           "of a 2D Numpy matrix of floats.")
      .def("freeze_hash_tables", &PyNetwork::freezeHashTables,
           "Freezes hash tables in the network. If you plan to use sparse "
           "inference, you may get a significant performance improvement if "
           "you call this one or two epochs before you finish training. "
           "Otherwise you should not call this method.")
      .def("save_for_inference", &PyNetwork::saveForInference,
           py::arg("filename"),
           "Saves the network to a file. The file path must not require any "
           "folders to be created. Saves only essential parameters for "
           "inference, e.g. not the optimizer state")
      .def_static("load", &PyNetwork::load, py::arg("filename"),
                  "Loads and builds a saved network from file.")
      .def("checkpoint", &PyNetwork::checkpoint, py::arg("filename"),
           "Saves the network to a file. The file path must not require any "
           "folders to be created. Saves all the paramters needed for "
           "tranining. "
           "This will throw an error if the model has been trimmed for "
           "inference.")
      .def("trim_for_inference", &PyNetwork::trimForInference,
           "Removes all parameters that are not essential for inference, "
           "shrinking the model")
      .def("reinitialize_optimizer_for_training",
           &PyNetwork::reinitOptimizerForTraining,
           "If the model previously was trimmed for inference, this will "
           "reinitialize the optimizer state, allowing training again.")
      .def("get_weights", &PyNetwork::getWeights, py::arg("layer_index"),
           "Returns the weight matrix at the given layer index as a 2D Numpy "
           "matrix.")
      .def("setTrainable", &PyNetwork::setTrainable, py::arg("layer_index"),
           py::arg("trainable"),
           "Sets whether the layer with the given layer_index is trainable. "
           "Layers are always trainable by default. "
           "trainable is false. Trainable by default")
      .def("set_weights", &PyNetwork::setWeights, py::arg("layer_index"),
           py::arg("new_weights"),
           "Sets the weight matrix at the given layer index to the given 2D "
           "Numpy matrix. Throws an error if the dimension of the given weight "
           "matrix does not match the layer's current weight matrix.")
      .def("ready_for_training", &PyNetwork::isReadyForTraining,
           "Returns False if the optimizer state is not initialized, True "
           "otherwise. Call resume_training to initialize optimizer")
      .def("get_biases", &PyNetwork::getBiases, py::arg("layer_index"),
           "Returns the bias array at the given layer index as a 1D Numpy "
           "array.")
      .def("set_biases", &PyNetwork::setBiases, py::arg("layer_index"),
           py::arg("new_biases"),
           "Sets the bias array at the given layer index to the given 1D Numpy "
           "array.")
      .def("set_layer_sparsity", &PyNetwork::setLayerSparsity,
           py::arg("layer_index"), py::arg("sparsity"),
           "Sets the sparsity of the layer at the given index. The 0th layer "
           "is the first layer after the input layer. Note that this will "
           "autotune the sampling config to work for the new sparsity.")
      .def("get_layer_sparsity", &PyNetwork::getLayerSparsity,
           py::arg("layer_index"),
           "Gets the sparsity of the layer at the given index. The 0th layer "
           "is the first layer after the input layer.")
#if THIRDAI_EXPOSE_ALL
      .def("get_sampling_config", &PyNetwork::getSamplingConfig,
           py::arg("layer_index"),
           "Returns the sampling config of the layer at layer_index.")
#endif
      ;

  py::class_<PyDLRM>(bolt_submodule, "DLRM",
                     "DLRM network with space-efficient embedding tables.")
      .def(py::init<thirdai::bolt::EmbeddingLayerConfig,
                    std::vector<
                        std::shared_ptr<thirdai::bolt::SequentialLayerConfig>>,
                    std::vector<
                        std::shared_ptr<thirdai::bolt::SequentialLayerConfig>>,
                    uint32_t>(),
           py::arg("embedding_layer"), py::arg("bottom_mlp"),
           py::arg("top_mlp"), py::arg("input_dim"),
           "Constructs a DLRM.\n"
           "Arguments:\n"
           " * embedding_layer: EmbeddingLayerConfig - Configuration of the "
           "embedding layer.\n"
           " * bottom_mlp: List of SequentialLayerConfig - Configurations of "
           "the sequence "
           "of layers in DLRM's bottom MLP.\n"
           " * top_mlp: List of SequentialLayerConfig - Configurations of the "
           "sequence of "
           "layers in DLRM's top MLP.\n"
           " * input_dim: Int (positive) - Dimension of input vectors in the "
           "dataset.")
      .def("train", &PyDLRM::train, py::arg("train_data"),
           py::arg("train_labels"), py::arg("loss_fn"),
           py::arg("learning_rate"), py::arg("epochs"), py::arg("rehash") = 0,
           py::arg("rebuild") = 0,
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           "Trains the network on the given training data.\n"
           "Arguments:\n"
           " * train_data: ClickThroughDataset - Training data.\n"
           " * train_labels: BoltDataset - Training labels.\n"
           " * loss_fn: LossFunction - The loss function to minimize.\n"
           " * learning_rate: Float (positive) - Learning rate.\n"
           " * epochs: Int (positive) - Number of training epochs over the "
           "training data.\n"
           " * rehash: Int (positive) - Optional. Number of training samples "
           "before "
           "rehashing neurons. "
           "If not provided, BOLT will autotune this parameter.\n\n"
           "\t\tBOLT's sparse training works by applying smart hash functions "
           "to "
           "all neurons in the "
           "network, and they have to be rehashed periodically. This parameter "
           "sets the frequency "
           "of rehashing.\n"
           " * rebuild: Int (positive) - Optional. Number of training samples "
           "before "
           "rebuilding hash tables "
           "and generating new smart hash functions. If not provided, BOLT "
           "will autotune this parameter.\n"
           " * metrics: List of str - Optional. The metrics to keep track of "
           "during training. "
           "See the section on metrics.\n"
           " * verbose: Boolean - Optional. If set to False, only displays "
           "progress bar. "
           "If set to True, prints additional information such as metrics and "
           "epoch times. "
           "Set to True by default.\n\n"

           "Returns a mapping from metric names to an array their values for "
           "every epoch.")
      .def("predict", &PyDLRM::predict, py::arg("test_data"),
           py::arg("test_labels"), py::arg("sparse_inference") = false,
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           py::arg("batch_limit") = std::numeric_limits<uint32_t>::max(),
           "Predicts the output given the input vectors and evaluates the "
           "predictions based on the given metrics.\n"
           "Arguments:\n"
           " * test_data: ClickThroughDataset - Test data.\n"
           " * test_labels: BoltDataset - Testing labels.\n"
           " * metrics: List of str - Optional. The metrics to keep track of "
           "during training. See the section on metrics.\n"
           " * sparse_inference: (Bool) - When this is true the model will use "
           "sparsity in inference. This will lead to faster inference but can "
           "cause a slight loss in accuracy. This option defaults to false.\n"
           " * verbose: Boolean - Optional. If set to False, only displays "
           "progress bar. "
           "If set to True, prints additional information such as metrics and "
           "inference times. "
           "Set to True by default.\n\n"

           "Returns a tuple consisting of (0) a mapping from metric names to "
           "their values "
           "and (1) output vectors (predictions) from the network in the form "
           "of a 2D Numpy matrix of floats.");

  py::class_<TextClassifier>(bolt_submodule, "TextClassifier")
      .def(py::init<const std::string&, uint32_t>(), py::arg("model_size"),
           py::arg("n_classes"),
           "Constructs a TextClassifier with autotuning.\n"
           "Arguments:\n"
           " * model_size: string - Either 'small', 'medium', 'large', or a "
           "size in Gb for the model, for example '6Gb' or '6 Gb'.\n"
           " * n_classes: int - How many classes or categories are in the "
           "labels of the dataset.\n")
      .def("train", &TextClassifier::train, py::arg("train_file"),
           py::arg("epochs"), py::arg("learning_rate"),
           "Trains the classifier on the given dataset.\n"
           "Arguments:\n"
           " * train_file: string - The path to the training dataset to use.\n"
           " * epochs: Int - How many epochs to train for.\n"
           " * learning_rate: Float - The learning rate to use for training.\n")
      .def("predict", &TextClassifier::predict, py::arg("test_file"),
           py::arg("output_file") = std::nullopt,
           "Runs the classifier on the specified test dataset and optionally "
           "logs the prediction to a file.\n"
           "Arguments:\n"
           " * test_file: string - The path to the test dataset to use.\n"
           " * output_file: string - Optional argument, if this is specified "
           "then the classifier will output the name of the class/category of "
           "each prediction this file with one prediction result on each "
           "line.\n")
      .def("save", &TextClassifier::save, py::arg("filename"),
           "Saves the classifier to a file. The file path must not require any "
           "folders to be created\n"
           "Arguments:\n"
           " * filename: string - The path to the save location of the "
           "classifier.\n")
      .def_static(
          "load", &TextClassifier::load, py::arg("filename"),
          "Loads and builds a saved classifier from file.\n"
          "Arguments:\n"
          " * filename: string - The location of the saved classifier.\n");

  auto graph_submodule = bolt_submodule.def_submodule("graph");

  py::class_<Node, NodePtr>(graph_submodule, "Node");  // NOLINT

  py::class_<FullyConnectedLayerNode, std::shared_ptr<FullyConnectedLayerNode>,
             Node>(graph_submodule, "FullyConnected")
      .def(py::init<uint64_t, ActivationFunction>(), py::arg("dim"),
           py::arg("activation"),
           "Constructs a dense FullyConnectedLayer object.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * activation: Enum specifying the activation function "
           "to use, no restrictions on case - We support five activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n")
      .def(py::init<uint64_t, std::string>(), py::arg("dim"),
           py::arg("activation"),
           "Constructs a dense FullyConnectedLayer object.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * activation: String specifying the activation function "
           "to use, no restrictions on case - We support five activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n")
      .def(py::init<uint64_t, float, ActivationFunction>(), py::arg("dim"),
           py::arg("sparsity"), py::arg("activation"),
           "Constructs a sparse FullyConnectedLayer object with sampling "
           "parameters autotuned.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * sparsity: Float - What fraction of nuerons to activate during "
           "training and sparse inference.\n"
           " * activation: Enum specifying the activation function "
           "to use, no restrictions on case - We support five activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n")
      .def(py::init<uint64_t, float, std::string>(), py::arg("dim"),
           py::arg("sparsity"), py::arg("activation"),
           "Constructs a sparse FullyConnectedLayer object with sampling "
           "parameters autotuned.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * sparsity: Float - What fraction of nuerons to activate during "
           "training and sparse inference.\n"
           " * activation: String specifying the activation function "
           "to use, no restrictions on case - We support five activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n")
#if THIRDAI_EXPOSE_ALL
      .def(py::init<uint64_t, float, ActivationFunction, SamplingConfig>(),
           py::arg("dim"), py::arg("sparsity"), py::arg("activation"),
           py::arg("sampling_config"),
           "Constructs a sparse FullyConnectedLayer object.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * sparsity: Float - What fraction of nuerons to activate during "
           "training and sparse inference.\n"
           " * activation: Enum specifying the activation function "
           "to use, no restrictions on case - We support five activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n"
           " * sampling_config (SamplingConfig) - Sampling config object to "
           "initialize hash tables/functions.")
      .def(py::init<uint64_t, float, std::string, SamplingConfig>(),
           py::arg("dim"), py::arg("sparsity"), py::arg("activation"),
           py::arg("sampling_config"),
           "Constructs a sparse FullyConnectedLayer object.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * sparsity: Float - What fraction of nuerons to activate during "
           "training and sparse inference.\n"
           " * activation: String specifying the activation function "
           "to use, no restrictions on case - We support five activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n"
           " * sampling_config (SamplingConfig) - Sampling config object to "
           "initialize hash tables/functions.")
#endif
      .def("__call__", &FullyConnectedLayerNode::addPredecessor,
           py::arg("prev_layer"),
           "Tells the graph which layer should act as input to this fully "
           "connected layer.");

  py::class_<Input, InputPtr, Node>(graph_submodule, "Input")
      .def(py::init<uint32_t>(), py::arg("dim"),
           "Constructs an input layer node for the graph.");

  py::class_<BoltGraph>(graph_submodule, "Model")
      .def(py::init<std::vector<InputPtr>, NodePtr>(), py::arg("inputs"),
           py::arg("output"),
           "Constructs a bolt model from a layer graph.\n"
           "Arguments:\n"
           " * inputs (List[Node]) - The input nodes to the graph. Note that "
           "inputs are mapped to input layers by their index.\n"
           " * output (Node) - The output node of the graph.")
      .def("compile", &BoltGraph::compile, py::arg("loss"),
           "Compiles the graph for the given loss function. In this step the "
           "order in which to compute the layers is determined and various "
           "checks are preformed to ensure the model architecture is correct.")
      .def("train", &BoltGraph::train<BoltBatch>, py::arg("train_data"),
           py::arg("train_labels"), py::arg("learning_rate"), py::arg("epochs"),
           py::arg("rebuild_hash_tables") = std::nullopt,
           py::arg("reconstruct_hash_functions") = std::nullopt,
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           "Trains the network on the given training data.\n"
           "Arguments:\n"
           " * train_data: BoltDataset - Training data. This is a BoltDataset "
           "as loaded by thirdai.dataset.load_bolt_svm_dataset or "
           "thirdai.dataset.load_bolt_csv_dataset.\n"
           " * train_labels: BoltDataset - Training labels. This is a "
           "BoltDataset as loaded by thirdai.dataset.load_bolt_svm_dataset or "
           "thirdai.dataset.load_bolt_csv_dataset.\n"
           " * learning_rate: Float (positive) - Learning rate.\n"
           " * epochs: Int (positive) - Number of training epochs over the "
           "training data.\n"
           " * rebuild_hash_tables: Int (positive) - Optional. Number of "
           "training samples before rebuilding hash tables. If not provided, "
           "BOLT will autotune this parameter.\n\n"
           " * reconstruct_hash_functions: Int (positive) - Optional. Number "
           "of training samples before generating new LSH hash functions. "
           "If not provided, BOLT will autotune this parameter.\n"
           " * metrics: List of str - Optional. The metrics to keep track of "
           "during training. See the section on metrics.\n"
           " * verbose: Boolean - Optional. If set to False, does not print "
           "anything during training. If set to True, prints additional "
           "information such as metrics and epoch times. Set to True by "
           "default.\n\n"

           "Returns a mapping from metric names to an array their values for "
           "every epoch.")
      .def("predict", &BoltGraph::predict<BoltBatch>, py::arg("test_data"),
           py::arg("test_labels"), py::arg("sparse_inference") = false,
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           py::arg("batch_limit") = std::numeric_limits<uint32_t>::max(),
           "Predicts the output given the input vectors and evaluates the "
           "predictions based on the given metrics.\n"
           "Arguments:\n"
           " * test_data: BoltDataset - Test data. This is a BoltDataset as "
           "loaded by thirdai.dataset.load_bolt_svm_dataset or "
           "thirdai.dataset.load_bolt_csv_dataset.\n"
           " * test_labels: BoltDataset - Test labels. This is a BoltDataset "
           "as loaded by thirdai.dataset.load_bolt_svm_dataset or "
           "thirdai.dataset.load_bolt_csv_dataset.\n"
           " * sparse_inference (Bool) - If true then sparse inference will be "
           "used.\n"
           " * metrics: List of str - Optional. The metrics to keep track of "
           "during training. "
           "See the section on metrics.\n"
           " * verbose: Boolean - Optional. If set to False, does not print "
           "anything during training. If set to True, prints additional "
           "information such as metrics and epoch times. Set to True by "
           "default.\n\n"

           "Returns a  a mapping from metric names to their values.");
}

void printMemoryWarning(uint64_t num_samples, uint64_t inference_dim) {
  std::cout << "Memory Error: Cannot allocate (" << num_samples << " x "
            << inference_dim
            << ") array for activations. Predict will return None for "
               "activations. Please breakup your test set into smaller pieces "
               "if you would like to have activations returned."
            << std::endl;
}

bool allocateActivations(uint64_t num_samples, uint64_t inference_dim,
                         uint32_t** active_neurons, float** activations,
                         bool output_sparse) {
  // We use a uint64_t here in case there is overflow when we multiply the two
  // quantities. If it's larger than a uint32_t then we skip allocating since
  // this would be a 16Gb array, and could potentially mess up indexing in other
  // parts of the code.
  uint64_t total_size = num_samples * inference_dim;
  if (total_size > std::numeric_limits<uint32_t>::max()) {
    printMemoryWarning(num_samples, inference_dim);
    return false;
  }
  try {
    if (output_sparse) {
      *active_neurons = new uint32_t[total_size];
    }
    *activations = new float[total_size];
    return true;
  } catch (std::bad_alloc& e) {
    printMemoryWarning(num_samples, inference_dim);
    return false;
  }
}

py::tuple constructNumpyArrays(py::dict&& py_metric_data, uint32_t num_samples,
                               uint32_t inference_dim, uint32_t* active_neurons,
                               float* activations, bool output_sparse,
                               bool alloc_success) {
  if (!alloc_success) {
    return py::make_tuple(py_metric_data, py::none());
  }

  // Deallocates the memory for the array since we are allocating it ourselves.
  py::capsule free_when_done_activations(
      activations, [](void* ptr) { delete static_cast<float*>(ptr); });

  py::array_t<float, py::array::c_style | py::array::forcecast>
      activations_array({num_samples, inference_dim},
                        {inference_dim * sizeof(float), sizeof(float)},
                        activations, free_when_done_activations);

  if (!output_sparse) {
    return py::make_tuple(py_metric_data, activations_array);
  }

  // Deallocates the memory for the array since we are allocating it ourselves.
  py::capsule free_when_done_active_neurons(
      active_neurons, [](void* ptr) { delete static_cast<uint32_t*>(ptr); });

  py::array_t<uint32_t, py::array::c_style | py::array::forcecast>
      active_neurons_array({num_samples, inference_dim},
                           {inference_dim * sizeof(uint32_t), sizeof(uint32_t)},
                           active_neurons, free_when_done_active_neurons);
  return py::make_tuple(py_metric_data, active_neurons_array,
                        activations_array);
}

}  // namespace thirdai::bolt::python