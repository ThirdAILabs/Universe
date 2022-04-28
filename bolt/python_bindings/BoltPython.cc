#include "BoltPython.h"
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

namespace thirdai::bolt::python {

void createBoltSubmodule(py::module_& module) {
  auto bolt_submodule = module.def_submodule("bolt");

  py::class_<thirdai::bolt::SamplingConfig>(
      bolt_submodule, "SamplingConfig",
      "SamplingConfig represents a layer's sampling hyperparameters.")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
           py::arg("hashes_per_table"), py::arg("num_tables"),
           py::arg("range_pow"), py::arg("reservoir_size"),
           "Builds a SamplingConfig object. range_pow must always be 3 * "
           "hashes_per_table.")
      .def(py::init<>(), "Builds a SamplingConfig object.");

  py::enum_<ActivationFunction>(
      bolt_submodule, "ActivationFunctions",
      "An enum of all available activation functions. To use it, pass it to "
      "the 'activation_function' parameter of a LayerConfig object.")
      .value("ReLU", ActivationFunction::ReLU,
             "Rectified Linear Units (ReLU) activation function; "
             "introduces non-linearity to the neural network.")
      .value("Linear", ActivationFunction::Linear,
             "Returns the outputs of a layer as-is.")
      .value("Softmax", ActivationFunction::Softmax,
             "Softmax activation function; converts logits to classification "
             "probabilities. Currently, this activation function can only be "
             "applied to the final layer in the neural network.");

  bolt_submodule.def("getActivationFunction", &getActivationFunction,
                     py::arg("name"),
                     "Converts an activation function name to "
                     "the corresponding enum.");

  // TODO(Geordie, Nicholas): put loss functions in its own submodule
  py::class_<LossFunction>(bolt_submodule, "LossFunction",
                           "Base class for all loss functions");  // NOLINT

  py::class_<CategoricalCrossEntropyLoss, LossFunction>(
      bolt_submodule, "CategoricalCrossEntropyLoss",
      "A loss function for classificaiton tasks.")
      .def(py::init<>(), "Constructs a CategoricalCrossEntropyLoss object.");

  py::class_<MeanSquaredError, LossFunction>(
      bolt_submodule, "MeanSquaredError",
      "A loss function that minimizes mean squared error (MSE) for regression "
      "tasks. "
      "MSE = sum( (actual - prediction)^2 )")
      .def(py::init<>(), "Constructs a MeanSquaredError object.");

  py::class_<WeightedMeanAbsolutePercentageErrorLoss, LossFunction>(
      bolt_submodule, "WeightedMeanAbsolutePercentageError",
      "A loss function to minimize weighted mean absolute percentage error "
      "(WMAPE) "
      "for regression tasks. WMAPE = 100% * sum(|actual - prediction|) / "
      "sum(|actual|)")
      .def(py::init<>(),
           "Constructs a WeightedMeanAbsolutePercentageError object.");

  py::class_<thirdai::bolt::FullyConnectedLayerConfig>(
      bolt_submodule, "LayerConfig", "Defines a fully-connected layer.\n")
      .def(py::init<uint64_t, float, ActivationFunction,
                    thirdai::bolt::SamplingConfig>(),
           py::arg("dim"), py::arg("load_factor"),
           py::arg("activation_function"), py::arg("sampling_config"),
           "Constructs the LayerConfig object.\n"
           "Arguments:\n"
           "  dim: Int - The dimension of the layer.\n"
           "  load_factor: Float - The fraction of neurons to use during "
           "sparse training "
           "and sparse inference. For example, load_factor=0.05 means the "
           "layer uses 5% of "
           "its neurons when processing an individual sample.\n"
           "  activation_function: ActivationFunctions enum - We support three "
           "activation "
           "functions: ReLU, Softmax, and Linear.\n"
           "  sampling_config: SamplingConfig - sampling configuration.")
      .def(py::init<uint64_t, ActivationFunction>(), py::arg("dim"),
           py::arg("activation_function"),
           "Constructs the LayerConfig object for a dense layer.\n"
           "Arguments:\n"
           "  dim: Int - The dimension of the layer.\n"
           "  activation_function: ActivationFunctions enum - We support three "
           "activation "
           "functions: ReLU, Softmax, and Linear.")
      .def(py::init<uint64_t, float, ActivationFunction>(), py::arg("dim"),
           py::arg("load_factor"), py::arg("activation_function"),
           "Constructs the LayerConfig object for a sparse layer with a "
           "default sampling "
           "configuration.\n"
           "Arguments:\n"
           "  dim: Int - The dimension of the layer.\n"
           "  activation_function: ActivationFunctions enum - We support three "
           "activation "
           "functions: ReLU, Softmax, and Linear.");

  py::class_<thirdai::bolt::EmbeddingLayerConfig>(
      bolt_submodule, "EmbeddingLayerConfig",
      "Defines a space-efficient embedding table lookup layer.")
      .def(py::init<uint32_t, uint32_t, uint32_t>(),
           py::arg("num_embedding_lookups"), py::arg("lookup_size"),
           py::arg("log_embedding_block_size"),
           "Constructs an embedding layer.\n"
           "Arguments:\n"
           "  num_embedding_lookups: Int - The number of embedding table "
           "lookups per categorical feature.\n"
           "  lookup_size: Int - Embedding dimension.\n"
           "  log_embedding_block_size: Int - log (base 2) of the size of the "
           "embedding table.");

  py::class_<PyNetwork>(bolt_submodule, "Network",
                        "Fully connected neural network.")
      .def(py::init<std::vector<thirdai::bolt::FullyConnectedLayerConfig>,
                    uint64_t>(),
           py::arg("layers"), py::arg("input_dim"),
           "Constructs the neural network.\n"
           "Arguments:\n"
           "  layers: List of LayerConfig - Configurations for the sequence of "
           "layers in the neural network.\n"
           "  input_dim: Int - Dimension of input vectors in the dataset.")
      .def("train", &PyNetwork::train, py::arg("train_data"),
           py::arg("loss_fn"), py::arg("learning_rate"), py::arg("epochs"),
           py::arg("rehash") = 0, py::arg("rebuild") = 0,
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           "Trains the network on the given training data.\n"
           "Arguments:\n"
           "  train_data: BoltDataset - Training data.\n"
           "  loss_fn: LossFunction - The loss function to minimize.\n"
           "  learning_rate: Float - Learning rate.\n"
           "  epochs: Int - Number of training epochs over the training data.\n"
           "  rehash: Int - Optional. Number of training samples before "
           "rehashing neurons. "
           "If not provided, BOLT will autotune this parameter.\n\n"
           "BOLT's sparse training works by applying smart hash functions to "
           "all neurons in the "
           "network, and they have to be rehashed periodically. This parameter "
           "sets the frequency "
           "of rehashing.\n"
           "  rebuild: Int - Optional. Number of training samples before "
           "rebuilding hash tables "
           "and generating new smart hash functions. If not provided, BOLT "
           "will autotune this parameter.\n"
           "  metrics: Optional. The metrics to keep track of during training. "
           "Options:\n"
           "    categorical_accuracy\n"
           "    weighted_mean_absolute_percentage_error\n"
           "  verbose: Boolean - Optional. If set to False, only displays "
           "progress bar. "
           "If set to True, prints additional information such as metrics and "
           "epoch times. "
           "Set to True by default.\n\n"

           "Returns a tuple consisting of (0) a mapping from metric names to "
           "an array their values for every epoch "
           "and (1) output vectors from the network in the form of a 2D Numpy "
           "matrix of floats.")
      .def("train", &PyNetwork::trainWithDenseNumpyArray,
           py::arg("train_examples"), py::arg("train_labels"),
           py::arg("batch_size"), py::arg("loss_fn"), py::arg("learning_rate"),
           py::arg("epochs"), py::arg("rehash") = 0, py::arg("rebuild") = 0,
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           "Trains the network on the given training data. This particular "
           "overload supports dense input "
           "vectors and categorical labels.\n"
           "Arguments:\n"
           "  train_examples: 2D Numpy matrix of floats - Training examples "
           "(input vectors).\n"
           "  train_labels: 1D Numpy array of integers - Categorical labels "
           "for training examples.\n"
           "  batch_size: Int - Size of training data batches.\n"
           "  loss_fn: LossFunction - The loss function to minimize.\n"
           "  learning_rate: Float - Learning rate.\n"
           "  epochs: Int - Number of training epochs over the training data.\n"
           "  rehash: Int - Optional. Number of training samples before "
           "rehashing neurons. "
           "If not provided, BOLT will autotune this parameter.\n\n"
           "BOLT's sparse training works by applying smart hash functions to "
           "all neurons in the "
           "network, and they have to be rehashed periodically. This parameter "
           "sets the frequency "
           "of rehashing.\n"
           "  rebuild: Int - Optional. Number of training samples before "
           "rebuilding hash tables "
           "and generating new smart hash functions. If not provided, BOLT "
           "will autotune this parameter.\n"
           "  metrics: Optional. The metrics to keep track of during training. "
           "Options:\n"
           "    categorical_accuracy\n"
           "    weighted_mean_absolute_percentage_error\n"
           "  verbose: Boolean - Optional. If set to False, only displays "
           "progress bar. "
           "If set to True, prints additional information such as metrics and "
           "epoch times. "
           "Set to True by default.\n\n"

           "Returns a tuple consisting of (0) a mapping from metric names to "
           "an array their values for every epoch "
           "and (1) output vectors from the network in the form of a 2D Numpy "
           "matrix of floats.")
      .def("train", &PyNetwork::trainWithSparseNumpyArray, py::arg("x_idxs"),
           py::arg("x_vals"), py::arg("x_offsets"), py::arg("y_idxs"),
           py::arg("y_vals"), py::arg("y_offsets"), py::arg("batch_size"),
           py::arg("loss_fn"), py::arg("learning_rate"), py::arg("epochs"),
           py::arg("rehash") = 0, py::arg("rebuild") = 0,
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           "Trains the network on the given training data. This particular "
           "overload supports sparse input "
           "vectors and sparse label vectors.\n\n"

           "Suppose we have N sparse vectors of floats. We represent this "
           "using three arrays:\n"
           "  Indices: 1D Numpy array of integers - Indices of nonzero "
           "elements in each vector, concatenated. "
           "For example, given the vectors {0.0, 1.5, 0.0, 9.0} and {0.0, 0.0, "
           "0.0, 4.0}, the corresponding "
           "'indices' array is [1, 3, 3]\n"
           "  Values: 1D Numpy array of floats - The values of nonzero "
           "elements in each vector, concatenated. "
           "For example, given the vectors {0.0, 1.5, 0.0, 9.0} and {0.0, 0.0, "
           "0.0, 4.0}, the corresponding "
           "'values' array is [1.5, 9.0, 4.0]\n"
           "  Offsets: 1D Numpy array of integers - The i-th element of this "
           "array is the number of nonzero "
           "elements in the i - 1-th vector. The first element is 0. "
           "Effectively, this array maps each vector to the corresponding "
           "entries in 'indices' and 'values'. "
           "Specifically, offsets[i] is the starting position of vector i in "
           "the 'indices' and 'values' arrays, "
           "while offsets[i] is the stopping position (exclusive). "
           "This means that indices[offsets[i], offsets[i + 1]] contain the "
           "indices of nonzero elements of "
           "vector i, and values[offsets[i], offsets[i + 1]] contain the "
           "values of nonzero elements of vector i. "
           "For example, given the vectors {0.0, 1.5, 0.0, 9.0} and {0.0, 0.0, "
           "0.0, 4.0}, the corresponding "
           "'offsets' array is [0, 2, 1]\n\n"

           "Arguments:\n"
           "  x_idxs: 1D Numpy array of integers - Indices array for training "
           "examples (input vectors).\n"
           "  x_vals: 1D Numpy array of floats - Values array of training "
           "examples (input vectors).\n"
           "  x_offsets: 1D Numpy array of integers - Offsets array for "
           "training examples (input vectors).\n"
           "  y_idxs: 1D Numpy array of integers - Indices array for training "
           "labels (ground truth vectors).\n"
           "  y_vals: 1D Numpy array of floats - Values array of training "
           "labels (ground truth vectors).\n"
           "  y_offsets: 1D Numpy array of integers - Offsets array for "
           "training labels (ground truth vectors).\n"
           "  loss_fn: LossFunction - The loss function to minimize.\n"
           "  learning_rate: Float - Learning rate.\n"
           "  epochs: Int - Number of training epochs over the training data.\n"
           "  rehash: Int - Optional. Number of training samples before "
           "rehashing neurons. "
           "If not provided, BOLT will autotune this parameter.\n\n"
           "BOLT's sparse training works by applying smart hash functions to "
           "all neurons in the "
           "network, and they have to be rehashed periodically. This parameter "
           "sets the frequency "
           "of rehashing.\n"
           "  rebuild: Int - Optional. Number of training samples before "
           "rebuilding hash tables "
           "and generating new smart hash functions. If not provided, BOLT "
           "will autotune this parameter.\n"
           "  metrics: Optional. The metrics to keep track of during training. "
           "Options:\n"
           "    categorical_accuracy\n"
           "    weighted_mean_absolute_percentage_error\n"
           "  verbose: Boolean - Optional. If set to False, only displays "
           "progress bar. "
           "If set to True, prints additional information such as metrics and "
           "epoch times. "
           "Set to True by default.\n\n"

           "Returns a tuple consisting of (0) a mapping from metric names to "
           "an array their values for every epoch "
           "and (1) output vectors from the network in the form of a 2D Numpy "
           "matrix of floats.")
      .def("predict", &PyNetwork::predict, py::arg("test_data"),
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           py::arg("batch_limit") = std::numeric_limits<uint32_t>::max(),
           "Predicts the output given the input vectors and evaluates the "
           "predictions based on the given metrics.\n"
           "Arguments:\n"
           "  test_data: BoltDataset - Test data.\n"
           "  metrics: Optional. The metrics to evaluate. Options:\n"
           "    categorical_accuracy\n"
           "    weighted_mean_absolute_percentage_error\n"
           "  verbose: Boolean - Optional. If set to False, only displays "
           "progress bar. "
           "If set to True, prints additional information such as metrics and "
           "inference times. "
           "Set to True by default.\n\n"

           "Returns a tuple consisting of (0) a mapping from metric names to "
           "their values "
           "and (1) output vectors (predictions) from the network in the form "
           "of a 2D Numpy matrix of floats.")
      .def("predict", &PyNetwork::predictWithDenseNumpyArray,
           py::arg("test_examples"), py::arg("test_labels"),
           py::arg("batch_size"),
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           py::arg("batch_limit") = std::numeric_limits<uint32_t>::max(),
           "Predicts the output given the input vectors and evaluates the "
           "predictions based on the given metrics.\n"
           "Arguments:\n"
           "  test_examples: 2D Numpy matrix of floats - Test examples (input "
           "vectors).\n"
           "  test_labels: 1D Numpy array of integers - Categorical labels for "
           "test examples.\n"
           "  batch_size: Int - Size of training data batches.\n"
           "  metrics: Optional. The metrics to evaluate. Options:\n"
           "    categorical_accuracy\n"
           "    weighted_mean_absolute_percentage_error\n"
           "  verbose: Boolean - Optional. If set to False, only displays "
           "progress bar. "
           "If set to True, prints additional information such as metrics and "
           "inference times. "
           "Set to True by default.\n\n"

           "Returns a tuple consisting of (0) a mapping from metric names to "
           "their values "
           "and (1) output vectors (predictions) from the network in the form "
           "of a 2D Numpy matrix of floats.")
      .def("predict", &PyNetwork::predictWithSparseNumpyArray,
           py::arg("x_idxs"), py::arg("x_vals"), py::arg("x_offsets"),
           py::arg("y_idxs"), py::arg("y_vals"), py::arg("y_offsets"),
           py::arg("batch_size"),
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           py::arg("batch_limit") = std::numeric_limits<uint32_t>::max(),
           "Predicts the output given the input vectors and evaluates the "
           "predictions based on the given metrics.\n\n"

           "Suppose we have N sparse vectors of floats. We represent this "
           "using three arrays:\n"
           "  Indices: 1D Numpy array of integers - Indices of nonzero "
           "elements in each vector, concatenated. "
           "For example, given the vectors {0.0, 1.5, 0.0, 9.0} and {0.0, 0.0, "
           "0.0, 4.0}, the corresponding "
           "'indices' array is [1, 3, 3]\n"
           "  Values: 1D Numpy array of floats - The values of nonzero "
           "elements in each vector, concatenated. "
           "For example, given the vectors {0.0, 1.5, 0.0, 9.0} and {0.0, 0.0, "
           "0.0, 4.0}, the corresponding "
           "'values' array is [1.5, 9.0, 4.0]\n"
           "  Offsets: 1D Numpy array of integers - The i-th element of this "
           "array is the number of nonzero "
           "elements in the i - 1-th vector. The first element is 0. "
           "Effectively, this array maps each vector to the corresponding "
           "entries in 'indices' and 'values'. "
           "Specifically, offsets[i] is the starting position of vector i in "
           "the 'indices' and 'values' arrays, "
           "while offsets[i] is the stopping position (exclusive). "
           "This means that indices[offsets[i], offsets[i + 1]] contain the "
           "indices of nonzero elements of "
           "vector i, and values[offsets[i], offsets[i + 1]] contain the "
           "values of nonzero elements of vector i. "
           "For example, given the vectors {0.0, 1.5, 0.0, 9.0} and {0.0, 0.0, "
           "0.0, 4.0}, the corresponding "
           "'offsets' array is [0, 2, 1]\n\n"

           "Arguments:\n"
           "  test_examples: 2D Numpy matrix of floats - Test examples (input "
           "vectors).\n"
           "  test_labels: 1D Numpy array of integers - Categorical labels for "
           "test examples.\n"
           "  batch_size: Int - Size of training data batches.\n"
           "  metrics: Optional. The metrics to evaluate. Options:\n"
           "    categorical_accuracy\n"
           "    weighted_mean_absolute_percentage_error\n"
           "  verbose: Boolean - Optional. If set to False, only displays "
           "progress bar. "
           "If set to True, prints additional information such as metrics and "
           "inference times. "
           "Set to True by default.\n\n"

           "Returns a tuple consisting of (0) a mapping from metric names to "
           "their values "
           "and (1) output vectors (predictions) from the network in the form "
           "of a 2D Numpy matrix of floats.")
      .def("enable_sparse_inference", &PyNetwork::enableSparseInference,
           "Enables sparse inference. Freezes smart hash tables. Do not call "
           "this method early on "
           "in the training routine. It is recommended to call this method "
           "right before the last training "
           "epoch.")
      .def("save", &PyNetwork::save, py::arg("filename"),
           "Saves the network to a file.")
      .def_static("load", &PyNetwork::load, py::arg("filename"),
                  "Loads and builds a saved network from file.")
      .def("get_weights", &PyNetwork::getWeights, py::arg("layer_index"),
           "Returns the weight matrix at the given layer index as a 2D Numpy "
           "matrix.")
      .def("set_weights", &PyNetwork::setWeights, py::arg("layer_index"),
           py::arg("new_weights"),
           "Sets the weight matrix at the given layer index to the given 2D "
           "Numpy matrix.")
      .def("get_biases", &PyNetwork::getBiases, py::arg("layer_index"),
           "Returns the bias array at the given layer index as a 1D Numpy "
           "array.")
      .def("set_biases", &PyNetwork::setBiases, py::arg("layer_index"),
           py::arg("new_biases"),
           "Sets the bias array at the given layer index to the given 1D Numpy "
           "array.");

  py::class_<PyDLRM>(bolt_submodule, "DLRM",
                     "DLRM network with space-efficient embedding tables.")
      .def(py::init<thirdai::bolt::EmbeddingLayerConfig,
                    std::vector<thirdai::bolt::FullyConnectedLayerConfig>,
                    std::vector<thirdai::bolt::FullyConnectedLayerConfig>,
                    uint32_t>(),
           py::arg("embedding_layer"), py::arg("bottom_mlp"),
           py::arg("top_mlp"), py::arg("input_dim"),
           "Constructs the DLRM.\n"
           "Arguments:\n"
           "  embedding_layer: EmbeddingLayerConfig - Configuration of the "
           "embedding layer.\n"
           "  bottom_mlp: List of LayerConfig - Configurations of the sequence "
           "of layers in DLRM's bottom MLP.\n"
           "  top_mlp: List of LayerConfig - Configurations of the sequence of "
           "layers in DLRM's top MLP.\n"
           "  input_dim: Int - Dimension of input vectors in the dataset.")
      .def("train", &PyDLRM::train, py::arg("train_data"), py::arg("loss_fn"),
           py::arg("learning_rate"), py::arg("epochs"), py::arg("rehash") = 0,
           py::arg("rebuild") = 0,
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           "Trains the network on the given training data.\n"
           "Arguments:\n"
           "  train_data: ClickthroughDataset - Training data.\n"
           "  loss_fn: LossFunction - The loss function to minimize.\n"
           "  learning_rate: Float - Learning rate.\n"
           "  epochs: Int - Number of training epochs over the training data.\n"
           "  rehash: Int - Optional. Number of training samples before "
           "rehashing neurons. "
           "If not provided, BOLT will autotune this parameter.\n\n"
           "BOLT's sparse training works by applying smart hash functions to "
           "all neurons in the "
           "network, and they have to be rehashed periodically. This parameter "
           "sets the frequency "
           "of rehashing.\n"
           "  rebuild: Int - Optional. Number of training samples before "
           "rebuilding hash tables "
           "and generating new smart hash functions. If not provided, BOLT "
           "will autotune this parameter.\n"
           "  metrics: Optional. The metrics to keep track of during training. "
           "Options:\n"
           "    categorical_accuracy\n"
           "    weighted_mean_absolute_percentage_error\n"
           "  verbose: Boolean - Optional. If set to False, only displays "
           "progress bar. "
           "If set to True, prints additional information such as metrics and "
           "epoch times. "
           "Set to True by default.\n\n"

           "Returns a tuple consisting of (0) a mapping from metric names to "
           "an array their values for every epoch "
           "and (1) output vectors from the network in the form of a 2D Numpy "
           "matrix of floats.")
      .def("predict", &PyDLRM::predict, py::arg("test_data"),
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           py::arg("batch_limit") = std::numeric_limits<uint32_t>::max(),
           "Predicts the output given the input vectors and evaluates the "
           "predictions based on the given metrics.\n"
           "Arguments:\n"
           "  test_data: ClickthroughDataset - Test data.\n"
           "  metrics: Optional. The metrics to evaluate. Options:\n"
           "    categorical_accuracy\n"
           "    weighted_mean_absolute_percentage_error\n"
           "  verbose: Boolean - Optional. If set to False, only displays "
           "progress bar. "
           "If set to True, prints additional information such as metrics and "
           "inference times. "
           "Set to True by default.\n\n"

           "Returns a tuple consisting of (0) a mapping from metric names to "
           "their values "
           "and (1) output vectors (predictions) from the network in the form "
           "of a 2D Numpy matrix of floats.");
}

}  // namespace thirdai::bolt::python
