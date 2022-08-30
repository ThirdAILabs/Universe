#include "BoltPython.h"
#include "BoltGraphPython.h"
#include <bolt/src/auto_classifiers/MultiLabelTextClassifier.h>
#include <bolt/src/auto_classifiers/TabularClassifier.h>
#include <bolt/src/auto_classifiers/TextClassifier.h>
#include <bolt/src/auto_classifiers/sequential_classifier/SequentialClassifier.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace thirdai::bolt::python {

void createBoltSubmodule(py::module_& module) {
  auto bolt_submodule = module.def_submodule("bolt");

#if THIRDAI_EXPOSE_ALL
#pragma message("THIRDAI_EXPOSE_ALL is defined")                 // NOLINT
  py::class_<thirdai::bolt::SamplingConfig, SamplingConfigPtr>(  // NOLINT
      bolt_submodule, "SamplingConfig");

  py::class_<thirdai::bolt::DWTASamplingConfig,
             std::shared_ptr<DWTASamplingConfig>, SamplingConfig>(
      bolt_submodule, "DWTASamplingConfig")
      .def(py::init<uint32_t, uint32_t, uint32_t>(), py::arg("num_tables"),
           py::arg("hashes_per_table"), py::arg("reservoir_size"));

  py::class_<thirdai::bolt::FastSRPSamplingConfig,
             std::shared_ptr<FastSRPSamplingConfig>, SamplingConfig>(
      bolt_submodule, "FastSRPSamplingConfig")
      .def(py::init<uint32_t, uint32_t, uint32_t>(), py::arg("num_tables"),
           py::arg("hashes_per_table"), py::arg("reservoir_size"));
#endif

  // TODO(Geordie, Nicholas): put loss functions in its own submodule

  /*
    The second template argument to py::class_ specifies the holder class,
    which by default would be a std::unique_ptr.
    See: https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html

    The third template argument to py::class_ specifies the parent class if
    there is a polymorphic relationship.
    See: https://pybind11.readthedocs.io/en/stable/advanced/classes.html
  */
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
      .def("predict_single", &TextClassifier::predictSingle,
           py::arg("sentence"),
           "Given a sentence, predict the output class. \n"
           "Arguments:\n"
           " * sentence: Sentence to predict on text classifier.\n")
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

  py::class_<MultiLabelTextClassifier>(bolt_submodule,
                                       "MultiLabelTextClassifier")
      .def(py::init<uint32_t>(), py::arg("n_classes"),
           "Constructs a MultiLabelTextClassifier with autotuning.\n"
           "Arguments:\n"
           " * n_classes: int - How many classes or categories are in the "
           "labels of the dataset.\n")
      .def(
          "train", &MultiLabelTextClassifier::train, py::arg("train_file"),
          py::arg("epochs"), py::arg("learning_rate"),
          py::arg("metrics") = std::vector<std::string>(),
          "Trains the classifier on the given dataset.\n"
          "Arguments:\n"
          " * train_file: string - The path to the training dataset to use. "
          "The file should not have a header. Each row is formatted as follows:"
          "'''<label1>,<label2>,...,<labelN>\\t<text>'''"
          "where label1...labelN are integers."
          " * epochs: Int - How many epochs to train for.\n"
          " * learning_rate: Float - The learning rate to use for training.\n"
          " * metrics: List[string] - Metrics to use during training.\n")
      .def(
          "predict_single_from_sentence",
          [](MultiLabelTextClassifier& model, std::string sentence,
             float activation_threshold = 0.95) {
            auto output = model.predictSingleFromSentence(std::move(sentence),
                                                          activation_threshold);

            return denseBoltVectorToNumpy(output);
          },
          py::arg("sentence"), py::arg("activation_threshold") = 0.95,
          "Given a sentence, predict the likelihood of each output "
          "class. \n"
          "Arguments:\n"
          " * sentence: string - The input sentence.\n")
      .def(
          "predict_single_from_tokens",
          [](MultiLabelTextClassifier& model,
             const std::vector<uint32_t>& tokens,
             float activation_threshold = 0.95) {
            auto output =
                model.predictSingleFromTokens(tokens, activation_threshold);

            return denseBoltVectorToNumpy(output);
          },
          py::arg("tokens"), py::arg("activation_threshold") = 0.95,
          "Given a list of tokens, predict the likelihood of each output "
          "class. \n"
          "Arguments:\n"
          " * tokens: List[Int] - A list of integer tokens.\n")
      .def("predict", &MultiLabelTextClassifier::predict, py::arg("test_file"),
           py::arg("metrics") = std::vector<std::string>(),
           "Runs the classifier on the specified test dataset and optionally "
           "logs the prediction to a file.\n"
           "Arguments:\n"
           " * test_file: string - The path to the test dataset to use.\n"
           " * metrics: List[string] - Metrics to use during prediction.\n"
           "then the classifier will output the name of the class/category of "
           "each prediction this file with one prediction result on each "
           "line.\n")
      .def("save", &MultiLabelTextClassifier::save, py::arg("filename"),
           "Saves the classifier to a file. The file path must not require any "
           "folders to be created\n"
           "Arguments:\n"
           " * filename: string - The path to the save location of the "
           "classifier.\n")
      .def_static(
          "load", &MultiLabelTextClassifier::load, py::arg("filename"),
          "Loads and builds a saved classifier from file.\n"
          "Arguments:\n"
          " * filename: string - The location of the saved classifier.\n");

  py::class_<SequentialClassifier>(bolt_submodule, "SequentialClassifier",
                                   "Autoclassifier for sequential predictions.")
      .def(py::init<const std::pair<std::string, uint32_t>&,
                    const std::pair<std::string, uint32_t>&, const std::string&,
                    const std::vector<std::string>&,
                    const std::vector<std::pair<std::string, uint32_t>>&,
                    const std::vector<
                        std::tuple<std::string, uint32_t, uint32_t>>&>(),
           py::arg("user"), py::arg("target"), py::arg("timestamp"),
           py::arg("static_text") = std::vector<std::string>(),
           py::arg("static_categorical") =
               std::vector<std::pair<std::string, uint32_t>>(),
           py::arg("sequential") =
               std::vector<std::tuple<std::string, uint32_t, uint32_t>>())
      .def("train", &SequentialClassifier::train, py::arg("train_file"),
           py::arg("epochs"), py::arg("learning_rate"),
           py::arg("metrics") = std::vector<std::string>({"recall@1"}))
      .def("predict", &SequentialClassifier::predict, py::arg("test_file"),
           py::arg("metrics") = std::vector<std::string>({"recall@1"}),
           py::arg("output_file") = std::nullopt);

  py::class_<DistributedPyNetwork>(
      bolt_submodule, "DistributedNetwork",
      "Fully connected Distributed neural network.")
      .def(py::init<std::vector<
                        std::shared_ptr<thirdai::bolt::SequentialLayerConfig>>,
                    uint64_t>(),
           py::arg("layers"), py::arg("input_dim"),
           "Constructs a neural network for one node.\n"
           "Arguments:\n"
           " * layers: List of SequentialLayerConfig - Configurations for the "
           "sequence of "
           "layers in the neural network.\n"
           " * input_dim: Int (positive) - Dimension of input vectors in the "
           "dataset.")
      .def("prepareNodeForDistributedTraining",
           &DistributedPyNetwork::prepareNodeForDistributedTraining,
           py::arg("train_data"), py::arg("train_labels"),
           py::arg("rehash") = 0, py::arg("rebuild") = 0,
           py::arg("verbose") = true,
           "Initializes the Distributed Training over a node\n"
           "Arguments:\n"
           "Trains the network on the given training data.\n"
           "Arguments:\n"
           " * train_data: BoltDataset - Training data.\n"
           " * train_labels: BoltDataset - Training labels.\n"
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
           " some basic info "
           "If set to True, prints additional information about training"
           "Set to True by default.\n\n"

           "Returns number of batches to be processed.")
      .def("calculateGradientSingleNode",
           &DistributedPyNetwork::calculateGradientSingleNode,
           py::arg("batch_idx"), py::arg("loss_fn"),
           "Calculates the gradient for the network on the given training "
           "batch.\n"
           "Arguments:\n"
           " * batch: Int (positive) The batch number for which gradients are "
           "needed to be calcualted\n"
           " * loss_fn: LossFunction - The loss function to minimize.\n"

           "Returns void")
      .def("updateParametersSingleNode",
           &DistributedPyNetwork::updateParametersSingleNode,
           py::arg("learning_rate"),
           "Updates the parameters for the neural network using the "
           "gradients values already present\n"
           "Important: while using this function the updates are always"
           " DENSE"
           "Arguments:\n"
           " * learning rate: Float (positive) - Learning rate.\n"

           "Returns void")
      .def("predictSingleNode", &DistributedPyNetwork::predictSingleNode,
           py::arg("test_data"), py::arg("test_labels"),
           py::arg("sparse_inference") = false,
           py::arg("metrics") = std::vector<std::string>(),
           py::arg("verbose") = true,
           py::arg("batch_limit") = std::numeric_limits<uint32_t>::max(),
           "Predicts the output given the input vectors and evaluates the "
           "predictions based on the given metrics.\n"
           "Arguments:\n"
           " * test_data: BoltDataset - Test data.\n"
           " * test_labels: BoltDataset - Testing labels.\n"
           " * batch_size: (Int) - Testing Batch Size.\n"
           " * sparse_inference: (Bool) - When this is true the model will use "
           "sparsity in inference. This will lead to faster inference but can "
           " * metrics: List of str - Optional. The metrics to keep track of "
           "during training. See the section on metrics.\n"
           "cause a slight loss in accuracy. This option defaults to false.\n"
           " * verbose: Boolean - Optional. If set to False, only displays "
           "progress bar. "
           "If set to True, prints additional information such as metrics and "
           "inference times. "
           "Set to True by default.\n\n"

           "Returns a tuple consisting of (0) a mapping from metric names to "
           "their values "
           "and (1) output vectors (predictions) from the network in the form "
           "of a 2D Numpy matrix of floats.")
      .def("get_biases", &DistributedPyNetwork::getBiases,
           py::arg("layer_index"),
           "Returns the bias array at the given layer index as a 1D Numpy "
           "array.")
      .def("set_biases", &DistributedPyNetwork::setBiases,
           py::arg("layer_index"), py::arg("new_biases"),
           "Sets the bias array at the given layer index to the given 1D Numpy "
           "array.")
      .def("set_biases_gradients", &DistributedPyNetwork::setBiasesGradients,
           py::arg("layer_index"), py::arg("new_biases_gradients"),
           "Sets the bias gradient array at the given layer index to the given"
           "1D Numpy array.")
      .def("set_weights_gradients", &DistributedPyNetwork::setWeightGradients,
           py::arg("layer_index"), py::arg("new_weights_gradients"),
           "Sets the weights gradient array at the given layer index to the "
           "given 2D "
           "Numpy matrix. Throws an error if the dimension of the given weight "
           "matrix does not match the layer's current weight matrix.")
      .def("get_biases_gradients", &DistributedPyNetwork::getBiasesGradients,
           py::arg("layer_index"),
           "Returns the bias gradient array at the given layer index as a 1D "
           "Numpy array.")
      .def("get_weights_gradients", &DistributedPyNetwork::getWeightsGradients,
           py::arg("layer_index"),
           "Returns the weight gradient matrix at the given layer index as a "
           "2D Numpy "
           "matrix.")
      .def("get_weights", &DistributedPyNetwork::getWeights,
           py::arg("layer_index"),
           "Returns the weight matrix at the given layer index as a 2D Numpy "
           "matrix.")
      .def("set_weights", &DistributedPyNetwork::setWeights,
           py::arg("layer_index"), py::arg("new_weights"),
           "Sets the weight matrix at the given layer index to the given 2D "
           "Numpy matrix. Throws an error if the dimension of the given weight "
           "matrix does not match the layer's current weight matrix.")
      .def("freeze_hash_tables", &DistributedPyNetwork::freezeHashTables,
           "Freezes hash tables in the network. If you plan to use sparse "
           "inference, you may get a significant performance improvement if "
           "you call this one or two epochs before you finish training. "
           "Otherwise you should not call this method.");
  py::class_<TabularClassifier>(bolt_submodule, "TabularClassifier")
      .def(py::init<const std::string&, uint32_t>(), py::arg("model_size"),
           py::arg("n_classes"),
           "Constructs a TabularClassifier with autotuning.\n"
           "Arguments:\n"
           " * model_size: string - Either 'small', 'medium', 'large', or a "
           "size in Gb for the model, for example '6Gb' or '6 Gb'.\n"
           " * n_classes: int - How many classes or categories are in the "
           "labels of the dataset.\n")
      .def("train", &TabularClassifier::train, py::arg("train_file"),
           py::arg("column_datatypes"), py::arg("epochs"),
           py::arg("learning_rate"),
           "Trains the classifier on the given dataset.\n"
           "Arguments:\n"
           " * train_file: string - The path to the training dataset to use. "
           "Data is assumed to be in CSV format with ',' delimiter and no "
           "header. \n"
           " * column_datatypes: List of str - How to interpret data types of "
           "columns"
           " in the dataset. One of 'numeric', 'categorical', 'label'\n"
           " * epochs: Int - How many epochs to train for.\n"
           " * learning_rate: Float - The learning rate to use for training.\n")
      .def(
          "predict", &TabularClassifier::predict, py::arg("test_file"),
          py::arg("output_file") = std::nullopt,
          "Runs the classifier on the specified test dataset and optionally "
          "logs the prediction to a file.\n"
          "Arguments:\n"
          " * test_file: string - The path to the test dataset to use. Data is "
          "assumed to be in CSV format with ',' delimiter and no header. \n"
          " * output_file: string - Optional argument, if this is specified "
          "then the classifier will output the name of the class/category of "
          "each prediction this file with one prediction result on each "
          "line.\n")
      .def("predict_single", &TabularClassifier::predictSingle,
           py::arg("input_row"),
           "Given a list of input values excluding the label column, predict "
           "the class.\n"
           "Arguments:\n"
           " * input_row: List of strings representing input values.\n")
      .def("save", &TabularClassifier::save, py::arg("filename"),
           "Saves the classifier to a file. The file path must not require any "
           "folders to be created\n"
           "Arguments:\n"
           " * filename: string - The path to the save location of the "
           "classifier.\n")
      .def_static(
          "load", &TabularClassifier::load, py::arg("filename"),
          "Loads and builds a saved classifier from file.\n"
          "Arguments:\n"
          " * filename: string - The location of the saved classifier.\n");

  createBoltGraphSubmodule(bolt_submodule);
}

}  // namespace thirdai::bolt::python