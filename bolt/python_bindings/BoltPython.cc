#include "BoltPython.h"
#include "AutoClassifierBase.h"
#include "AutoClassifiers.h"
#include "BoltGraphPython.h"
#include <bolt/src/auto_classifiers/MultiLabelTextClassifier.h>
#include <bolt/src/auto_classifiers/sequential_classifier/SequentialClassifier.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <dataset/src/DataLoader.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace thirdai::bolt::python {

template <typename CLASSIFIER>
void defineAutoClassifierCommonMethods(py::class_<CLASSIFIER>& py_class) {
  py_class
      .def("train",
           py::overload_cast<const std::string&, uint32_t, float,
                             std::optional<uint32_t>, std::optional<uint32_t>>(
               &CLASSIFIER::train),
           py::arg("filename"), py::arg("epochs"), py::arg("learning_rate"),
           py::arg("batch_size") = std::nullopt,
           py::arg("max_in_memory_batches") = std::nullopt)
      .def("train",
           py::overload_cast<const std::shared_ptr<dataset::DataLoader>&,
                             uint32_t, float, std::optional<uint32_t>>(
               &CLASSIFIER::train),
           py::arg("data_source"), py::arg("epochs"), py::arg("learning_rate"),
           py::arg("max_in_memory_batches") = std::nullopt)
      .def("evaluate",
           py::overload_cast<const std::string&>(&CLASSIFIER::evaluate),
           py::arg("filename"))
      .def("evaluate",
           py::overload_cast<const std::shared_ptr<dataset::DataLoader>&>(
               &CLASSIFIER::evaluate),
           py::arg("data_source"))
      .def("predict", &CLASSIFIER::predict, py::arg("input"))
      .def("predict_batch", &CLASSIFIER::predictBatch, py::arg("inputs"))
      .def("save", &CLASSIFIER::save, py::arg("filename"))
      .def_static("load", &CLASSIFIER::load, py::arg("filename"));
}

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

  py::class_<RandomSamplingConfig, std::shared_ptr<RandomSamplingConfig>,
             SamplingConfig>(bolt_submodule, "RandomSamplingConfig")
      .def(py::init<>());
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

  py::class_<MarginBCE, std::shared_ptr<MarginBCE>, LossFunction>(
      bolt_submodule, "MarginBCE")
      .def(py::init<float>(), py::arg("margin"));

  /**
   * Text Classifier Definition
   */
  py::class_<TextClassifier> text_classifier(bolt_submodule, "TextClassifier");

  text_classifier.def(
      py::init<uint32_t, uint32_t>(), py::arg("internal_model_dim"),
      py::arg("n_classes"),
      "Constructs a TextClassifier with autotuning.\n"
      "Arguments:\n"
      " * internal_model_dim: int - Specifies the internal dimension used in "
      "the model.\n"
      " * n_classes: int - How many classes or categories are in the "
      "labels of the dataset.\n");

  defineAutoClassifierCommonMethods(text_classifier);

  /**
   * Multi Label Text Classifier Definition
   */
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

  /**
   * Tabular Classifier Definition
   */
  py::class_<TabularClassifier> tabular_classifier(bolt_submodule,
                                                   "TabularClassifier");

  tabular_classifier.def(
      py::init<uint32_t, uint32_t, std::vector<std::string>>(),
      py::arg("internal_model_dim"), py::arg("n_classes"),
      py::arg("column_datatypes"));

  defineAutoClassifierCommonMethods(tabular_classifier);

  /**
   * Binary Text Classifier
   */
  py::class_<BinaryTextClassifier> binary_text_classifier(
      bolt_submodule, "BinaryTextClassifier");

  binary_text_classifier.def(
      py::init<uint32_t, uint32_t, std::optional<float>, bool>(),
      py::arg("n_outputs"), py::arg("internal_model_dim"),
      py::arg("sparsity") = std::nullopt,
      py::arg("use_sparse_inference") = true);

  defineAutoClassifierCommonMethods(binary_text_classifier);

  /**
   * Sequential Classifier
   */
  py::class_<SequentialClassifier>(bolt_submodule, "SequentialClassifier",
                                   "Autoclassifier for sequential predictions.")
      .def(py::init<
               const std::pair<std::string, uint32_t>&,
               const std::pair<std::string, uint32_t>&, const std::string&,
               const std::vector<std::string>&,
               const std::vector<std::pair<std::string, uint32_t>>&,
               const std::vector<std::tuple<std::string, uint32_t, uint32_t>>&,
               std::optional<char>>(),
           py::arg("user"), py::arg("target"), py::arg("timestamp"),
           py::arg("static_text") = std::vector<std::string>(),
           py::arg("static_categorical") =
               std::vector<std::pair<std::string, uint32_t>>(),
           py::arg("sequential") =
               std::vector<std::tuple<std::string, uint32_t, uint32_t>>(),
           py::arg("multi_class_delim") = std::nullopt)
      .def("train", &SequentialClassifier::train, py::arg("train_file"),
           py::arg("epochs"), py::arg("learning_rate"),
           py::arg("metrics") = std::vector<std::string>({"recall@1"}))
#if THIRDAI_EXPOSE_ALL
      .def("summarizeModel", &SequentialClassifier::summarizeModel)
#endif
      .def("predict", &SequentialClassifier::predict, py::arg("test_file"),
           py::arg("metrics") = std::vector<std::string>({"recall@1"}),
           py::arg("output_file") = std::nullopt, py::arg("print_last_k") = 1)
      .def("save", &SequentialClassifier::save, py::arg("filename"))
      .def_static("load", &SequentialClassifier::load, py::arg("filename"));

  createBoltGraphSubmodule(bolt_submodule);
}

}  // namespace thirdai::bolt::python