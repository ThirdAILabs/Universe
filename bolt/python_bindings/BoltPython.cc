#include "BoltPython.h"
#include "BoltGraphPython.h"
#include <bolt/python_bindings/ConversionUtils.h>
#include <bolt/src/auto_classifiers/sequential_classifier/SequentialClassifier.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <dataset/src/DataLoader.h>
#include <dataset/src/batch_processors/TabularMetadataProcessor.h>
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

py::module_ createBoltSubmodule(py::module_& module) {
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

  /**
   * Sequential Classifier
   */
  py::class_<SequentialClassifier>(bolt_submodule, "SequentialClassifier",
                                   "Autoclassifier for sequential predictions.")
      .def(
          py::init<
              const std::pair<std::string, uint32_t>,
              const std::pair<std::string, uint32_t>, const std::string,
              const std::vector<std::string>,
              const std::vector<std::pair<std::string, uint32_t>>,
              const std::vector<std::tuple<std::string, uint32_t, uint32_t>>,
              const std::vector<std::string>, std::optional<char>, std::string,
              std::optional<uint32_t>, std::optional<uint32_t>>(),
          py::arg("user"), py::arg("label"), py::arg("timestamp"),
          py::arg("static_text") = std::vector<std::string>(),
          py::arg("static_category") =
              std::vector<std::pair<std::string, uint32_t>>(),
          py::arg("track_categories") =
              std::vector<std::tuple<std::string, uint32_t, uint32_t>>(),
          py::arg("track_quantities") = std::vector<std::string>(),
          py::arg("multi_class_delim") = std::nullopt,
          py::arg("time_granularity") = "daily",
          py::arg("time_to_predict_ahead") = std::nullopt,
          py::arg("history_length_for_inference") = std::nullopt,
          "Constructs a SequentialClassifier.\n"
          "Arguments:\n"
          " * user: Tup[str, int] - Column name for user IDs and the number of "
          "unique user IDs.\n"
          " * label: Tup[str, int] - Column name for label IDs and the number "
          "of unique IDs.\n"
          " * timestamp: str - Column name for timestamps. Timestamps must be "
          "in YYYY-MM-DD format.\n"
          " * static_text (optional): List[str] - List of column names for "
          "static text information.\n"
          " * static_category (optional): List[Tup[str, int]] - List of "
          "(column name, num unique categories) pairs for static categorical "
          "features.\n"
          " * track_categories (optional): List[Tup[str, int, int]] - List of "
          "(column name, num unique categories, max sequence length) triplets "
          "for trackable categorical features. SequentialClassifier tracks the "
          "last max_sequence_length categories associated with a user ID.\n"
          " * track_quantities (optional): List[str] - List of column names "
          "for trackable numerical features.\n",
          " * multi_class_delim (optional): str - A single character to "
          "delimit multi-class categorical feature columns. This delimiter "
          "applies to columns specified in `static_category` and "
          "`track_categories`. Defaults to None.\n"
          " * time_granularity (optional): str - The granularity of quantity "
          "tracking. Options: 'daily'/'d', 'weekly'/'w', 'biweekly'/'b', "
          "'monthly'/'m'. E.g. If granularity is 'weekly' and there is a "
          "record of a $50 transaction on Monday and a $100 transaction on "
          "Tuesday, SequentialClassifier will treat this as a $150 transaction "
          "during the week.\n"
          " * time_to_predict_ahead (required only if track_quantities is "
          "non-empty): int - How far ahead the model needs to learn to "
          "predict. Time unit is in terms of the selected `time_granulartiy`. "
          "E.g. time_to_predict_ahead=5 and granularity='weekly' means the "
          "model learns to predict 5 weeks ahead.\n"
          " * history_length_for_inference (required only if track_quantities "
          "is non-empty): int - The length of history of tracked quantities "
          "that the model can use to make predictions. Length is in terms of "
          "the selected `time_granulartiy`. E.g. "
          "history_length_for_inference=5 and granularity='weekly' means the "
          "model uses the last 5 weeks of counts to make predictions.\n")
      .def("train", &SequentialClassifier::train, py::arg("train_file"),
           py::arg("epochs"), py::arg("learning_rate"),
           py::arg("metrics") = std::vector<std::string>({"recall@1"}),
           "Trains a Sequential classifier Model using the data provided in "
           "train_file"
           "Arguments:\n"
           " * train_file: String - The path to the train file\n"
           " * epochs: int - Number of epochs you want to train the model\n"
           " * learning_rate: float - learning rate\n"
           " * metrics: List[Str] - What all metrics you want to track during "
           "training default to 'recall@1'\n"
           "Example: let classfier is SequentialClassifier Object which "
           "generated after initialization then "
           " classifier.train(train_file = '/home/train_file.csv',epochs = "
           "10,learning_rate = 0.001). ")
#if THIRDAI_EXPOSE_ALL
      .def("summarizeModel", &SequentialClassifier::summarizeModel,
           "Deprecated\n")
#endif
      .def("predict", &SequentialClassifier::predict, py::arg("test_file"),
           py::arg("metrics") = std::vector<std::string>({"recall@1"}),
           py::arg("output_file") = std::nullopt, py::arg("print_last_k") = 1,
           "Predicts t")
      .def("predict_single", &SequentialClassifier::predictSingle,
           py::arg("input_sample"), py::arg("top_k") = 1,
           "Computes the top k classes and their probabilities for a single "
           "input sample. "
           "Returns a list of (class name. probability) pairs\n"
           "Arguments:\n"
           " * sample: Dict[str, str] - The input sample as a dictionary where "
           "the keys "
           "are column names as specified in the schema and the values are the "
           "respective "
           "column values.\n"
           " * k: Int (positive) - The number of top results to return.\n")
      .def("save", &SequentialClassifier::save, py::arg("filename"))
      .def_static("load", &SequentialClassifier::load, py::arg("filename"))
      .def("explain", &SequentialClassifier::explain, py::arg("input_sample"),
           py::arg("neuron_to_explain") = std::optional<uint32_t>());

  createBoltGraphSubmodule(bolt_submodule);

  return bolt_submodule;
}

}  // namespace thirdai::bolt::python