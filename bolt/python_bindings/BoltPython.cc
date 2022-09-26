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
              const std::vector<std::string>, std::string,
              std::optional<uint32_t>, std::optional<uint32_t>,
              std::optional<char>>(),
          py::arg("user"), py::arg("label"), py::arg("timestamp"),
          py::arg("static_text") = std::vector<std::string>(),
          py::arg("static_category") =
              std::vector<std::pair<std::string, uint32_t>>(),
          py::arg("track_categories") =
              std::vector<std::tuple<std::string, uint32_t, uint32_t>>(),
          py::arg("track_quantities") = std::vector<std::string>(),
          py::arg("time_granularity") = "daily",
          py::arg("time_to_predict_ahead") = std::nullopt,
          py::arg("history_length_for_inference") = std::nullopt,
          py::arg("multi_class_delim") = std::nullopt,
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
          "for trackable numerical features.\n"
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
          "model uses the last 5 weeks of counts to make predictions.\n"
          " * multi_class_delim (optional): str - A single character to "
          "delimit multi-class categorical feature columns. This delimiter "
          "applies to columns specified in `static_category` and "
          "`track_categories`. Defaults to None.\n"
          "Example:\n"
          "```\n"
          "from thirdai import bolt\n\n"
          "model = bolt.SequentialClassifier(\n"
          "    # Required arguments"
          "    user=('userId', 50), # (col name, n unique users)\n"
          "    target=('movieId', 50), # (col name, n unique users)\n"
          "    timestamp='timestamp',\n\n"
          "    # Static features\n"
          "    static_text=['userBio'],\n"
          "    static_category=['userAgeGroup'],\n\n"
          "    # For each user, track last 5 items, 50 unique items in total.\n"
          "    track_categories=[('movieId', 50, 5)],\n\n"
          "    # For each user, track a 30-day quantity history for\n"
          "    # predicting 1 day ahead. Track counts at weekly granularity.\n"
          "    track_quantities=['movie_duration'],\n"
          "    time_granularity='daily',\n"
          "    time_to_predict_ahead=1,\n"
          "    history_length_for_inference=30\n\n"
          "    # Categorical columns are delimited by spaces\n"
          "    multi_class_delim=' '\n"
          ")\n"
          "```\n")
      .def("train", &SequentialClassifier::train, py::arg("train_file"),
           py::arg("epochs"), py::arg("learning_rate"),
           py::arg("metrics") = std::vector<std::string>({"recall@1"}),
           "Trains a Sequential classifier Model using the data provided in "
           "train_file"
           "Arguments:\n"
           " * train_file: str - The path to the training dataset. The dataset "
           "has to be a CSV file with a header.\n"
           " * epochs: int - Number of epochs to train the model.\n"
           " * learning_rate: float - Learning rate\n"
           " * metrics (Optional): List[Str] - Metrics to track during "
           "training. Defaults "
           "to ['recall@1'] Metrics are currently restricted to any 'recall@k' "
           "where k is a positive (nonzero) integer.\n"
           "Example:\n"
           "```\n"
           "from thirdai import bolt\n\n"
           "model = bolt.SequentialClassifier(...)\n"
           "model.train(\n"
           "    'train_data.csv',\n"
           "    epochs=3,\n"
           "    learning_rate=0.0001,\n"
           "    metrics=['recall@1', 'recall@10', 'recall@100']\n"
           ")\n"
           "```\n")
#if THIRDAI_EXPOSE_ALL
      .def("summarizeModel", &SequentialClassifier::summarizeModel,
           "Deprecated\n")
#endif
      .def("predict", &SequentialClassifier::predict, py::arg("test_file"),
           py::arg("metrics") = std::vector<std::string>({"recall@1"}),
           py::arg("output_file") = std::nullopt,
           py::arg("write_top_k_to_file") = 1,
           "Predicts the output classes and evaluates the predictions on a "
           "test dataset. Optionally writes top k predictions to a file if "
           "output file name is provided. This method cannot be called on an "
           "untrained model.\n"
           "Arguments:\n"
           " * test_file: str - The path to the test dataset. The dataset has "
           "to be a CSV file with a header.\n"
           " * metrics (optional): List[str] - Metrics to evaluate the "
           "predictions. Defaults to ['recall@1']. Metrics are currently "
           "restricted to any 'recall@k' where k is a positive (nonzero) "
           "integer.\n"
           " * output_file (optional): str: An optional path to a file to "
           "write predictions to. If not provided, predictions will not be "
           "written to file.\n"
           " * write_top_k_to_file (optional): int: Number of top predictions "
           "to write to file per input sample. Defaults to 1.\n"
           "Example:\n"
           "```\n"
           "from thirdai import bolt\n\n"
           "model = bolt.SequentialClassifier(...)\n"
           "model.train(...)\n"
           "model.predict(\n"
           "    'test_data.csv',\n"
           "    metrics=['recall@1', 'recall@10', 'recall@100']\n"
           "    output_file='predictions.txt',\n"
           "    write_top_k_to_file=10,\n"
           ")\n"
           "```\n")
      .def(
          "predict_single", &SequentialClassifier::predictSingle,
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
          " * k: Int (positive) - The number of top results to return.\n"
          "Returns a list of (prediction, score) tuples."
          "Example:\n"
          "```\n"
          "# Suppose we construct a SequentialClassifier as follows:\n"
          "model = SequentialClassifier(\n"
          "    user=('name', 500),\n"
          "    label=('movie', 5001),\n"
          "    timestamp='timestamp',\n"
          "    static_categorical=[('age_group', 7)]\n"
          "    track_categories=[('movie', 5001, 10)]\n"
          "    track_quantities=['movie_duration']\n"
          "    time_granularity='daily',\n"
          "    time_to_predict_ahead=1,\n"
          "    history_length_for_inference=30\n\n"
          ")\n\n"
          "# Suppose there is a user 'arun' for whome we want to recommend\n"
          "# the next movie to watch, then the input sample would be as "
          "follows."
          "input_sample = {\n"
          "    'name': 'arun',\n"
          "    'timestamp': '2022-02-02',\n"
          "    'age_group': '20-39',\n"
          "    'movie': 'null_token'\n"
          "    'movie_duration': '0'\n"
          "})\n\n"
          "model.predict_single(input_sample, top_k=10)\n"
          "```\n"
          "Notice that in the example above, we will not know the movie or "
          "movie duration since they are both tied to the variable that we are "
          "trying to predict. In this scenario, we can pass in random values "
          "and the model automatically ignores them. If you pass a unique "
          "'null token' in place of a tracked categorical feature for clarity, "
          "the number of unique values for that column (passed into the "
          "constructor) must include this 'null token'. In this example, there "
          "are 5000 movies, but we define the number of unique movies as 5001 "
          "to include the 'null token'.")
      .def("save", &SequentialClassifier::save, py::arg("filename"),
           "Serializes the SequentialClassifier into a file on disk. Example:\n"
           "```\n"
           "from thirdai import bolt\n\n"
           "model = bolt.SequentialClassifier(...)\n"
           "model.save('seq_class_savefile.bolt')\n"
           "```\n")
      .def_static(
          "load", &SequentialClassifier::load, py::arg("filename"),
          "Loads a serialized SequentialClassifier from a file on disk. "
          "Example:\n"
          "```\n"
          "from thirdai import bolt\n\n"
          "model = bolt.SequentialClassifier.load('seq_class_savefile.bolt')\n"
          "```\n")
      .def(
          "explain", &SequentialClassifier::explain, py::arg("input_sample"),
          py::arg("target_label") = std::nullopt,
          "The Root Cause Analysis method which gives us relevant "
          "explanations of the input with respect to the given target label.\n"
          "Arguments:\n"
          " * input_sample: Dict[str, str] - The input sample as a dictionary "
          "where the keys are column names as specified in the schema and the "
          "values are the respective column values.\n"
          " * target_label (Optional): str - The label class with respect to "
          "which we want the explanations for the input. Returns a list of "
          "Explanation objects with the following fields: `column_number`, "
          "`column_name`, `keyword`, and `percentage_significance`.\n"
          "Example:\n"
          "```\n"
          "# Suppose we construct a SequentialClassifier as follows:\n"
          "model = SequentialClassifier(\n"
          "    user=('name', 500),\n"
          "    label=('salary', 5),\n"
          "    timestamp='timestamp',\n"
          "    static_categorical=[('age_group', 7)]\n"
          "    track_categories=[('expenditure_level', 7, 30)]\n"
          ")\n\n"
          "# Suppose there is a user identified as `arun` and we want to\n"
          "# know why his salary is in the '<=50k' group, then we may call\n"
          "# the explain(...) method as follows:\n\n"
          "input_sample = {\n"
          "    'name': 'arun',\n"
          "    'timestamp': '2022-02-02',\n"
          "    'age_group': '20-39',\n"
          "    'expenditure_level': 'high'\n"
          "})\n"
          "explanations = model.explain(input_sample, target_label='<=50k')\n\n"
          "# Let's now print the explanations\n\n"
          "for explanation in explanations:\n"
          "    print(explanation.column_num)\n"
          "    print(explanation.column_name)\n"
          "    print(explanation.percentage_significance)\n"
          "    print(explanation.keyword)\n"
          "```\n");

  createBoltGraphSubmodule(bolt_submodule);

  return bolt_submodule;
}

}  // namespace thirdai::bolt::python