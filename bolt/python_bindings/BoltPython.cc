#include "BoltPython.h"
#include "BoltGraphPython.h"
#include <bolt/python_bindings/ConversionUtils.h>
#include <bolt/src/auto_classifiers/sequential_classifier/ConstructorUtilityTypes.h>
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
      ":math:`MSE = sum( (actual - prediction)^2 )`")
      .def(py::init<>(), "Constructs a MeanSquaredError object.");

  py::class_<WeightedMeanAbsolutePercentageErrorLoss,
             std::shared_ptr<WeightedMeanAbsolutePercentageErrorLoss>,
             LossFunction>(
      bolt_submodule, "WeightedMeanAbsolutePercentageError",
      "A loss function to minimize weighted mean absolute percentage error "
      "(WMAPE) "
      "for regression tasks. :math:`WMAPE = 100% * sum(|actual - prediction|) "
      "/ sum(|actual|)`")
      .def(py::init<>(),
           "Constructs a WeightedMeanAbsolutePercentageError object.");

  auto oracle_types_submodule = bolt_submodule.def_submodule("types");

  py::class_<sequential_classifier::DataType>(  // NOLINT
      oracle_types_submodule, "ColumnType", "Base class for bolt types.");

  oracle_types_submodule.def("categorical",
                             sequential_classifier::DataType::categorical,
                             py::arg("n_unique_classes"));
  oracle_types_submodule.def("numerical",
                             sequential_classifier::DataType::numerical);
  oracle_types_submodule.def("text", sequential_classifier::DataType::text,
                             py::arg("average_n_words") = std::nullopt);
  oracle_types_submodule.def("date", sequential_classifier::DataType::date);

  auto oracle_temporal_submodule = bolt_submodule.def_submodule("temporal");

  py::class_<sequential_classifier::TemporalConfig>(  // NOLINT
      oracle_temporal_submodule, "TemporalConfig",
      "Base class for temporal feature configs.");

  oracle_temporal_submodule.def(
      "categorical", sequential_classifier::TemporalConfig::categorical,
      py::arg("column_name"), py::arg("track_last_n"),
      py::arg("include_current_row") = false);
  oracle_temporal_submodule.def(
      "numerical", sequential_classifier::TemporalConfig::numerical,
      py::arg("column_name"), py::arg("history_length"),
      py::arg("include_current_row") = false);

  /**
   * Sequential Classifier
   */
  py::class_<SequentialClassifier>(bolt_submodule, "Oracle",
                                   "Autoclassifier for sequential predictions.")
      .def(
          py::init<std::map<std::string, sequential_classifier::DataType>,
                   std::map<std::string,
                            std::vector<std::variant<std::string, sequential_classifier::TemporalConfig>>>,
                   std::string, std::string, uint32_t>(),
          py::arg("data_types"), py::arg("temporal_tracking_relationships"),
          py::arg("target"), py::arg("time_granularity") = "daily",
          py::arg("lookahead") = 0,
          R"pbdoc(  
    Trains the network on the given training data and labels with the given training
    config.

    Args:
        train_data (List[BoltDataset] or BoltDataset): The data to train the model with. 
            There should be exactly one BoltDataset for each Input node in the Bolt
            model, and each BoltDataset should have the same total number of 
            vectors and the same batch size. The batch size for training is the 
            batch size of the passed in BoltDatasets (you can specify this batch 
            size when loading or creating a BoltDataset).
        train_labels (BoltDataset): The labels to use as ground truth during 
            training. There should be the same number of total vectors and the
            same batch size in this BoltDataset as in the train_data list.
        train_config (TrainConfig): The object describing all other training
            configuration details. See the TrainConfig documentation for more
            information as to possible options. This includes the number of epochs
            to train for, the verbosity of the training, the learning rate, and so
            much more!

    Returns:
        Dict[Str, List[float]]:
        A dictionary from metric name to a list of the value of that metric 
        for each epoch (this also always includes an entry for 'epoch_times'). The 
        metrics that are returned are the metrics requested in the TrainConfig.

    Notes:
        Sparse bolt training was originally based off of SLIDE. See [1]_ for more details

    References:
        .. [1] "SLIDE : In Defense of Smart Algorithms over Hardware Acceleration for Large-Scale Deep Learning Systems" 
                https://arxiv.org/pdf/1903.03129.pdf.

    Examples:
        >>> train_config = (
                bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=3)
                .with_metrics(["categorical_accuracy"])
            )
        >>> metrics = model.train(
                train_data=train_data, train_labels=train_labels, train_config=train_config
            )
        >>> print(metrics)
        {'epoch_times': [1.7, 3.4, 5.2], 'categorical_accuracy': [0.4665, 0.887, 0.9685]}

    That's all for now, folks! More docs coming soon :)

    )pbdoc")
      .def("train", &SequentialClassifier::train, py::arg("train_file"),
           py::arg("epochs"), py::arg("learning_rate"),
           py::arg("metrics") = std::vector<std::string>({"recall@1"}),
           R"pbdoc(  
    Trains the model using the data provided in train_file.

    Args:
        train_file (str): The path to the training dataset. The dataset
            has to be a CSV file with a header.
        epochs (int): Number of epochs to train the model.
        learning_rate (float): Learning rate. We recommend a learning 
            rate of 0.0001 or lower.
        metrics (List[str]): Metrics to track during training. Defaults to 
            ["recall@1"]. Metrics are currently restricted to any 'recall@k' 
            where k is a positive (nonzero) integer.

    Returns:
        Dict[Str, List[float]]:
        A dictionary from metric name to a list of the value of that metric 
        for each epoch (this also always includes an entry for 'epoch_times'
        measured in seconds). The metrics that are returned are the metrics 
        passed to the `metrics` parameter.
    
    Example:
        >>> metrics = model.train(
                train_file="train_file.csv", epochs=3, learning_rate=0.0001, metrics=["recall@1", "recall@10"]
            )
        >>> print(metrics)
        {'epoch_times': [1.7, 3.4, 5.2], 'recall@1': [0.0922, 0.187, 0.268], 'recall@10': [0.4665, 0.887, 0.9685]}
    
    Notes:
        * Temporal tracking relationships helps Oracle make better predictions by 
        taking temporal context into account. For example, Oracle may keep track of 
        the last few movies that a user has watched to better recommend the next movie.
        `model.train()` automatically updates Oracle's temporal context.
           )pbdoc"
        )
#if THIRDAI_EXPOSE_ALL
      .def("summarizeModel", &SequentialClassifier::summarizeModel,
           "Deprecated\n")
#endif
      .def("evaluate", &SequentialClassifier::predict, py::arg("validation_file"),
           py::arg("metrics") = std::vector<std::string>({"recall@1"}),
           py::arg("output_file") = std::nullopt,
           py::arg("write_top_k_to_file") = 1,
           R"pbdoc(  
    Evaluates how well the model predicts output classes on a validation 
    dataset. Optionally writes top k predictions to a file if output file 
    name is provided for external evaluation. This method cannot be called 
    on an untrained model.

    Args:
        validation_file (str): The path to the validation dataset to 
            evaluate on. The dataset has to be a CSV file with a header.
        metrics (List[str]): Metrics to track during training. Defaults to 
            ["recall@1"]. Metrics are currently restricted to any 'recall@k' 
            where k is a positive (nonzero) integer.
        output_file (str): An optional path to a file to write predictions 
            to. If not provided, predictions will not be written to file.
        write_top_k_to_file (int): Only relevant if `output_file` is provided. 
            Number of top predictions to write to file per input sample. 
            Defaults to 1.

    Returns:
        Dict[Str, float]:
        A dictionary from metric name to the value of that metric (this also 
        always includes an entry for 'test_time' measured in milliseconds). 
        The metrics that are returned are the metrics passed to the `metrics` 
        parameter.
    
    Example:
        >>> metrics = model.evaluate(
                validation_file="validation_file.csv", metrics=["recall@1", "recall@10"], output_file="predictions.txt", write_top_k_to_file=10
            )
        >>> print(metrics)
        {'test_time': 20.0, 'recall@1': [0.0922, 0.187, 0.268], 'recall@10': [0.4665, 0.887, 0.9685]}
    
    Notes: 
        * Temporal tracking relationships helps Oracle make better predictions by 
        taking temporal context into account. For example, Oracle may keep track of 
        the last few movies that a user has watched to better recommend the next movie.
        `model.evaluate()` automatically updates Oracle's temporal context.
           )pbdoc"
        )
    
      .def(
          "predict", &SequentialClassifier::predictSingle,
          py::arg("input_sample"), py::arg("top_k") = 1,
          R"pbdoc(  
    Computes the top k classes and their probabilities for a single input sample. 

    Args:
        input_sample (Dict[str, str]): The input sample as a dictionary 
            where the keys are column names as specified in data_types and the "
            values are the respective column values. 
        top_k (int): The number of top results to return. Must be > 1.

    Returns:
        List[Tuple[str, float]]:
        A sorted list of pairs containing the predicted class name and the score for 
        that prediction. The pairs are sorted in descending order from highest
        score to lowest score.
    
    Example:
        >>> # Suppose we configure and train Oracle as follows:
        >>> model = bolt.Oracle(
                data_types={
                    "user_id": bolt.types.categorical(n_unique_classes=5000),
                    "timestamp": bolt.types.date(),
                    "special_event": bolt.types.categorical(n_unique_classes=20),
                    "movie_title": bolt.types.categorical(n_unique_classes=500)
                },
                temporal_tracking_relationships={
                    "user_id": ["movie_title"]
                },
                target="movie_title"
            )
        >>> model.train(
                train_file="train_file.csv", epochs=3, learning_rate=0.0001, metrics=["recall@1", "recall@10"]
            )
        >>> # Make a single prediction
        >>> predictions = model.predict(
                input_sample={"user_id": "A33225", "timestamp": "2022-02-02", "special_event": "christmas"}, top_k=3
            )
        >>> print(predictions)
        [("Gone With The Wind", 0.322), ("Titanic", 0.225), ("Pretty Woman", 0.213)]
    
    Notes: 
        * Only columns that are known at the time of inference need to be passed to
        `model.predict()`. For example, notice that while we have a "movie_title" 
        column in the `data_types` argument, we did not pass it to `model.predict()`. 
        This is because we do not know the movie title at the time of inference – that 
        is the target that we are trying to predict after all.

        * Temporal tracking relationships helps Oracle make better predictions by 
        taking temporal context into account. For example, Oracle may keep track of 
        the last few movies that a user has watched to better recommend the next movie. 
        Thus, Oracle is at its best when its internal temporal context gets updated with
        new true samples. `model.predict()` does not update Oracle's temporal context.
        To do this, we need to use `model.index()`. Read about `model.index()` for details.
           )pbdoc"
        )
      .def(
          "explain", &SequentialClassifier::explain, py::arg("input_sample"),
          py::arg("target") = std::nullopt,
          R"pbdoc(  
    Identifies the columns that are most responsible for a predicted outcome 
    and provides a brief description of the column's value.
    
    If a target is provided, the model will identify the columns that need 
    to change for the model to predict the target class.
    
    Args:
        input_sample (Dict[str, str]): The input sample as a dictionary 
            where the keys are column names as specified in data_types and the "
            values are the respective column values. 
        target (str): Optional. The desired target class. If provided, the
        model will identify the columns that need to change for the model to 
        predict the target class.

    Returns:
        List[Explanation]:
        A sorted list of `Explanation` objects that each contain the following fields:
        `column_number`, `column_name`, `keyword`, and `percentage_significance`.
        `column_number` and `column_name` identify the responsible column, 
        `keyword` is a brief description of the value in this column, and
        `percentage_significance` represents this column's contribution to the
        predicted outcome. The list is sorted in descending order by the 
        `percentage_significance` field of each element.
    
    Example:
        >>> # Suppose we configure and train Oracle as follows:
        >>> model = bolt.Oracle(
                data_types={
                    "user_id": bolt.types.categorical(n_unique_classes=5000),
                    "timestamp": bolt.types.date(),
                    "special_event": bolt.types.categorical(n_unique_classes=20),
                    "movie_title": bolt.types.categorical(n_unique_classes=500)
                },
                temporal_tracking_relationships={
                    "user_id": "movie_title"
                },
                target="movie_title"
            )
        >>> model.train(
                train_file="train_file.csv", epochs=3, learning_rate=0.0001, metrics=["recall@1", "recall@10"]
            )
        >>> # Make a single prediction
        >>> explanations = model.explain(
                input_sample={"user_id": "A33225", "timestamp": "2022-02-02", "special_event": "christmas"}, target="Home Alone 2"
            )
        >>> print(explanations[0].column_name)
        "special_event"
        >>> print(explanations[0].percentage_significance)
        0.25
        >>> print(explanations[0].keyword)
        "christmas"
        >>> print(explanations[1].column_name)
        "movie_id"
        >>> print(explanations[1].percentage_significance)
        0.22
        >>> print(explanations[1].keyword)
        "Previously seen 'Home Alone 1'"
    
    Notes: 
        * The `column_name` field of the `Explanation` object is irrelevant in this case
        since `model.explain()` uses column names.

        * Only columns that are known at the time of inference need to be passed to
        `model.explain()`. For example, notice that while we have a "movie_title" 
        column in the `data_types` argument, we did not pass it to `model.explain()`. 
        This is because we do not know the movie title at the time of inference – that 
        is the target that we are trying to predict after all.

        * Temporal tracking relationships helps Oracle make better predictions by 
        taking temporal context into account. For example, Oracle may keep track of 
        the last few movies that a user has watched to better recommend the next movie. 
        Thus, Oracle is at its best when its internal temporal context gets updated with
        new true samples. `model.explain()` does not update Oracle's temporal context.
        To do this, we need to use `model.index()`. Read about `model.index()` for details.
           )pbdoc"
        )
      .def(
          "index_single", &SequentialClassifier::indexSingle, py::arg("sample"),
          "Indexes a single true sample to keep the SequentialClassifier's "
          "internal quantity and category trackers up to date.\n"
          "Arguments:\n"
          " * input_sample: Dict[str, str] - The input sample as a dictionary "
          "where the keys are column names as specified in the schema and the "
          "values are the respective column values.\n"
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
          "# Suppose we recorded a new sample with the following information:\n"
          "input_sample = {\n"
          "    'name': 'arun',\n"
          "    'salary': '<=50k',\n"
          "    'timestamp': '2022-02-02',\n"
          "    'age_group': '20-39',\n"
          "    'expenditure_level': 'high'\n"
          "})\n\n"
          "# Then we will call index_single as follows:\n"
          "model.index_single(input_sample)\n"
          "```\n")
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
          "```\n");

  createBoltGraphSubmodule(bolt_submodule);

  return bolt_submodule;
}

}  // namespace thirdai::bolt::python