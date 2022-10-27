#include "BoltPython.h"
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

void createBoltSubmodule(py::module_& bolt_submodule) {
  py::class_<TrainConfig, TrainConfigPtr>(bolt_submodule, "TrainConfig")
      .def(py::init(&TrainConfig::makeConfig), py::arg("learning_rate"),
           py::arg("epochs"))
      .def("with_metrics", &TrainConfig::withMetrics, py::arg("metrics"))
      .def("silence", &TrainConfig::silence)
#if THIRDAI_EXPOSE_ALL
      // We do not want to expose these methods to customers to hide complexity.
      .def("with_rebuild_hash_tables", &TrainConfig::withRebuildHashTables,
           py::arg("rebuild_hash_tables"))
      .def("with_reconstruct_hash_functions",
           &TrainConfig::withReconstructHashFunctions,
           py::arg("reconstruct_hash_functions"))
      // We do not want to expose this method because it will not work correctly
      // with the ModelPipeline since it won't sae the entire pipeline.
      .def("with_save_parameters", &TrainConfig::withSaveParameters,
           py::arg("save_prefix"), py::arg("save_frequency"))
#endif
      .def("with_callbacks", &TrainConfig::withCallbacks, py::arg("callbacks"))
      .def("with_validation", &TrainConfig::withValidation,
           py::arg("validation_data"), py::arg("validation_labels"),
           py::arg("predict_config"), py::arg("validation_frequency") = 0,
           py::arg("save_best_per_metric") = "",
           R"pbdoc(
Add validation options to execute validation during training. Can be used to
configure input data and labels, frequency to validate and optionally saving
best model per a specified metric.

Args:
    validation_data (dataset.BoltDataset): 
        Input dataset for validation
    validation_label (dataset.BoltDataset): 
        Ground truth labels to use during validation
    predict_config (bolt.graph.PredictConfig): 
        See PredictConfig.
    validation_frequency (int, optional): 
        Interval of updates (batches) to run validation and report
        metrics. Defaults to 0, which is no validation amidst
        training.
    save_best_per_metric (str, optional): 
        Whether to save best model based on validation. Needs
        with_save_parameters(...) configured.  Defaults to empty
        string, which implies no saving best model. Note that this requires the
        tracked metric to be configured via `with_metrics(...)`.

)pbdoc")
      .def_property_readonly(
          "num_epochs", [](TrainConfig& config) { return config.epochs(); },
          "Returns the number of epochs a model with this TrainConfig will "
          "train for.")
      .def_property_readonly(
          "learning_rate",
          [](TrainConfig& config) { return config.learningRate(); },
          "Returns the learning rate a model with this TrainConfig will train "
          "with.")
      .def(getPickleFunction<TrainConfig>())
      .def("with_log_loss_frequency", &TrainConfig::withLogLossFrequency,
           py::arg("log_loss_frequency"));

  py::class_<PredictConfig>(bolt_submodule, "PredictConfig")
      .def(py::init(&PredictConfig::makeConfig))
      .def("enable_sparse_inference", &PredictConfig::enableSparseInference)
      .def("with_metrics", &PredictConfig::withMetrics, py::arg("metrics"))
      .def("silence", &PredictConfig::silence)
      .def("return_activations", &PredictConfig::returnActivations);

  auto oracle_types_submodule = bolt_submodule.def_submodule("types");

  py::class_<sequential_classifier::DataType>(  // NOLINT
      oracle_types_submodule, "ColumnType", "Base class for bolt types.");

  oracle_types_submodule.def(
      "categorical", sequential_classifier::DataType::categorical,
      py::arg("n_unique_classes"), py::arg("delimiter") = std::nullopt,
      py::arg("consecutive_integer_ids") = false,
      R"pbdoc(
    Categorical column type. Use this object if a column contains categorical 
    data (each unique value is treated as a class). Examples include user IDs, 
    movie titles, or age groups.

    Args:
        n_unique_classes (int): Number of unique categories in the column.
            Oracle throws an error if the column contains more than the 
            specified number of unique values.
        delimiter (str): Optional. Defaults to None. A single character 
            (length-1 string) that separates multiple values in the same 
            column. If not provided, Oracle assumes that there is only
            one value in the column.
        consecutive_integer_ids (bool): Optional. Defaults to None. When set to
            True, the values of this column are assumed to be integers ranging 
            from 0 to n_unique_classes - 1. Otherwise, the values are assumed to 
            be arbitrary strings (including strings of integral ids that are 
            not within [0, n_unique_classes - 1]).
    
    Example:
        >>> deployment.UniversalDeepTransformer(
                data_types: {
                    "user_id": bolt.types.categorical(n_unique_classes=5000)
                }
                ...
            )
                             )pbdoc");
  oracle_types_submodule.def("numerical",
                             sequential_classifier::DataType::numerical,
                             R"pbdoc(
    Numerical column type. Use this object if a column contains numerical 
    data (the value is treated as a quantity). Examples include hours of 
    a movie watched, sale quantity, or population size.

    Example:
        >>> deployment.UniversalDeepTransformer(
                data_types: {
                    "hours_watched": bolt.types.numerical()
                }
                ...
            )
                             )pbdoc");
  oracle_types_submodule.def("text", sequential_classifier::DataType::text,
                             py::arg("average_n_words") = std::nullopt,
                             py::arg("embedding_size") = "m",
                             py::arg("use_attention") = false,
                             R"pbdoc(
    Text column type. Use this object if a column contains text data 
    (the meaning of the text matters). Examples include descriptions, 
    search queries, and user bios.

    Args:
        average_n_words (int): Optional. Average number of words in the 
            text column in each row. If provided, Oracle may make 
            optimizations as appropriate.
        embedding_size (str): Optional. One of "small"/"s", "medium"/"m",
            or "large"/"l". Defaults to "m".
        use_attention (bool): Optional. If true, oracle is guaranteed to
            use attention when processing this text column. Otherwise, 
            oracle will only use attention when appropriate.
    
    Example:
        >>> deployment.UniversalDeepTransformer(
                data_types: {
                    "user_motto": bolt.types.text(average_n_words=10),
                    "user_bio": bolt.types.text()
                }
                ...
            )

                             )pbdoc");
  oracle_types_submodule.def("date", sequential_classifier::DataType::date,
                             R"pbdoc(
    Date column type. Use this object if a column contains date strings. 
    Date strings must be in YYYY-MM-DD format.
 
    Example:
        >>> deployment.UniversalDeepTransformer(
                data_types: {
                    "timestamp": bolt.types.date()
                }
                ...
            )
                             )pbdoc");

  auto oracle_temporal_submodule = bolt_submodule.def_submodule("temporal");

  py::class_<sequential_classifier::TemporalConfig>(  // NOLINT
      oracle_temporal_submodule, "TemporalConfig",
      "Base class for temporal feature configs.");

  oracle_temporal_submodule.def(
      "categorical", sequential_classifier::TemporalConfig::categorical,
      py::arg("column_name"), py::arg("track_last_n"),
      py::arg("column_known_during_inference") = false,
      R"pbdoc(
    Temporal categorical config. Use this object to configure how a 
    categorical column is tracked over time. 

    Args:
        column_name (str): The name of the tracked column.
        track_last_n (int): Number of last categorical values to track
            per tracking id.
        column_known_during_inference (bool): Optional. Whether the 
            value of the tracked column is known during inference. Defaults 
            to False.

    Example:
        >>> # Suppose each row of our data has the following columns: "product_id", "timestamp", "ad_spend_level", "sales_performance"
        >>> # We want to predict the current week's sales performance for each product using temporal context.
        >>> # For each product ID, we would like to track both their ad spend level and sales performance over time.
        >>> # Ad spend level is known at the time of inference but sales performance is not. Then we can configure Oracle as follows:
        >>> model = deployment.UniversalDeepTransformer(
                data_types={
                    "product_id": bolt.types.categorical(n_unique_classes=5000),
                    "timestamp": bolt.types.date(),
                    "ad_spend_level": bolt.types.categorical(n_unique_classes=5),
                    "sales_performance": bolt.types.categorical(n_unique_classes=5),
                },
                temporal_tracking_relationships={
                    "product_id": [
                        bolt.temporal.categorical(column_name="ad_spend_level", track_last_n=5, column_known_during_inference=True),
                        bolt.temporal.categorical(column_name="ad_spend_level", track_last_n=25, column_known_during_inference=True),
                        bolt.temporal.categorical(column_name="sales_performance", track_last_n=5), # column_known_during_inference defaults to False
                    ]
                },
                ...
            )
    
    Notes:
        - Temporal categorical features are tracked as a set; if we track the last 5 ad spend levels,
          we capture what the last 5 ad spend levels are, but we do not capture their order.
        - The same column can be tracked more than once, allowing us to capture both short and
          long term trends.
      )pbdoc");
  oracle_temporal_submodule.def(
      "numerical", sequential_classifier::TemporalConfig::numerical,
      py::arg("column_name"), py::arg("history_length"),
      py::arg("column_known_during_inference") = false,
      R"pbdoc(
    Temporal numerical config. Use this object to configure how a 
    numerical column is tracked over time. 

    Args:
        column_name (str): The name of the tracked column.
        history_length (int): Amount of time to look back. Time is in terms 
            of the time granularity passed to the Oracle constructor.
        column_known_during_inference (bool): Optional. Whether the 
            value of the tracked column is known during inference. Defaults 
            to False.

    Example:
        >>> # Suppose each row of our data has the following columns: "product_id", "timestamp", "ad_spend", "sales_performance"
        >>> # We want to predict the current week's sales performance for each product using temporal context.
        >>> # For each product ID, we would like to track both their ad spend and sales performance over time.
        >>> # Ad spend is known at the time of inference but sales performance is not. Then we can configure Oracle as follows:
        >>> model = deployment.UniversalDeepTransformer(
                data_types={
                    "product_id": bolt.types.categorical(n_unique_classes=5000),
                    "timestamp": bolt.types.date(),
                    "ad_spend": bolt.types.numerical(),
                    "sales_performance": bolt.types.categorical(n_unique_classes=5),
                },
                target="sales_performance"
                time_granularity="weekly",
                temporal_tracking_relationships={
                    "product_id": [
                        # Track last 5 weeks of ad spend
                        bolt.temporal.numerical(column_name="ad_spend", history_length=5, column_known_during_inference=True),
                        # Track last 10 weeks of ad spend
                        bolt.temporal.numerical(column_name="ad_spend", history_length=10, column_known_during_inference=True),
                        # Track last 5 weeks of sales quantity
                        bolt.temporal.numerical(column_name="sales_quantity", history_length=5), # column_known_during_inference defaults to False
                    ]
                },

            )
    
    Notes:
        - The same column can be tracked more than once, allowing us to capture both short and
          long term trends.
      )pbdoc");

  /**
   * Sequential Classifier
   */
  py::class_<SequentialClassifier>(bolt_submodule, "Oracle",
                                   R"pbdoc( 
    An all-purpose classifier for tabular datasets. In addition to learning from
    the columns of a single row, Oracle can make use of "temporal context". For 
    example, if used to build a movie recommender, Oracle may use information 
    about the last 5 movies that a user has watched to recommend the next movie.
    Similarly, if used to forecast the outcome of marketing campaigns, Oracle may 
    use several months' worth of campaign history for each product to make better
    forecasts.
    
                                   )pbdoc")
      .def(py::init<std::map<std::string, sequential_classifier::DataType>,
                    std::map<std::string,
                             std::vector<std::variant<
                                 std::string,
                                 sequential_classifier::TemporalConfig>>>,
                    std::string, std::string, uint32_t>(),
           py::arg("data_types"),
           py::arg("temporal_tracking_relationships") = std::map<
               std::string,
               std::vector<std::variant<
                   std::string, sequential_classifier::TemporalConfig>>>(),
           py::arg("target"), py::arg("time_granularity") = "daily",
           py::arg("lookahead") = 0,
           R"pbdoc(  
    Constructor.

    Args:
        data_types (Dict[str, bolt.types.ColumnType]): A mapping from column name to column type. 
            This map specifies the columns that we want to pass into the model; it does 
            not need to include all columns in the dataset.

            Column type is one of:
            - `bolt.types.categorical(n_unique_values: int)`
            - `bolt.types.numerical()`
            - `bolt.types.text(average_n_words: int=None)`
            - `bolt.types.date()`
            See bolt.types for details.

            If `temporal_tracking_relationships` is non-empty, there must one 
            bolt.types.date() column. This column contains date strings in YYYY-MM-DD format.
            There can only be one bolt.types.date() column.
        temporal_tracking_relationships (Dict[str, List[str or bolt.temporal.TemporalConfig]]): Optional. 
            A mapping from column name to a list of either other column names or bolt.temporal objects.
            This mapping tells Oracle what columns can be tracked over time for each key.
            For example, we may want to tell Oracle that we want to track a user's watch 
            history by passing in a map like `{"user_id": ["movie_id"]}`

            If we provide a mapping from a string to a list of strings like the above, 
            the temporal tracking configuration will be autotuned. We can take control by 
            passing in bolt.temporal objects intead of strings.

            bolt.temporal object is one of:
            - `bolt.temporal.categorical(column_name: str, track_last_n: int, column_known_during_inference: bool=False)
            - `bolt.temporal.numerical(column_name: str, history_length: int, column_known_during_inference: bool=False)
            See bolt.temporal for details.
        target (str): Name of the column that contains the value to be predicted by
            Oracle. The target column has to be a categorical column.
        time_granularity (str): Optional. Either `"daily"`/`"d"`, `"weekly"`/`"w"`, `"biweekly"`/`"b"`, 
            or `"monthly"`/`"m"`. Interval of time that we are interested in. Temporal numerical 
            features are clubbed according to this time granularity. E.g. if 
            `time_granularity="w"` and the numerical values on days 1 and 2 are
            345.25 and 201.1 respectively, then Oracle captures a single numerical 
            value of 546.26 for the week instead of individual values for the two days.
            Defaults to "daily".
        lookahead (str): Optional. How far into the future the model needs to predict. This length of
            time is in terms of time_granularity. E.g. 'time_granularity="daily"` and 
            `lookahead=5` means the model needs to learn to predict 5 days ahead. Defaults to 0
            (predict the immediate next thing).

    Examples:
        >>> # Suppose each row of our data has the following columns: "product_id", "timestamp", "ad_spend", "sales_quantity", "sales_performance"
        >>> # We want to predict next week's sales performance for each product using temporal context.
        >>> # For each product ID, we would like to track both their ad spend and sales quantity over time.
        >>> model = bolt.Oracle(
                data_types={
                    "product_id": bolt.types.categorical(n_unique_classes=5000),
                    "timestamp": bolt.types.date(),
                    "ad_spend": bolt.types.numerical(),
                    "sales_quantity": bolt.types.numerical(),
                    "sales_performance": bolt.types.categorical(n_unique_classes=5),
                },
                temporal_tracking_relationships={
                    "product_id": [
                        # Track last 5 weeks of ad spend
                        bolt.temporal.numerical(column_name="ad_spend", history_length=5),
                        # Track last 10 weeks of ad spend
                        bolt.temporal.numerical(column_name="ad_spend", history_length=10),
                        # Track last 5 weeks of sales performance
                        bolt.temporal.categorical(column_name="sales_performance", history_length=5),
                    ]
                },
                target="sales_performance"
                time_granularity="weekly",
                lookahead=2 # predict 2 weeks ahead
            )
        >>> # Alternatively suppose our data has the following columns: "user_id", "movie_id", "hours_watched", "timestamp"
        >>> # We want to build a movie recommendation system.
        >>> # Then we may configure Oracle as follows:
        >>> model = bolt.Oracle(
                data_types={
                    "user_id": bolt.types.categorical(n_unique_classes=5000),
                    "timestamp": bolt.types.date(),
                    "movie_id": bolt.types.categorical(n_unique_classes=3000),
                    "hours_watched": bolt.types.numerical(),
                },
                temporal_tracking_relationships={
                    "user_id": [
                        "movie_id", # autotuned movie temporal tracking
                        bolt.temporal.numerical(column_name="hours_watched", history_length="5") # track last 5 days of hours watched.
                    ]
                },
                target="movie_id"
            )

    Notes:
        - Refer to the documentation bolt.types.ColumnType and bolt.temporal.TemporalConfig to better understand column types 
          and temporal tracking configurations.

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
        - If temporal tracking relationships are provided, Oracle can make better predictions 
          by taking temporal context into account. For example, Oracle may keep track of 
          the last few movies that a user has watched to better recommend the next movie.
          `model.train()` automatically updates Oracle's temporal context.
        - `model.train()` resets Oracle's temporal context at the start of training to 
          prevent unwanted information from leaking into the training routine.
           )pbdoc")
#if THIRDAI_EXPOSE_ALL
      .def("summarizeModel", &SequentialClassifier::summarizeModel,
           "Deprecated\n")
#endif
      .def("evaluate", &SequentialClassifier::predict,
           py::arg("validation_file"),
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
        Dict[str, float]:
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
        - If temporal tracking relationships are provided, Oracle can make better predictions 
          by taking temporal context into account. For example, Oracle may keep track of 
          the last few movies that a user has watched to better recommend the next movie.
          `model.evaluate()` automatically updates Oracle's temporal context.
           )pbdoc")

      .def("predict", &SequentialClassifier::predictSingle,
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
        - Only columns that are known at the time of inference need to be passed to
          `model.predict()`. For example, notice that while we have a "movie_title" 
          column in the `data_types` argument, we did not pass it to `model.predict()`. 
          This is because we do not know the movie title at the time of inference – that 
          is the target that we are trying to predict after all.

        - If temporal tracking relationships are provided, Oracle can make better predictions 
          by taking temporal context into account. For example, Oracle may keep track of 
          the last few movies that a user has watched to better recommend the next movie. 
          Thus, Oracle is at its best when its internal temporal context gets updated with
          new true samples. `model.predict()` does not update Oracle's temporal context.
          To do this, we need to use `model.index()`. Read about `model.index()` for details.
           )pbdoc")
      .def("explain", &SequentialClassifier::explain, py::arg("input_sample"),
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
        absolute value of the `percentage_significance` field of each element.
    
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
                input_sample={"user_id": "A33225", "timestamp": "2022-02-02", "special_event": "christmas"}, target="Home Alone"
            )
        >>> print(explanations[0].column_name)
        "special_event"
        >>> print(explanations[0].percentage_significance)
        25.2
        >>> print(explanations[0].keyword)
        "christmas"
        >>> print(explanations[1].column_name)
        "movie_id"
        >>> print(explanations[1].percentage_significance)
        -22.3
        >>> print(explanations[1].keyword)
        "Previously seen 'Die Hard'"
    
    Notes: 
        - The `column_name` field of the `Explanation` object is irrelevant in this case
          since `model.explain()` uses column names.
        - `percentage_significance` can be positive or negative depending on the 
          relationship between the responsible column and the prediction. In the above
          example, the `percentage_significance` associated with the explanation
          "Previously seen 'Die Hard'" is negative because recently watching "Die Hard" is 
          negatively correlated with the target class "Home Alone".
        - Only columns that are known at the time of inference need to be passed to
          `model.explain()`. For example, notice that while we have a "movie_title" 
          column in the `data_types` argument, we did not pass it to `model.explain()`. 
          This is because we do not know the movie title at the time of inference – that 
          is the target that we are trying to predict after all.
        - If temporal tracking relationships are provided, Oracle can make better predictions 
          by taking temporal context into account. For example, Oracle may keep track of 
          the last few movies that a user has watched to better recommend the next movie. 
          Thus, Oracle is at its best when its internal temporal context gets updated with
          new true samples. `model.explain()` does not update Oracle's temporal context.
          To do this, we need to use `model.index()`. Read about `model.index()` for details.
           )pbdoc")
      .def("index", &SequentialClassifier::indexSingle, py::arg("sample"),
           R"pbdoc(

    Indexes a single true sample to keep Oracle's temporal context up to date.

    If temporal tracking relationships are provided, Oracle can make better predictions 
    by taking temporal context into account. For example, Oracle may keep track of 
    the last few movies that a user has watched to better recommend the next movie. 
    Thus, Oracle is at its best when its internal temporal context gets updated with
    new true samples. `model.index()` does exactly this. 

    Args: 
        input_sample (Dict[str, str]): The input sample as a dictionary 
            where the keys are column names as specified in data_types and the "
            values are the respective column values. 

    Example:
        >>> # Suppose we configure and train Oracle to do movie recommendation as follows:
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
        >>> # We then deploy the model for inference. Inference is performed by calling model.predict()
        >>> predictions = model.predict(
                input_sample={"user_id": "A33225", "timestamp": "2022-02-02", "special_event": "christmas"}, top_k=3
            )
        >>> # Suppose we later learn that user "A33225" ends up watching "Die Hard 3". 
        >>> # We can call model.index() to keep Oracle's temporal context up to date.
        >>> model.index(
                input_sample={"user_id": "A33225", "timestamp": "2022-02-02", "special_event": "christmas", "movie_title": "Die Hard 3"}
            )
          )pbdoc")
      .def("save", &SequentialClassifier::save, py::arg("filename"),
           R"pbdoc(  
    Serializes an instance of Oracle into a file on disk. The serialized Oracle includes 
    its current temporal context.
    
    Args:
        filename (str): The file on disk to serialize this instance of Oracle into.

    Example:
        >>> model.Oracle(...)
        >>> model.save("oracle_savefile.bolt")
           )pbdoc")
      .def_static("load", &SequentialClassifier::load, py::arg("filename"),
                  R"pbdoc(  
    Loads a serialized instance of Oracle from a file on disk. The loaded Oracle includes 
    the temporal context from before serialization.
    
    Args:
        filename (str): The file on disk to load the instance of Oracle from.

    Example:
        >>> model.Oracle(...)
        >>> model = bolt.Oracle.load("oracle_savefile.bolt")
           )pbdoc");
}

}  // namespace thirdai::bolt::python