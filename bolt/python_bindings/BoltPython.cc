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
//   py::class_<LossFunction, std::shared_ptr<LossFunction>>(  // NOLINT
//       bolt_submodule, "LossFunction", "Base class for all loss functions");

//   py::class_<CategoricalCrossEntropyLoss,
//              std::shared_ptr<CategoricalCrossEntropyLoss>, LossFunction>(
//       bolt_submodule, "CategoricalCrossEntropyLoss",
//       "A loss function for multi-class (one label per sample) classification "
//       "tasks.")
//       .def(py::init<>(), "Constructs a CategoricalCrossEntropyLoss object.");

//   py::class_<BinaryCrossEntropyLoss, std::shared_ptr<BinaryCrossEntropyLoss>,
//              LossFunction>(
//       bolt_submodule, "BinaryCrossEntropyLoss",
//       "A loss function for multi-label (multiple class labels per each sample) "
//       "classification tasks.")
//       .def(py::init<>(), "Constructs a BinaryCrossEntropyLoss object.");

//   py::class_<MeanSquaredError, std::shared_ptr<MeanSquaredError>, LossFunction>(
//       bolt_submodule, "MeanSquaredError",
//       "A loss function that minimizes mean squared error (MSE) for regression "
//       "tasks. "
//       ":math:`MSE = sum( (actual - prediction)^2 )`")
//       .def(py::init<>(), "Constructs a MeanSquaredError object.");

//   py::class_<WeightedMeanAbsolutePercentageErrorLoss,
//              std::shared_ptr<WeightedMeanAbsolutePercentageErrorLoss>,
//              LossFunction>(
//       bolt_submodule, "WeightedMeanAbsolutePercentageError",
//       "A loss function to minimize weighted mean absolute percentage error "
//       "(WMAPE) "
//       "for regression tasks. :math:`WMAPE = 100% * sum(|actual - prediction|) "
//       "/ sum(|actual|)`")
//       .def(py::init<>(),
//            "Constructs a WeightedMeanAbsolutePercentageError object.");

  auto universal_transformer_types_submodule = bolt_submodule.def_submodule("types");

  py::class_<sequential_classifier::DataType>(  // NOLINT
      universal_transformer_types_submodule, "ColumnType", "Base class for bolt types.");

  universal_transformer_types_submodule.def(
      "categorical", sequential_classifier::DataType::categorical,
      py::arg("n_unique_classes"),
      py::arg("seperator"),
                             R"pbdoc(
    Categorical column type. Use this object if a column contains categorical 
    data (each unique value is treated as a class). Examples include user IDs, 
    movie titles, or age groups.

    Args:
        n_unique_classes (int): Number of unique categories in the column.
            UniversalTransformer throws an error if the column contains more than the 
            specified number of unique values.
        seperator (str): `Optional.` seperator if multiple categories need to be mapped 
    Example:
        >>> bolt.UniversalTransformer(
                data_types: {
                    "user_id": bolt.types.categorical(n_unique_classes=5000)
                }
                ...
            )
                             )pbdoc");
  universal_transformer_types_submodule.def("numerical",
                             sequential_classifier::DataType::numerical,
                             R"pbdoc(
    Numerical column type. Use this object if a column contains numerical 
    data (the value is treated as a quantity). Examples include hours of 
    a movie watched, sale quantity, or population size.

    Example:
        >>> bolt.UniversalTransformer(
                data_types: {
                    "hours_watched": bolt.types.numerical()
                }
                ...
            )
                             )pbdoc");
  universal_transformer_types_submodule.def(
      "text", sequential_classifier::DataType::text,
                             py::arg("input_embedding_size"),
                             py::arg("average_n_words") = std::nullopt,
                             R"pbdoc(
    Text column type. Use this object if a column contains text data 
    (the meaning of the text matters). Examples include descriptions, 
    search queries, and user bios.

    Args:
        input_embedding_size (str): Optional. select from choice of small, medium and large depending on number of data samples.
        average_n_words (int): Optional. Average number of words in the 
            text column in each row. If provided, UniversalTransformer may make 
            optimizations as appropriate.
    
    Example:
        >>> bolt.UniversalTransformer(
                data_types: {
                    "user_motto": bolt.types.text(average_n_words=10),
                    "user_bio": bolt.types.text()
                }
                ...
            )

                             )pbdoc");
  universal_transformer_types_submodule.def("date", sequential_classifier::DataType::date,
                             R"pbdoc(
    Date column type. Use this object if a column contains date strings. 
    Date strings must be in YYYY-MM-DD format.
 
    Example:
        >>> bolt.UniversalTransformer(
                data_types: {
                    "timestamp": bolt.types.date()
                }
                ...
            )
                             )pbdoc");

  auto universal_transformer_temporal_submodule = bolt_submodule.def_submodule("temporal");

  py::class_<sequential_classifier::TemporalConfig>(  // NOLINT
      universal_transformer_temporal_submodule, "TemporalConfig",
      "Base class for temporal feature configs.");

  universal_transformer_temporal_submodule.def(
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
        >>> # Ad spend level is known at the time of inference but sales performance is not. Then we can configure UniversalTransformer as follows:
        >>> model = bolt.UniversalTransformer(
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
  universal_transformer_temporal_submodule.def(
      "numerical", sequential_classifier::TemporalConfig::numerical,
      py::arg("column_name"), py::arg("history_length"),
      py::arg("column_known_during_inference") = false,
      R"pbdoc(
    Temporal numerical config. Use this object to configure how a 
    numerical column is tracked over time. 

    Args:
        column_name (str): The name of the tracked column.
        history_length (int): Amount of time to look back. Time is in terms 
            of the time granularity passed to the UniversalTransformer constructor.
        column_known_during_inference (bool): Optional. Whether the 
            value of the tracked column is known during inference. Defaults 
            to False.

    Example:
        >>> # Suppose each row of our data has the following columns: "product_id", "timestamp", "ad_spend", "sales_performance"
        >>> # We want to predict the current week's sales performance for each product using temporal context.
        >>> # For each product ID, we would like to track both their ad spend and sales performance over time.
        >>> # Ad spend is known at the time of inference but sales performance is not. Then we can configure UniversalTransformer as follows:
        >>> model = bolt.UniversalTransformer(
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
  py::class_<SequentialClassifier>(bolt_submodule, "UniversalTransformer",
                                   R"pbdoc( 
    ThirdAI's universal deep learning engine for a variety of supervised datasets with built in 
    explainabilty. UniversalTransformer can be used for Forecasting, Personalization Recommendation, Query-Product Recommendation,
    Text Classification and many more use cases.
    
    In addition to learning from
    the columns of a single row, UniversalTransformer can make use of "temporal context". For 
    example, if used to build a movie recommender, UniversalTransformer may use information 
    about the last 5 movies that a user has watched to recommend the next movie.
    Similarly, if used to forecast the outcome of marketing campaigns, UniversalTransformer may 
    use several months' worth of campaign history for each product to make better
    forecasts.
    
                                   )pbdoc")
      .def(py::init<std::map<std::string, sequential_classifier::DataType>,
                    std::map<std::string,
                             std::vector<std::variant<
                                 std::string,
                                 sequential_classifier::TemporalConfig>>>,
                    std::string, std::map<std::string, std::string>, std::string, uint32_t>(),
           py::arg("data_types"), py::arg("temporal_tracking_relationships"),
           py::arg("target"), py::arg("model_parameters"), py::arg("time_granularity") = "daily",
           py::arg("lookahead") = 0,
           R"pbdoc(  
    Constructor.

    Args:
        data_types (Dict[str, bolt.types.ColumnType]): A mapping from column name to column type. 
            This map specifies the columns that we want to pass into the model; it does 
            not need to include all columns in the dataset.

            ColumnType is one of:
                - `bolt.types.categorical(n_unique_values: int)`
                - `bolt.types.numerical()`
                - `bolt.types.text(average_n_words: int=None)`
                - `bolt.types.date()`
            See bolt.types for details.

            If `temporal_tracking_relationships` is non-empty, there must be a 
            `bolt.types.date()` column. This column contains date strings in YYYY-MM-DD format.
            There can only be one `bolt.types.date()` column.
        temporal_tracking_relationships (Dict[str, List[str or bolt.temporal.TemporalConfig]]): Optional. A mapping 
            from column name to a list of either other column names or `bolt.temporal()` objects.
            This mapping tells UniversalTransformer what columns can be tracked over time for each key.
            For example, we may want to tell UniversalTransformer that we want to track a user's watch 
            history by passing in a map like `{"user_id": ["movie_id"]}`.

            If we provide a mapping from a string to a list of strings like the above, 
            the temporal tracking configuration will be autotuned. Alternatively, you can specify the `history_length` by 
            passing in `bolt.temporal` objects intead of strings (please refer to the examples section below).

            `bolt.temporal` object is one of:
                - `bolt.temporal.categorical(column_name: str, track_last_n: int, column_known_during_inference: bool=False)`
                - `bolt.temporal.numerical(column_name: str, history_length: int, column_known_during_inference: bool=False)`
            
            See bolt.temporal for details.
        target (str): Name of the column that contains the value to be predicted by
            UniversalTransformer. If the target column is of type `bolt.types.categorical()`, the task is automatically assumed to be classification. 
            If the target column is of type `bolt.types.numerical()`, the task us automatically assumed to be regression.
        model_parameters (Dict[str, str]): `Optional.` Model parameters to accomodate any hyper parameter changes desired. 
            Parameters like hidden embedding size, sparsity can be changed.
        time_granularity (str): Optional. 
            Either of:
                - "daily"/"d" 
                - "weekly"/"w"
                - "biweekly"/"b" 
                - "monthly"/"m"
            

            Interval of time that we are interested in. Temporal numerical 
            features are grouped according to this time granularity. 
            

            E.g. if `time_granularity="w"` and the numerical values on days 1 and 2 are
            345.25 and 201.1 respectively, then UniversalTransformer captures a single numerical 
            value of 546.26 for the week instead of individual values for the two days.
            

            Defaults to "daily".
        lookahead (str): `Optional.` How far into the future the model needs to predict. This length of
            time is in terms of time_granularity. 
            
            E.g. 'time_granularity="daily"` and `lookahead=5` means the model needs to learn to predict 5 days ahead. Defaults to 0
            (predict the target in the current row only).

    Examples:
        >>> # Example 1: Forecasting
        >>> # Suppose each row of our data has the following columns: "product_id", "timestamp", "ad_spend", "sales_quantity", "sales_performance"
        >>> # We want to predict next week's sales performance for each product using temporal context.
        >>> # For each `product_id`, we would like to track both their ad spend and sales quantity over time.
        >>> model = bolt.UniversalTransformer(
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


        >>> # Example 2: Personalized Recommendations
        >>> # Alternatively suppose our data has the following columns: "user_id", "movie_id", "hours_watched", "timestamp"
        >>> # We want to build a movie recommendation system.
        >>> # Then we may configure UniversalTransformer as follows:
        >>> model = bolt.UniversalTransformer(
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
        

        >>> # Example 3: Query-Category Recommendation 
        >>> # Alternatively suppose our data has the following columns: "query", "categories"
        >>> # We want to build a category classification given an input query.
        >>> # sample data : yellow shirt, Mens Shirts;Women Shirts;Shirts
        >>> # Then we may configure UniversalTransformer as follows:
        >>> model = bolt.UniversalTransformer(
                data_types={
                    "query": bolt.types.text(embedding="medium"),
                    "categories": bolt.types.categorical_text(n_unique_classes=3000, sep=";"),
                },
                target="categories",
                model_parameters={"embedding_representation":"1024"}
            )
        

        >>> # Example 4: Sentiment Classification
        >>> # Alternatively suppose our data has the following columns: "sentence", "sentiment"
        >>> # We want to build a category classification given an input query.
        >>> # sample data : i like the movie<tab>positive
        >>> # Then we may configure UniversalTransformer as follows:
        >>> model = bolt.UniversalTransformer(
                data_types={
                    "sentence": bolt.types.text(),
                    "sentiment": bolt.types.categorical(n_unique_classes=2),
                },
                target="sentiment"
            )


    Notes:
        - Refer to the documentation bolt.types.ColumnType and bolt.temporal.TemporalConfig to better understand column types 
          and temporal tracking configurations.

    )pbdoc")

      .def("train", &SequentialClassifier::train, py::arg("train_filename"),
           py::arg("train_config"),
           R"pbdoc(  
    Trains the model using the data provided in train_file.

    Args:
        train_file (str): The path to the training dataset. The dataset
            has to be a CSV file with a header. All columns names used in the `__init__`
            step have to be present in the csv. Additional columns will be ignored.
        train_config (TrainConfig): please see the examples and refer to `training_config` 
            documentation for more details

    Returns:
        Dict[Str, List[float]]:
        A dictionary from metric name to a list of the value of that metric 
        for each epoch (this also always includes an entry for 'epoch_times'
        measured in seconds). The metrics that are returned are the metrics 
        passed to the `metrics` parameter.
    
    Example:
        >>> train_config = (
                bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=10)
                .with_metrics(["recall@1", "recall@10"])
                .with_save_parameters(save_prefix="model", save_frequency=32)
            )
        >>> metrics = model.train(
                train_file="train_file.csv", train_config=train_config,
            )
        >>> print(metrics)
        {'epoch_times': [1.7, 3.4, 5.2], 'recall@1': [0.0922, 0.187, 0.268], 'recall@10': [0.4665, 0.887, 0.9685]}
    
    Notes:
        - If temporal tracking relationships are provided, UniversalTransformer can make better predictions 
          by taking temporal context into account. For example, UniversalTransformer may keep track of 
          the last few movies that a user has watched to better recommend the next movie.
          `model.train()` automatically updates UniversalTransformer's temporal context.
        - `model.train()` resets UniversalTransformer's temporal context at the start of training to 
          prevent unwanted information from leaking into the training routine.
           
    TODO:
        - Add train_config support in UniversalTransformer aka sequential)pbdoc")
    
      .def("embedding_representation", &SequentialClassifier::hiddenRepresentation,
            py::arg("input_sample"),
           R"pbdoc(  
    Provide embedding representation from the penultimate layer.

    Args:
        input_sample (Dict[str, str]): The input sample as a dictionary 
            where the keys are column names as specified in data_types and the "
            values are the respective column values. 

    Returns:
        1-D Numpy array of embedding dimension
    
    Example:
        >>> # Suppose we configure and train UniversalTransformer as follows:
        >>> model = bolt.UniversalTransformer(
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
                train_file="train_file.csv", train_config=train_config
            )
        >>> # Make a single prediction
        >>> embedding = model.embedding_representation(
                input_sample={"user_id": "A33225", "timestamp": "2022-02-02", "special_event": "christmas"}, top_k=3
            )
        >>> print(embedding)
        [0.12, 0.14, ...]
    
    Notes: 
        - Only penultimate layer representations can be extracted.
        - embedding representation will be dense irrespective of sparsity.
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
    Evaluates the model predictions with respect to the output classes on a validation/test 
    dataset. Optionally, writes top k predictions to a file if output file 
    name is provided for external evaluation. This method cannot be called 
    on an untrained model.

    Args:
        validation_file (str): The path to the validation dataset to 
            evaluate on. The dataset has to be a CSV file with a header (with the columns that were used in the training).
        metrics (List[str]): Metrics to track during training. Defaults to 
            ["precision@1"]. Metrics are currently restricted to 'precision@1' and 'f_measure(t)' 
            where t is a threshold between 0 and 1.
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
                validation_file="validation_file.csv", metrics=["precision@1", "f_measure(0.9)"], output_file="predictions.txt", write_top_k_to_file=10
            )
        >>> print(metrics)
        {'test_time': 20.0, 'precision@1': 0.0922, 'f_measure(0.9)': 0.4665}
    
    Notes: 
        - If temporal tracking relationships are provided, UniversalTransformer can make better predictions 
          by taking temporal context into account. For example, UniversalTransformer may keep track of 
          the last few movies that a user has watched to better recommend the next movie.
          `model.evaluate()` automatically updates UniversalTransformer's temporal context.
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
        >>> # Suppose we configure and train UniversalTransformer as follows:
        >>> model = bolt.UniversalTransformer(
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
        >>> train_config = (
                        bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=10)
                        .with_metrics(["recall@1", "recall@10"])
                        .with_save_parameters(save_prefix="model", save_frequency=32)
                    )
        >>> model.train(
                train_file="train_file.csv", train_config=train_config
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

        - If temporal tracking relationships are provided, UniversalTransformer can make better predictions 
          by taking temporal context into account. For example, UniversalTransformer may keep track of 
          the last few movies that a user has watched to better recommend the next movie. 
          Thus, UniversalTransformer is at its best when its internal temporal context gets updated with
          new true samples. `model.predict()` does not update UniversalTransformer's temporal context.
          To do this, we need to use `model.index()`. Read about `model.index()` for details.
           )pbdoc")
      .def("explain", &SequentialClassifier::explain, py::arg("input_sample"),
           py::arg("target") = std::nullopt, py::arg("comprehensive") = std::nullopt,
           R"pbdoc(  
    Identifies the columns that are most responsible for a predicted outcome 
    and provides a numerical insight into the column's content/value.
    
    If a target is provided, the model will identify the columns that need 
    to change for the model to predict the target class.
    
    Args:
        input_sample (Dict[str, str]): The input sample as a dictionary 
            where the keys are column names as specified in data_types and the
            values are the respective column values. 
        target (str): Optional. The desired target class. If provided, the
            method will identify the columns that need to change for the model to 
            predict the target class.
        comprehensive (bool): Optional. Comprehensive explanation in case of text inputs.
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
        >>> # Example 1: Suppose we configure and train UniversalTransformer as follows:
        >>> model = bolt.UniversalTransformer(
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
        >>> train_config = (
                        bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=10)
                        .with_metrics(["recall@1", "recall@10"])
                        .with_save_parameters(save_prefix="model", save_frequency=32)
                    )
        >>> model.train(
                train_file="train_file.csv", train_config=train_config
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
    Example:
        >>> # Example 2: Suppose we configure and train UniversalTransformer as follows for a loan approval system:
        >>> model = bolt.UniversalTransformer(
                data_types={
                    "user_id": bolt.types.categorical(n_unique_classes=5000),
                    "age": bolt.types.numerical(),
                    "education": bolt.types.categorical(n_unique_classes=20),
                    "approved": bolt.types.categorical(n_unique_classes=2)
                },
                target="education"
            )
        >>> model.train(
                train_file="train_file.csv", train_config=train_config
            )
        >>> # Make a single prediction
        >>> explanations = model.explain(
                input_sample={"user_id": "A33225", "age": "25", "education": "graduate"}, target="approved"
            )
        >>> print(explanations[0].column_name)
        "education"
        >>> print(explanations[0].percentage_significance)
        65.4
        >>> print(explanations[1].column_name)
        "age"
        >>> print(explanations[1].percentage_significance)
        15.3

    Example:
        >>> # Example 3: Suppose we configure and train UniversalTransformer as follows for sentiment analysis.
        >>> model = bolt.UniversalTransformer(
                data_types={
                    "sentence": bolt.types.text(),
                    "sentiment": bolt.types.categorical(n_unique_classes=2),
                },
                target="sentiment"
            )
        >>> model.train(
                train_file="train_file.csv", train_config=train_config
            )
        >>> # Make a single prediction
        >>> explanations = model.explain(
                input_sample={"sentence": "i really like the movie"}, comprehensive=true
            )
        >>> print(explanations)
        {sentiment:positive, reason: "'really like'"} # index of words responsible

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
        - If temporal tracking relationships are provided, UniversalTransformer can make better predictions 
          by taking temporal context into account. For example, UniversalTransformer may keep track of 
          the last few movies that a user has watched to better recommend the next movie. 
          Thus, UniversalTransformer is at its best when its internal temporal context gets updated with
          new true samples. `model.explain()` does not update UniversalTransformer's temporal context.
          To do this, we need to use `model.index()`. Read about `model.index()` for details.
           )pbdoc")
      .def("index", &SequentialClassifier::indexSingle, py::arg("sample"),
           R"pbdoc(

    Indexes a single true sample to keep UniversalTransformer's temporal context up to date.

    If temporal tracking relationships are provided, UniversalTransformer can make better predictions 
    by taking temporal context into account. For example, UniversalTransformer may keep track of 
    the last few movies that a user has watched to better recommend the next movie. 
    Thus, UniversalTransformer is at its best when its internal temporal context gets updated with
    new true samples. `model.index()` does exactly this. 

    Args: 
        input_sample (Dict[str, str]): The input sample as a dictionary 
            where the keys are column names as specified in data_types and the "
            values are the respective column values. 

    Example:
        >>> # Suppose we configure and train UniversalTransformer to do movie recommendation as follows:
        >>> model = bolt.UniversalTransformer(
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
        >>> # We can call model.index() to keep UniversalTransformer's temporal context up to date.
        >>> model.index(
                input_sample={"user_id": "A33225", "timestamp": "2022-02-02", "special_event": "christmas", "movie_title": "Die Hard 3"}
            )
          )pbdoc")
      .def("save", &SequentialClassifier::save, py::arg("filename"),
           R"pbdoc(  
    Serializes an instance of UniversalTransformer into a file on disk. The serialized UniversalTransformer includes 
    its current temporal context.
    
    Args:
        filename (str): The file on disk to serialize this instance of UniversalTransformer into.

    Example:
        >>> model.save("universal_transformer_savefile.bolt")
           )pbdoc")
      .def_static("load", &SequentialClassifier::load, py::arg("filename"),
                  R"pbdoc(  
    Loads a serialized instance of UniversalTransformer from a file on disk. The loaded UniversalTransformer includes 
    the temporal context from before serialization.
    
    Args:
        filename (str): The file on disk to load the instance of UniversalTransformer from.

    Example:
        >>> model = bolt.UniversalTransformer.load("universal_transformer_savefile.bolt")
           )pbdoc");

  createBoltGraphSubmodule(bolt_submodule);

  return bolt_submodule;
}

}  // namespace thirdai::bolt::pythons