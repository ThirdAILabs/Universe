#include "BoltPython.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <auto_ml/src/deployment_config/dataset_configs/udt/DataTypes.h>
#include <dataset/src/DataLoader.h>
#include <dataset/src/batch_processors/TabularMetadataProcessor.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <search/src/Generator.h>
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
           py::arg("eval_config"), py::arg("validation_frequency") = 0,
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
    eval_config (bolt.EvalConfig): 
        See EvalConfig.
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

  py::class_<EvalConfig>(bolt_submodule, "EvalConfig")
      .def(py::init(&EvalConfig::makeConfig))
      .def("enable_sparse_inference", &EvalConfig::enableSparseInference)
      .def("with_metrics", &EvalConfig::withMetrics, py::arg("metrics"))
      .def("silence", &EvalConfig::silence)
      .def("return_activations", &EvalConfig::returnActivations);

  auto udt_types_submodule = bolt_submodule.def_submodule("types");

  py::class_<automl::deployment::DataType,
             automl::deployment::DataTypePtr>(  // NOLINT
      udt_types_submodule, "ColumnType", "Base class for bolt types.")
      .def("__str__", &automl::deployment::DataType::toString);

  py::class_<automl::deployment::CategoricalMetadataConfig,
             automl::deployment::CategoricalMetadataConfigPtr>(
      udt_types_submodule, "metadata")
      .def(py::init<std::string, std::string,
                    automl::deployment::ColumnDataTypes, char>(),
           py::arg("filename"), py::arg("key_column_name"),
           py::arg("data_types"), py::arg("delimiter") = ',',
           R"pbdoc(
    A configuration object for processing a metadata file to enrich categorical
    features from the main dataset. To illustrate when this is useful, suppose
    we are building a movie recommendation system. The contents of the training
    dataset may look something like the following:

    user_id,movie_id,timestamp
    A526,B894,2022-01-01
    A339,B801,2022-01-01
    A293,B801,2022-01-01
    ...

    If you have additional information about users or movies, such as users' 
    age groups or movie genres, you can use that information to enrich your 
    model. Adding these features into the main dataset as new columns is wasteful
    because the same users and movies ids will be repeated many times throughout
    the dataset. Instead, we can put them all in a metadata file and UDT will
    inject these features where appropriate.

    Args:
        filename (str): Path to metadata file. The file should be in CSV format.
        key_column_name (str): The name of the column whose values are used as
            keys to map metadata features back to values in the main dataset. 
            This column does not need to be passed into the `data_types` argument. 
        data_types (Dict[str, bolt.types.ColumnType]): A mapping from column name 
            to column type. Column type is one of:
            - `bolt.types.categorical`
            - `bolt.types.numerical`
            - `bolt.types.text`
            - `bolt.types.date`
        delimiter (str): Optional. Defaults to ','. A single character 
            (length-1 string) that separates the columns of the metadata file.
    
    Example:
        >>> for line in open("user_meta.csv"):
        >>>     print(line)
        user_id,age
        A526,52
        A531,22
        A339,29
        ...
        >>> deployment.UniversalDeepTransformer(
                data_types: {
                    "user_id": bolt.types.categorical(
                        delimiter=' ',
                        metadata=bolt.types.metadata(
                            filename="user_meta.csv", 
                            data_types={"age": bolt.types.numerical()}, 
                            key_column_name="user_id"
                        )
                    )
                }
                ...
            )
                             )pbdoc");
  py::class_<automl::deployment::CategoricalDataType,
             automl::deployment::DataType,
             automl::deployment::CategoricalDataTypePtr>(udt_types_submodule,
                                                         "categorical")
      .def(py::init<std::optional<char>,
                    automl::deployment::CategoricalMetadataConfigPtr>(),
           py::arg("delimiter") = std::nullopt, py::arg("metadata") = nullptr,
           R"pbdoc(
    Categorical column type. Use this object if a column contains categorical 
    data (each unique value is treated as a class). Examples include user IDs, 
    movie titles, or age groups.

    Args:
        delimiter (str): Optional. Defaults to None. A single character 
            (length-1 string) that separates multiple values in the same 
            column. This can only be used for the target column. If not 
            provided, UDT assumes that there is only one value in the column.
        metadata (metadata): Optional. A metadata object to be used when there 
            is a separate metadata file corresponding to this categorical 
            column.
    
    Example:
        >>> deployment.UniversalDeepTransformer(
                data_types: {
                    "user_id": bolt.types.categorical(
                        delimiter=' ',
                        metadata=bolt.types.metadata(filename="user_meta.csv", data_types={"age": bolt.types.numerical()}, key_column_name="user_id")
                    )
                }
                ...
            )
                             )pbdoc");
  py::class_<automl::deployment::NumericalDataType,
             automl::deployment::DataType,
             automl::deployment::NumericalDataTypePtr>(udt_types_submodule,
                                                       "numerical")
      .def(py::init<std::pair<double, double>, std::string>(), py::arg("range"),
           py::arg("granularity") = "m",
           R"pbdoc(
    Numerical column type. Use this object if a column contains numerical 
    data (the value is treated as a quantity). Examples include hours of 
    a movie watched, sale quantity, or population size.

    Args:
        range (tuple(float, float)): The expected range (min to max) of the
            numeric quantity. The more accurate this range to the test data, the 
            better the model performance.
        granularity (str): Optional. One of "extrasmall"/"xs", "small"/"s", "medium"/"m",
            "large"/"l" or "extralarge"/"xl" . Defaults to "m".


    Example:
        >>> deployment.UniversalDeepTransformer(
                data_types: {
                    "hours_watched": bolt.types.numerical(range=(0, 25), granularity="xs")
                }
                ...
            )
                             )pbdoc");
  py::class_<automl::deployment::TextDataType, automl::deployment::DataType,
             automl::deployment::TextDataTypePtr>(udt_types_submodule, "text")
      .def(py::init<std::optional<double>, bool>(),
           py::arg("average_n_words") = std::nullopt,
           py::arg("use_attention") = false,
           R"pbdoc(
    Text column type. Use this object if a column contains text data 
    (the meaning of the text matters). Examples include descriptions, 
    search queries, and user bios.

    Args:
        average_n_words (float): Optional. Average number of words in the 
            text column in each row. If provided, UDT may make 
            optimizations as appropriate.
        use_attention (bool): Optional. If true, udt is guaranteed to
            use attention when processing this text column. Otherwise, 
            udt will only use attention when appropriate.
    
    Example:
        >>> deployment.UniversalDeepTransformer(
                data_types: {
                    "user_motto": bolt.types.text(average_n_words=10),
                    "user_bio": bolt.types.text()
                }
                ...
            )

                             )pbdoc");
  py::class_<automl::deployment::DateDataType, automl::deployment::DataType,
             automl::deployment::DateDataTypePtr>(udt_types_submodule, "date")
      .def(py::init<>(),
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

  auto udt_temporal_submodule = bolt_submodule.def_submodule("temporal");

  py::class_<automl::deployment::TemporalConfig>(  // NOLINT
      udt_temporal_submodule, "TemporalConfig",
      "Base class for temporal feature configs.");

  udt_temporal_submodule.def("categorical",
                             automl::deployment::TemporalConfig::categorical,
                             py::arg("column_name"), py::arg("track_last_n"),
                             py::arg("column_known_during_inference") = false,
                             py::arg("use_metadata") = false,
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
        use_metadata (bool): Optional. Whether to use the metadata of the N 
            tracked items, if metadata is provided in the corresponding 
            categorical column type object. Ignored if no metadata is provided. 
            Defaults to False.

    Example:
        >>> # Suppose each row of our data has the following columns: "product_id", "timestamp", "ad_spend_level", "sales_performance"
        >>> # We want to predict the current week's sales performance for each product using temporal context.
        >>> # For each product ID, we would like to track both their ad spend level and sales performance over time.
        >>> # Ad spend level is known at the time of inference but sales performance is not. Then we can configure UDT as follows:
        >>> model = deployment.UniversalDeepTransformer(
                data_types={
                    "product_id": bolt.types.categorical(),
                    "timestamp": bolt.types.date(),
                    "ad_spend_level": bolt.types.categorical(),
                    "sales_performance": bolt.types.categorical(),
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
  udt_temporal_submodule.def("numerical",
                             automl::deployment::TemporalConfig::numerical,
                             py::arg("column_name"), py::arg("history_length"),
                             py::arg("column_known_during_inference") = false,
                             R"pbdoc(
    Temporal numerical config. Use this object to configure how a 
    numerical column is tracked over time. 

    Args:
        column_name (str): The name of the tracked column.
        history_length (int): Amount of time to look back. Time is in terms 
            of the time granularity passed to the UDT constructor.
        column_known_during_inference (bool): Optional. Whether the 
            value of the tracked column is known during inference. Defaults 
            to False.

    Example:
        >>> # Suppose each row of our data has the following columns: "product_id", "timestamp", "ad_spend", "sales_performance"
        >>> # We want to predict the current week's sales performance for each product using temporal context.
        >>> # For each product ID, we would like to track both their ad spend and sales performance over time.
        >>> # Ad spend is known at the time of inference but sales performance is not. Then we can configure UDT as follows:
        >>> model = deployment.UniversalDeepTransformer(
                data_types={
                    "product_id": bolt.types.categorical(),
                    "timestamp": bolt.types.date(),
                    "ad_spend": bolt.types.numerical(range=(0, 10000)),
                    "sales_performance": bolt.types.categorical(),
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
}

void createModelsSubmodule(py::module_& bolt_submodule) {
  auto models_submodule = bolt_submodule.def_submodule("models");

#if THIRDAI_EXPOSE_ALL
  py::class_<bolt::QueryCandidateGeneratorConfig,
             bolt::QueryCandidateGeneratorConfigPtr>(models_submodule,
                                                     "GeneratorConfig")
      .def(py::init<std::string, uint32_t, uint32_t, uint32_t,
                    std::vector<uint32_t>, std::optional<uint32_t>, uint32_t,
                    uint32_t, uint32_t>(),
           py::arg("hash_function"), py::arg("num_tables"),
           py::arg("hashes_per_table"), py::arg("range"), py::arg("n_grams"),
           py::arg("reservoir_size") = std::nullopt,
           py::arg("source_column_index") = 0,
           py::arg("target_column_index") = 0, py::arg("batch_size") = 10000,
           R"pbdoc(
    Initializes a QueryCandidateGeneratorConfig object.

     Args:
        hash_function (str): A specific hash function 
            to use. Supported hash functions include MinHash
            and DensifiedMinHash
        num_tables (int): Number of hash tables to construct.
        hashes_per_table (int): Number of hashes per table.
        range (int) : The range for the hash function used. 
        n_grams (List[int]): List of N-gram blocks to use. 
        reservoir_size (int): Reservoir size to use when the flash index is 
            constructed with reservoir sampling. 
        source_column_index (int): Index of the column in the input CSV
            that contains incorrect queries.
        target_column_index (int): Index of the column in the input CSV
            that contains the target queries for reformulation. 
        batch_size (int): batch size. It is defaulted to 10000. 
    Returns: 
        QueryCandidateGeneratorConfig

    Example:
        >>> generator_config = bolt.models.GeneratorConfig(
                hash_function="DensifiedMinHash",
                num_tables=100,
                hashes_per_table=15,
                input_dim=100,
                top_k=5,
                n_grams=[3,4],
                has_incorrect_queries=True,
                batch_size=10000,
            )
            )pbdoc")
      .def("save", &bolt::QueryCandidateGeneratorConfig::save,
           py::arg("file_name"),
           R"pbdoc(
    Saves a query candidate generator config object at the specified file path. 
    This can be used to provide a query candidate generator architecture to customers.

    Args:
        file_name (str): File path specification for where to save the 
                generator configuration object. 

    Returns:
        None

            )pbdoc")

      .def_static("load", &bolt::QueryCandidateGeneratorConfig::load,
                  py::arg("config_file_name"),
                  R"pbdoc(
    Loads a query candidate generator config object from a specific file location. 

    Args:
        config_file_name (str): Path to the file containing a saved config.

        Returns:
            QueryCandidateGeneratorConfig:

            )pbdoc");

#endif
}

}  // namespace thirdai::bolt::python
