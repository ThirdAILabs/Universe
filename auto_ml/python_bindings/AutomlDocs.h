#pragma once

namespace thirdai::automl::python::docs {

const char* const MODEL_PIPELINE_INIT_FROM_CONFIG = R"pbdoc(
Constructs a ModelPipeline from a deployment config and a set of input parameters.

Args:
    deployment_config (DeploymentConfig): A config for the ModelPipeline.
    parameters (Dict[str, Union[bool, int, float, str]]): A mapping from parameter 
        names to values. This is used to pass in any additional parameters required
        to construct the desired model. The keys should be the names of the parameters
        as strings and the values can be integers, floats, strings, or bools depending
        on what the type of the parameter is. An exception will be thrown if a required
        parameter is not specified or if the the parameter is not the right type.

Returns
    ModelPipeline:

Examples:
    >>> model = bolt.Pipeline(
            deployment_config=deployment.DeploymentConfig(...),
            parameters={"size": "large", "output_dim": num_classes, "delimiter": ","},
        )

)pbdoc";

const char* const MODEL_PIPELINE_INIT_FROM_SAVED_CONFIG = R"pbdoc(
Constructs a ModelPipeline from a serialized deployment config and a set of input 
parameters.

Args:
    config_path (str): A path to a serialized deployment config for the ModelPipeline.
    parameters (Dict[str, Union[bool, int, float, str]]): A mapping from parameter 
        names to values. This is used to pass in any additional parameters required
        to construct the desired model. The keys should be the names of the parameters
        as strings and the values can be integers, floats, strings, or bools depending
        on what the type of the parameter is. An exception will be thrown if a required
        parameter is not specified or if the the parameter is not the right type.

Returns
    ModelPipeline:

Examples:
    >>> model = bolt.Pipeline(
            config_path="path_to_a_config",
            parameters={"size": "large", "output_dim": num_classes, "delimiter": ","},
        )

)pbdoc";

const char* const MODEL_PIPELINE_TRAIN_FILE = R"pbdoc(
Trains a ModelPipeline on a given dataset using a file on disk.

Args:
    filename (str): Path to the dataset file.
    train_config (bolt.TrainConfig): The training config specifies the number
        of epochs and learning_rate, and optionally allows for specification of a
        validation dataset, metrics, callbacks, and how frequently to log metrics 
        during training. 
    batch_size (Optional[int]): This is an optional parameter indicating which batch
        size to use for training. If not specified the default batch size from the 
        TrainEvalParameters is used.
    validation (Optional[bolt.Validation]): This is an optional parameter that specifies 
        a validation dataset, metrics, and interval to use during training.
    max_in_memory_batches (Optional[int]): The maximum number of batches to load in
        memory at a given time. If this is specified then the dataset will be processed
        in a streaming fashion.

Returns:
    None

Examples:
    >>> train_config = bolt.TrainConfig(
            epochs=5, learning_rate=0.01
        ).with_metrics(["mean_squared_error"])
    >>> model.train(
            filename="./train_file", train_config=train_config , max_in_memory_batches=12
        )

)pbdoc";

const char* const MODEL_PIPELINE_TRAIN_DATA_LOADER = R"pbdoc(
Trains a ModelPipeline on a given dataset using any DataLoader.

Args:
    data_source (dataset.DataLoader): A data loader for the given dataset.
    train_config (bolt.TrainConfig): The training config specifies the number
        of epochs and learning_rate, and optionally allows for specification of a
        validation dataset, metrics, callbacks, and how frequently to log metrics 
        during training. 
    validation (Optional[bolt.Validation]): This is an optional parameter that specifies 
        a validation dataset, metrics, and interval to use during training.
    max_in_memory_batches (Optional[int]): The maximum number of batches to load in
        memory at a given time. If this is specified then the dataset will be processed
        in a streaming fashion.

Returns:
    None

Examples:
    >>> train_config = bolt.TrainConfig(epochs=5, learning_rate=0.01)
    >>> model.train(
            data_source=dataset.CSVDataLoader(...), train_config=train_config, max_in_memory_batches=12
        )

)pbdoc";

const char* const MODEL_PIPELINE_EVALUATE_FILE = R"pbdoc(
Evaluates the ModelPipeline on the given dataset and returns a numpy array of the 
activations.

Args:
    filename (str): Path to the dataset file.
    eval_config (Option[bolt.EvalConfig]): The predict config is optional
        and allows for specification of metrics to compute and whether to use sparse
        inference.
    return_predicted_class (bool): Optional, defaults to false. When true the model
        will output the predicted class for each sample rather than the activations 
        of the final layer. This has no effect for regression models.
    return_metrics (bool): Optional, defaults to false. When true, the model will
        output the evaluation metrics rather than activations of the final layer.
        If true, this nullifies the `return_predicted_class` argument.

Returns:
    (np.ndarray or Tuple[np.ndarray, np.ndarray] or Dict): 
    If return_metrics = True, returns a dictionary that maps metric names to their
    values. Otherwise, returns a numpy array of the activations if the output is 
    dense, or a tuple of the active neurons and activations if the output is sparse. 
    The shape of each array will be (dataset_length, num_nonzeros_in_output).

Examples:
    >>> eval_config = bolt.EvalConfig().with_metrics(["categorical_accuracy"])
    >>> activations = model.evaluate(filename="./test_file", eval_config=eval_config)

)pbdoc";

const char* const MODEL_PIPELINE_EVALUATE_DATA_LOADER = R"pbdoc(
Evaluates the ModelPipeline on the given dataset and returns a numpy array of the 
activations.

Args:
    data_source (dataset.DataLoader): A data loader for the given dataset.
    eval_config (Option[bolt.EvalConfig]): The predict config is optional
        and allows for specification of metrics to compute and whether to use sparse
        inference.
    return_predicted_class (bool): Optional, defaults to false. When true the model
        will output the predicted class for each sample rather than the activations 
        of the final layer. This has no effect for regression models.
    return_metrics (bool): Optional, defaults to false. When true, the model will
        output the evaluation metrics rather than activations of the final layer.
        If true, this nullifies the `return_predicted_class` argument.

Returns:
    (np.ndarray or Tuple[np.ndarray, np.ndarray] or Dict): 
    If return_metrics = True, returns a dictionary that maps metric names to their
    values. Otherwise, returns a numpy array of the activations if the output is 
    dense, or a tuple of the active neurons and activations if the output is sparse. 
    The shape of each array will be (dataset_length, num_nonzeros_in_output).

Examples:
    >>> (active_neurons, activations) = model.evaluate(data_source=dataset.CSVDataLoader(...))

)pbdoc";

const char* const MODEL_PIPELINE_PREDICT = R"pbdoc(
Performs inference on a single sample.

Args:
    input_sample (str): A str representing the input. This will be processed in the 
        same way as the dataset, and thus should be the same format as a line in 
        the dataset, except with the label columns removed.
    use_sparse_inference (bool, default=False): Whether or not to use sparse inference.
    return_predicted_class (bool): Optional, defaults to false. When true the model
        will output the predicted class rather than the activations 
        of the final layer. This has no effect for regression models.
Returns: 
    (np.ndarray or Tuple[np.ndarray, np.ndarray]): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (num_nonzeros_in_output, ).

Examples:
    >>> activations = model.predict("The blue cat jumped")

)pbdoc";

const char* const MODEL_PIPELINE_EXPLAIN = R"pbdoc(
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
    
)pbdoc";

const char* const MODEL_PIPELINE_PREDICT_TOKENS = R"pbdoc(
Performs inference on a single sample represented as bert tokens

Args:
    tokens (List[int]): A list of integers representing bert tokens.
    use_sparse_inference (bool, default=False): Whether or not to use sparse inference.

Returns: 
    (np.ndarray or Tuple[np.ndarray, np.ndarray]): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (num_nonzeros_in_output, ).

Examples:
    >>> activations = model.predict_tokens(tokens=[9, 42, 19, 71, 33])

)pbdoc";

const char* const MODEL_PIPELINE_PREDICT_BATCH = R"pbdoc(
Performs inference on a batch of samples samples in parallel.

Args:
    input_samples (List[str]): A list of strings representing each sample. Each 
        input will be processed in the same way as the dataset, and thus should 
        be the same format as a line in the dataset, except with the label columns 
        removed.
    use_sparse_inference (bool, default=False): Whether or not to use sparse inference.
    return_predicted_class (bool): Optional, defaults to false. When true the model
        will output the predicted class for each sample rather than the activations 
        of the final layer. This has no effect for regression models.

Returns: 
    (np.ndarray or Tuple[np.ndarray, np.ndarray]): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (batch_size, num_nonzeros_in_output).

Examples:
    >>> activations = model.predict_batch([
            "The cat ran",
            "The dog sat", 
            "The cow ate grass"
        ])

)pbdoc";

const char* const MODEL_PIPELINE_SAVE = R"pbdoc(
Saves a serialized version of the ModelPipeline.

Args:
    filename (str): The file to save the serialized ModelPipeline in.

Returns:
    None

)pbdoc";

const char* const MODEL_PIPELINE_LOAD = R"pbdoc(
Loads a saved deployment config. 

Args:
    filename (str): The file which contains the serialized config.

Returns:
    DeploymentConfig:

)pbdoc";

const char* const MODEL_PIPELINE_GET_DATA_PROCESSOR = R"pbdoc(
Returns the data processor of the model pipeline.
)pbdoc";

const char* const TEMPORAL_CONTEXT_RESET = R"pbdoc(
Resets UDT's temporal trackers. When temporal relationships are supplied, 
UDT assumes that we feed it data in chronological order. Thus, if we break 
this assumption, we need to first reset the temporal trackers. 
An example of when you would use this is when you want to repeat the UDT 
training routine on the same dataset. Since you would be training on data from 
the same time period as before, we need to first reset the temporal trackers so 
that we don't double count events.

)pbdoc";

const char* const TEMPORAL_CONTEXT_UPDATE = R"pbdoc(
Updates the temporal trackers.

If temporal tracking relationships are provided, UDT can make better predictions 
by taking temporal context into account. For example, UDT may keep track of 
the last few movies that a user has watched to better recommend the next movie. 
Thus, UDT is at its best when its internal temporal context gets updated with
new true samples. `.update_temporal_trackers()` does exactly this. 

)pbdoc";

const char* const TEMPORAL_CONTEXT_UPDATE_BATCH = R"pbdoc(
Updates the temporal trackers with batch input.

If temporal tracking relationships are provided, UDT can make better predictions 
by taking temporal context into account. For example, UDT may keep track of 
the last few movies that a user has watched to better recommend the next movie. 
Thus, UDT is at its best when its internal temporal context gets updated with
new true samples. Just like `.update_temporal_trackers()`, 
`.batch_update_temporal_trackers()` does exactly this, except with a batch input.

)pbdoc";

const char* const VALIDATION = R"pbdoc(
Creates a validation object that stores the necessary information for the model 
to perform validation during training.

Args:
    filename (str): The name of the validation file.
    metrics (List[str]): The metrics to compute for validation.
    interval (Optional[int]): The interval, in number of batches, between computing 
        validation. For instance, `interval=10` means that validation metrics will 
        be computed every 10 batches. If it is not specified then validation will 
        be done after each epoch.
    use_sparse_inference (bool): Optional, defaults to False. When True, sparse 
        inference will be used during validation.

Examples:
    >>> validation = bolt.Validation(
            filename="validation.csv", metrics=["categorical_accuracy"], interval=10
        )
    >>> model.train("train.csv", epochs=5, validation=validation)
)pbdoc";

const char* const UDT_CLASS = R"pbdoc(
UniversalDeepTransformer (UDT) An all-purpose classifier for tabular datasets and 
generator model for performing query reformulation. 
In addition to learning from the columns of a single row, the UDT classifier can
make use of "temporal context". For example, if used to build a movie recommender,
UDT may use information about the last 5 movies that a user has watched to recommend
the next movie. Similarly, if used to forecast the outcome of marketing campaigns, 
UDT may use several months' worth of campaign history for each product to make 
better forecasts.

In addition to exposing a classifier model for tabular data, UDT exposes a generator
model for query reformulation tasks. For instance, given a dataset with consisting of
queries that have spelling mistakes, UDT can be used to generate relevant reformulations
with the correct spelling. 
)pbdoc";

const char* const UDT_INIT = R"pbdoc(
UniversalDeepTransformer (UDT) Constructor.

Args:
    data_types (Dict[str, bolt.types.ColumnType]): A mapping from column name to column type. 
        This map specifies the columns that we want to pass into the model; it does 
        not need to include all columns in the dataset.

        Column type is one of:
        - `bolt.types.categorical`
        - `bolt.types.numerical`
        - `bolt.types.text`
        - `bolt.types.date`
        See bolt.types for details.

        If `temporal_tracking_relationships` is non-empty, there must one and only one
        bolt.types.date() column. This column contains date strings in YYYY-MM-DD format.
    temporal_tracking_relationships (Dict[str, List[Union[str, bolt.temporal.TemporalConfig]]]): Optional. 
        A mapping from column name to a list of either other column names or bolt.temporal objects.
        This mapping tells UDT what columns can be tracked over time for each key.
        For example, we may want to tell UDT that we want to track a user's watch 
        history by passing in a map like `{"user_id": ["movie_id"]}`

        If we provide a mapping from a string to a list of strings like the above, 
        the temporal tracking configuration will be autotuned. You can achieve finer 
        grained control by passing in bolt.temporal objects intead of strings.

        A bolt.temporal object is one of:
        - `bolt.temporal.categorical`
        - `bolt.temporal.numerical`
        See bolt.temporal for details.
    target (str): Name of the column that contains the value to be predicted by
        UDT. The target column has to be a categorical column.
    n_target_classes (int): Number of target classes.
    integer_target (bool): Whether the target classes are integers in the range 0 to n_target_classes - 1.
    time_granularity (str): Optional. Either `"daily"`/`"d"`, `"weekly"`/`"w"`, `"biweekly"`/`"b"`, 
        or `"monthly"`/`"m"`. Interval of time that UDT should use for temporal features. Temporal numerical 
        features are clubbed according to this time granularity. E.g. if 
        `time_granularity="w"` and the numerical values on days 1 and 2 are
        345.25 and 201.1 respectively, then UDT captures a single numerical 
        value of 546.26 for the week instead of individual values for the two days.
        Defaults to "daily".
    lookahead (str): Optional. How far into the future the model should predict. This length of
        time is in terms of time_granularity. E.g. 'time_granularity="daily"` and 
        `lookahead=5` means that the model should learn to predict 5 days into the future. Defaults to 0
        (predict the current value of the target).
    delimiter (str): Optional. Defaults to ','. A single character 
        (length-1 string) that separates the columns of the CSV training / validation dataset.
    model_config (Optional[str]): This overwrites the autotuned model with a custom model 
        defined by the given config file.

Examples:
    >>> # Suppose each row of our data has the following columns: "product_id", "timestamp", "ad_spend", "sales_quantity", "sales_performance"
    >>> # We want to predict next week's sales performance for each product using temporal context.
    >>> # For each product ID, we would like to track both their ad spend and sales quantity over time.
    >>> model = bolt.UniversalDeepTransformer(
            data_types={
                "product_id": bolt.types.categorical(),
                "timestamp": bolt.types.date(),
                "ad_spend": bolt.types.numerical(range=(0, 10000)),
                "sales_quantity": bolt.types.numerical(range=(0, 20)),
                "sales_performance": bolt.types.categorical(),
            },
            temporal_tracking_relationships={
                "product_id": [
                    # We can use multiple bolt.temporal objects with the same column name but 
                    # different history lengths to track different intervals of the same variable
                    # Track last 5 weeks of ad spend
                    bolt.temporal.numerical(column_name="ad_spend", history_length=5),
                    # Track last 10 weeks of ad spend
                    bolt.temporal.numerical(column_name="ad_spend", history_length=10),
                    # Track last 5 weeks of sales performance
                    bolt.temporal.categorical(column_name="sales_performance", history_length=5),
                ]
            },
            target="sales_performance",
            n_target_classes=5,
            time_granularity="weekly",
            lookahead=2 # predict 2 weeks ahead
        )
    >>> # Alternatively suppose our data has the following columns: "user_id", "movie_id", "hours_watched", "timestamp"
    >>> # We want to build a movie recommendation system.
    >>> # Then we may configure UDT as follows:
    >>> model = bolt.UniversalDeepTransformer(
            data_types={
                "user_id": bolt.types.categorical(),
                "timestamp": bolt.types.date(),
                "movie_id": bolt.types.categorical(),
                "hours_watched": bolt.types.numerical(range=(0, 25)),
            },
            temporal_tracking_relationships={
                "user_id": [
                    "movie_id", # autotuned movie temporal tracking
                    bolt.temporal.numerical(column_name="hours_watched", history_length="5") # track last 5 days of hours watched.
                ]
            },
            target="movie_id",
            n_target_classes=3000
        )

Notes:
    - Refer to the documentation bolt.types.ColumnType and bolt.temporal.TemporalConfig to better understand column types 
      and temporal tracking configurations.
)pbdoc";

const char* const UDT_GENERATOR_INIT = R"pbdoc(
UniversalDeepTransformer (UDT) Constructor. 

Args:
    target_column (str): Column name specifying the target queries in the input dataset. 
        Queries in this column are the target that the UDT model learns to predict. 
    source_column (str): Column name specifying the source queries in the input dataset. 
        The UDT model uses is trained based on these queries. 
    dataset_size (str): The size of the input dataset. This size factor informs what
        UDT model to create. 

        The dataset size can be one of the following:
        - small
        - medium
        - large

Example:
    >>> # Suppose we have an input CSV dataset consisting of grammatically or syntactically
    >>> # incorrect queries that we want to reformulate. We will assume that the dataset also
    >>> # has a target correct query for each incorrect query. We can initialize a UDT model
    >>> # for query reformulation as follows:
    >>> model = bolt.UniversalDeepTransformer(
            target_column="queries_for_prediction", 
            source_column="incorrect_queries",
            dataset_size="medium"
        )
)pbdoc";

const char* const UDT_GENERATOR_TRAIN = R"pbdoc(
Trains a UniversalDeepTransformer (UDT) model for query reformulation on a given dataset 
using a file on disk.

Args:
    filename (str): Path to the dataset file.

Returns:
    None

Examples:
    >>> model = bolt.UniversalDeepTransformer(
            target_column_index=0, 
            source_column_index=1,
            dataset_size="medium"
        )
    >>> model.train(filename="./train_file")

Note:
    - The UDT model expects different training inputs for query reformulation versus
        other machine learning tasks. 
)pbdoc";

const char* const UDT_CLASS_NAME = R"pbdoc(
Returns the target class name associated with an output neuron ID.

Args:
    neuron_id (int): The index of the neuron in UDT's output layer. This is 
        useful for mapping the activations returned by `evaluate()` and 
        `predict()` back to class names.

Returns:
    str:
    The class names that corresponds to the given neuron_id.

Example:
    >>> activations = model.predict(
            input_sample={"user_id": "A33225", "timestamp": "2022-12-25", "special_event": "christmas"}
        )
    >>> top_recommendation = np.argmax(activations)
    >>> model.class_name(top_recommendation)
    "Die Hard"
)pbdoc";

const char* const UDT_GENERATOR_EVALUATE = R"pbdoc(
Evaluates the UniversalDeepTransformer (UDT) model on the given dataset and returns
a list of generated queries as reformulations of the incorrect queries. 

Args:
    filename (str): Path to the dataset file 
    top_k (int): The number of candidate query reformulations suggested by the UDT model.
        The default value for k is 5.

Returns:
    List[List[str]]
    Returns a list of k reformulations for each incorrect query to be reformulated in the 
    input dataset. 

Notes:
    - If the input dataset file contains pairs of correct and incorrect queries, this 
     method will also print out the recall value at k. 

Examples:
    >>> model = bolt.UniversalDeepTransformer(
            target_column_index=0, 
            source_column_index=1,
            dataset_size="medium"
        )
    >>> model.train(filename="./train_file")
    >>> reformulated_queries = model.evaluate(filename="./test_file", top_k=5)
)pbdoc";

const char* const UDT_PREDICT = R"pbdoc(
Performs inference on a single sample.

Args:
    input_sample (Dict[str, str]): The input sample as a dictionary 
        where the keys are column names and the values are the respective column values. 
    use_sparse_inference (bool, default=False): Whether or not to use sparse inference.

Returns: 
    (np.ndarray, Tuple[np.ndarray, np.ndarray], or List[int]): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (num_nonzeros_in_output, ). When the 
    `consecutive_integer_ids` argument of target column's categorical ColumnType
    object is set to False (as it is by default), UDT creates an internal 
    mapping between target class names and neuron ids. You can map neuron ids back to
    target class names by calling the `class_names()` method. If the `prediction_depth`
    of the model is > 1 and the task is classification then it will return a numpy array 
    of integers indicating the predicted class for each timstamp up to `prediction_depth`.

Examples:
    >>> # Suppose we configure UDT as follows:
    >>> model = bolt.UniversalDeepTransformer(
            data_types={
                "user_id": bolt.types.categorical(),
                "timestamp": bolt.types.date(),
                "special_event": bolt.types.categorical(),
                "movie_title": bolt.types.categorical()
            },
            temporal_tracking_relationships={
                "user_id": ["movie_title"]
            },
            target="movie_title",
            n_target_classes=500
        )
    >>> # Make a single prediction
    >>> activations = model.predict(
            input_sample={"user_id": "A33225", "timestamp": "2022-12-25", "special_event": "christmas"}
        )

Notes:
    - The values of columns that are tracked temporally may be unknown during inference
      (the column_known_during_inference attribute of the bolt.temporal objects are False
      by default). These columns do not need to be passed into `model.predict()`.
      For example, we did not pass the "movie_title" column to `model.predict()`.
      All other columns must be passed in.
    - If temporal tracking relationships are provided, UDT can make better predictions 
      by taking temporal context into account. For example, UDT may keep track of 
      the last few movies that a user has watched to better recommend the next movie. 
      Thus, UDT is at its best when its internal temporal context gets updated with
      new true samples. `model.predict()` does not update UDT's temporal context.
      To do this without retraining the model, we need to use `model.index()` or 
      `model.index_batch()`. Read about `model.index()` and `model.index_batch()` 
      for details.

)pbdoc";

const char* const UDT_GENERATOR_PREDICT = R"pbdoc(
Performs inference on a single sample.

Args:
    input_query (str): The input query as a string. 
    top_k (int): The number of candidate query reformulations suggested by the UDT model
        for this input. The default value for k is 5. 

Returns:
    List[str]
    Returns a list of k reformulations suggested by the UDT model for the given input
    sample.

Example:
    >>> model = bolt.UniversalDeepTransformer(
            target_column_index=0, 
            source_column_index=1,
            dataset_size="medium"
        )
    >>> model.train(filename="./train_file")
    >>> udt_refomulation_suggestions = model.predict(input_query="sample query", top_k=5)
    
)pbdoc";

const char* const UDT_PREDICT_BATCH = R"pbdoc(
Performs inference on a batch of samples in parallel.

Args:
    input_samples (List[Dict[str, str]]): A list of input sample as dictionaries 
        where the keys are column names as specified in data_types and the 
        values are the respective column values. 
    use_sparse_inference (bool, default=False): Whether or not to use sparse inference.

Returns: 
    (np.ndarray, Tuple[np.ndarray, np.ndarray], or List[List[int]]): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (batch_size, num_nonzeros_in_output). When the 
    `consecutive_integer_ids` argument of target column's categorical ColumnType
    object is set to False (as it is by default), UDT creates an internal 
    mapping between target class names and neuron ids. You can map neuron ids back to
    target class names by calling the `class_names()` method. If the `prediction_depth`
    of the model is > 1 and the task is classification then it will return a numpy 
    array of shape `(batch_size, prediction_depth)` which gives the predictions at
    each timestep for each element in the batch.

Examples:
    >>> activations = model.predict_batch([
            {"user_id": "A33225", "timestamp": "2022-12-25", "special_event": "christmas"},
            {"user_id": "A25978", "timestamp": "2022-12-25", "special_event": "christmas"}, 
            {"user_id": "A25978", "timestamp": "2022-12-26", "special_event": "christmas"}"
        ])


Notes: 
    - The values of columns that are tracked temporally may be unknown during inference
      (the column_known_during_inference attribute of the bolt.temporal objects are False
      by default). These columns do not need to be passed into `model.predict_batch()`.
      For example, we did not pass the "movie_title" column to `model.predict_batch()`.
      All other columns must be passed in.
    - If temporal tracking relationships are provided, UDT can make better predictions 
      by taking temporal context into account. For example, UDT may keep track of 
      the last few movies that a user has watched to better recommend the next movie. 
      Thus, UDT is at its best when its internal temporal context gets updated with
      new true samples. `model.predict_batch()` does not update UDT's temporal context.
      To do this without retraining the model, we need to use `model.index()` or 
      `model.index_batch()`. Read about `model.index()` and `model.index_batch()` 
      for details.

)pbdoc";

const char* const UDT_GENERATOR_PREDICT_BATCH = R"pbdoc(
Performs inference on a batch of sammples in parallel.

Args:
    input_queries (List[str]): A list of target queries to be reformulated. 
    top_k (int): The number of candidate query reformulations suggested by the UDT model
        for this input batch. The default value for k is 5. 

Returns:
    List[List[str]]
    Returns a list of k reformulations suggested by the UDT model for each of the given 
    input samples.

Example:
    >>> input_queries = # An arbitrary list of incorrect queries. 
    >>> model = bolt.UniversalDeepTransformer(
            target_column_index=0, 
            source_column_index=1,
            dataset_size="medium"
        )
    >>> model.train(filename="./train_file")
    >>> udt_refomulation_suggestions = model.predict(input_queries=input_queries, top_k=5)
    
)pbdoc";

const char* const UDT_EMBEDDING_REPRESENTATION = R"pbdoc(
Performs inference on a single sample and returns the penultimate layer of 
UniversalDeepTransformer (UDT) so that it can be used as an embedding 
representation for downstream applications.


Args:
    input_sample (Dict[str, str]): The input sample as a dictionary 
        where the keys are column names as specified in data_types and the 
        values are the respective column values. 

Returns: 
    np.ndarray: 
    Returns a numpy array of the penultimate layer's activations.

Examples:
    >>> # Suppose we configure UDT as follows:
    >>> model = bolt.UniversalDeepTransformer(
            data_types={
                "user_id": bolt.types.categorical(),
                "timestamp": bolt.types.date(),
                "special_event": bolt.types.categorical(),
                "movie_title": bolt.types.categorical()
            },
            temporal_tracking_relationships={
                "user_id": ["movie_title"]
            },
            target="movie_title",
            n_target_classes=500,
        )
    >>> # Get an embedding representation
    >>> embedding = model.embedding_representation(
            input_sample={"user_id": "A33225", "timestamp": "2022-12-25", "special_event": "christmas"}
        )

Notes: 
    - The values of columns that are tracked temporally may be unknown during inference
      (the column_known_during_inference attribute of the bolt.temporal objects are False
      by default). These columns do not need to be passed into `model.embedding_representation()`.
      For example, we did not pass the "movie_title" column to `model.embedding_representation()`.
      All other columns must be passed in.
    - If temporal tracking relationships are provided, UDT can make better predictions 
      by taking temporal context into account. For example, UDT may keep track of 
      the last few movies that a user has watched to better recommend the next movie. 
      Thus, UDT is at its best when its internal temporal context gets updated with
      new true samples. `model.predict()` does not update UDT's temporal context.
      To do this without retraining the model, we need to use `model.index()` or 
      `model.index_batch()`. Read about `model.index()` and `model.index_batch()` 
      for details.


)pbdoc";

const char* const UDT_INDEX = R"pbdoc(
Indexes a single true sample to keep UniversalDeepTransformer's (UDT) temporal 
context up to date.

If temporal tracking relationships are provided, UDT can make better predictions 
by taking temporal context into account. For example, UDT may keep track of 
the last few movies that a user has watched to better recommend the next movie. 
Thus, UDT is at its best when its internal temporal context gets updated with
new true samples. `model.index()` does exactly this. 

Args: 
    input_sample (Dict[str, str]): The input sample as a dictionary 
        where the keys are column names as specified in data_types and the "
        values are the respective column values. 

Example:
    >>> # Suppose we configure UDT to do movie recommendation as follows:
    >>> model = bolt.UniversalDeepTransformer(
            data_types={
                "user_id": bolt.types.categorical(),
                "timestamp": bolt.types.date(),
                "special_event": bolt.types.categorical(),
                "movie_title": bolt.types.categorical()
            },
            temporal_tracking_relationships={
                "user_id": ["movie_title"]
            },
            target="movie_title",
            n_target_classes=500,
        )
    >>> # We then deploy the model for inference. Inference is performed by calling model.predict()
    >>> activations = model.predict(
            input_sample={"user_id": "A33225", "timestamp": "2022-12-25", "special_event": "christmas"}
        )
    >>> # Suppose we later learn that user "A33225" ends up watching "Die Hard 3". 
    >>> # We can call model.index() to keep UDT's temporal context up to date.
    >>> model.index(
            input_sample={"user_id": "A33225", "timestamp": "2022-12-25", "special_event": "christmas", "movie_title": "Die Hard 3"}
        )
)pbdoc";

const char* const UDT_INDEX_BATCH = R"pbdoc(
Indexes a batch of true samples to keep UniversalDeepTransformer's (UDT) temporal 
context up to date.

If temporal tracking relationships are provided, UDT can make better predictions 
by taking temporal context into account. For example, UDT may keep track of 
the last few movies that a user has watched to better recommend the next movie. 
Thus, UDT is at its best when its internal temporal context gets updated with
new true samples. `model.index_batch()` does exactly this with a batch of samples. 

Args: 
    input_samples (List[Dict[str, str]]): The input sample as a dictionary 
        where the keys are column names as specified in data_types and the "
        values are the respective column values. 

Example:
    >>> # Suppose we configure UDT to do movie recommendation as follows:
    >>> model = bolt.UniversalDeepTransformer(
            data_types={
                "user_id": bolt.types.categorical(),
                "timestamp": bolt.types.date(),
                "special_event": bolt.types.categorical(),
                "movie_title": bolt.types.categorical()
            },
            temporal_tracking_relationships={
                "user_id": ["movie_title"]
            },
            target="movie_title",
            n_target_classes=500,
        )
    >>> # We then deploy the model for inference. Inference is performed by calling model.predict()
    >>> activations = model.predict(
            input_sample={"user_id": "A33225", "timestamp": "2022-12-25", "special_event": "christmas"}
        )
    >>> # Suppose we later learn what users actually watched.
    >>> # We can call model.index_batch() to keep UDT's temporal context up to date.
    >>> model.index_batch(
            input_samples=[
                {"user_id": "A33225", "timestamp": "2022-12-25", "special_event": "christmas", "movie_title": "Die Hard 3"},
                {"user_id": "A39574", "timestamp": "2022-12-25", "special_event": "christmas", "movie_title": "Home Alone"},
                {"user_id": "A39574", "timestamp": "2022-12-26", "special_event": "christmas", "movie_title": "Home Alone 2"},
            ]
        )
)pbdoc";

const char* const UDT_INDEX_METADATA = R"pbdoc(
Indexes a single column metadata sample.

Args: 
    column_name (str): The name of the column associated with the metadata.
    update (Dict[str, str]): The metadata sample as a dictionary 
        where the keys are column names as specified in the data_types map of 
        the metadata config and the values are the respective column values. This
        map should also contain the metadata key.

Example:
    >>> model = bolt.UniversalDeepTransformer(
            data_types={
                "user_id": bolt.types.categorical(
                    metadata=bolt.types.metadata(
                        filename="user_meta.csv", 
                        data_types={"age": bolt.types.numerical()}, 
                        key_column_name="user_id"
                    )
                )
            },
            target="movie_title",
            n_target_classes=500,
        )
    >>> model.index_metadata(
            column_name="user_id", 
            update={
                "user_id": "XAEA12", # "user_id" column from metadata config's key_column_name
                "age": "2", # "age" column as from metadata config's data_types
            },
        )
)pbdoc";

const char* const UDT_INDEX_METADATA_BATCH = R"pbdoc(
Indexes a batch of column metadata samples.

Args: 
    column_name (str): The name of the column associated with the metadata.
    updates (List[Dict[str, str]]): The metadata samples as a list of dictionaries 
        where the keys are column names as specified in the data_types map of 
        the metadata config and the values are the respective column values. The
        maps should also contain the metadata key.

Example:
    >>> model = bolt.UniversalDeepTransformer(
            data_types={
                "user_id": bolt.types.categorical(
                    metadata=bolt.types.metadata(
                        filename="user_meta.csv", 
                        data_types={"age": bolt.types.numerical()}, 
                        key_column_name="user_id"
                    )
                )
            },
            target="movie_title",
            n_target_classes=500,
        )
    >>> model.index_metadata(
            column_name="user_id", 
            updates=[
                {
                    "user_id": "XAEA12", # "user_id" column from metadata config's key_column_name
                    "age": "2", # "age" column as from metadata config's data_types
                },
                {
                    "user_id": "A22298",
                    "age": "52",
                },
                {
                    "user_id": "B39915",
                    "age": "33",
                },
            ],
        )
)pbdoc";

const char* const UDT_RESET_TEMPORAL_TRACKERS = R"pbdoc(
Resets UniversalDeepTransformer's (UDT) temporal context. When temporal 
relationships are supplied, UDT assumes that we feed it data in chronological 
order. Thus, if we break this assumption, we need to first reset the temporal 
trackers. An example of when you would use this is when you want to repeat the 
UDT training routine on the same dataset. Since you would be training on data 
from the same time period as before, we need to first reset the temporal trackers 
so that we don't double count events.

Args:
    None

Returns:
    None

Example:
    >>> model.reset_temporal_trackers()

)pbdoc";

const char* const UDT_EXPLAIN = R"pbdoc(
Identifies the columns that are most responsible for a predicted outcome 
and provides a brief description of the column's value.

If a target is provided, the model will identify the columns that need 
to change for the model to predict the target class.

Args:
    input_sample (Dict[str, str]): The input sample as a dictionary 
        where the keys are column names as specified in data_types and the "
        values are the respective column values. 
    target_class (str): Optional. The desired target class. If provided, the
        model will identify the columns that need to change for the model to 
        predict the target class.


Returns:
    List[Explanation]:
    A sorted list of `Explanation` objects that each contain the following fields:
    `column_number`, `column_name`, `keyword`, and `percentage_significance`.
    `column_number` and `column_name` identify the responsible column, 
    `keyword` is a brief description of the column value, and
    `percentage_significance` represents this column's contribution to the
    predicted outcome. The list is sorted in descending order by the 
    absolute value of the `percentage_significance` field of each element.
    See `dataset.Explanation` for details.

Example:
    >>> # Suppose we configure UDT as follows:
    >>> model = bolt.UniversalDeepTransformer(
            data_types={
                "user_id": bolt.types.categorical(),
                "timestamp": bolt.types.date(),
                "special_event": bolt.types.categorical(),
                "movie_title": bolt.types.categorical()
            },
            temporal_tracking_relationships={
                "user_id": "movie_title"
            },
            target="movie_title",
            n_target_classes=500,
        )
    >>> # Make a single prediction
    >>> explanations = model.explain(
            input_sample={"user_id": "A33225", "timestamp": "2022-02-02", "special_event": "christmas"}, target_class=35
        )
    >>> print(explanations[0])
    column_number: 0 | column_name: "special_event" | keyword: "christmas" | percentage_significance: 25.2
    >>> print(explanations[1])
    column_number: 1 | column_name: "movie_title" | keyword: "'Die Hard' is one of last 5 values" | percentage_significance: -22.3
    
Notes: 
    - `percentage_significance` can be positive or negative depending on the 
      relationship between the responsible column and the prediction. In the above
      example, the `percentage_significance` associated with the explanation
      "'Die Hard' is one of last 5 values" is negative because recently watching "Die Hard" is 
      negatively correlated with the target class "Home Alone". A large negative value
      is just as "explanatory" as a large positive value.
    - The values of columns that are tracked temporally may be unknown during inference
      (the column_known_during_inference attribute of the bolt.temporal objects are False
      by default). These columns do not need to be passed into `model.explain()`.
      For example, we did not pass the "movie_title" column to `model.explain()`.
      All other columns must be passed in.
    - If temporal tracking relationships are provided, UDT can make better predictions 
      by taking temporal context into account. For example, UDT may keep track of 
      the last few movies that a user has watched to better recommend the next movie. 
      Thus, UDT is at its best when its internal temporal context gets updated with
      new true samples. `model.explain()` does not update UDT's temporal context.
      To do this without retraining the model, we need to use `model.index()` or 
      `model.index_batch()`. Read about `model.index()` and `model.index_batch()` 
      for details.

)pbdoc";

const char* const UDT_SAVE = R"pbdoc(
Serializes an instance of UniversalDeepTransformer (UDT) into a file on disk. 
The serialized UDT includes its current temporal context.

Args:
    filename (str): The file on disk to serialize this instance of UDT into.

Example:
    >>> model.save("udt_savefile.bolt")
)pbdoc";

const char* const UDT_GENERATOR_SAVE = R"pbdoc(
Serializes an instance of UDTGenerator to a file on disk. 

Args:
    filename (str): The file on disk to serialize in which the instance of 
        UDTGenerator is to be serialized. 

Example:
    >>> model.save("udt_savefile.bolt")
)pbdoc";

const char* const UDT_CLASSIFIER_AND_GENERATOR_LOAD = R"pbdoc(
Loads a serialized instance of a UniversalDeepTransformer (UDT) model from a 
file on disk. 

Args:
    filename (str): The file on disk from where to load an instance of UDT.

Returns:
    UniversalDeepTransformer:
    The loaded instance of UDT

Example:
    >>> model = bolt.UniversalDeepTransformer(...)
    >>> model = bolt.UniversalDeepTransformer.load("udt_savefile.bolt")
)pbdoc";

const char* const UDT_CONFIG_INIT = R"pbdoc(
A configuration object for UDT.

UDT is an all-purpose classifier for tabular datasets. In addition to learning from
the columns of a single row, UDT can make use of "temporal context". For 
example, if used to build a movie recommender, UDT may use information 
about the last 5 movies that a user has watched to recommend the next movie.
Similarly, if used to forecast the outcome of marketing campaigns, UDT may 
use several months' worth of campaign history for each product to make better
forecasts.

Args:
    data_types (Dict[str, bolt.types.ColumnType]): A mapping from column name to column type. 
        This map specifies the columns that we want to pass into the model; it does 
        not need to include all columns in the dataset.

        Column type is one of:
        - `bolt.types.categorical()`
        - `bolt.types.numerical(range: tuple(float, float))`
        - `bolt.types.text(average_n_words: float=None)`
        - `bolt.types.date()`
        See bolt.types for details.

        If `temporal_tracking_relationships` is non-empty, there must one 
        bolt.types.date() column. This column contains date strings in YYYY-MM-DD format.
        There can only be one bolt.types.date() column.
    temporal_tracking_relationships (Dict[str, List[str or bolt.temporal.TemporalConfig]]): Optional. 
        A mapping from column name to a list of either other column names or bolt.temporal objects.
        This mapping tells UDT what columns can be tracked over time for each key.
        For example, we may want to tell UDT that we want to track a user's watch 
        history by passing in a map like `{"user_id": ["movie_id"]}`

        If we provide a mapping from a string to a list of strings like the above, 
        the temporal tracking configuration will be autotuned. We can take control by 
        passing in bolt.temporal objects intead of strings.

        bolt.temporal object is one of:
        - `bolt.temporal.categorical(column_name: str, track_last_n: int, column_known_during_inference: bool=False)
        - `bolt.temporal.numerical(column_name: str, history_length: int, column_known_during_inference: bool=False)
        See bolt.temporal for details.
    target (str): Name of the column that contains the value to be predicted by
        UDT. The target column has to be a categorical column.
    time_granularity (str): Optional. Either `"daily"`/`"d"`, `"weekly"`/`"w"`, `"biweekly"`/`"b"`, 
        or `"monthly"`/`"m"`. Interval of time that we are interested in. Temporal numerical 
        features are clubbed according to this time granularity. E.g. if 
        `time_granularity="w"` and the numerical values on days 1 and 2 are
        345.25 and 201.1 respectively, then UDT captures a single numerical 
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
    >>> config = deployment.UDTConfig(
            data_types={
                "product_id": bolt.types.categorical(),
                "timestamp": bolt.types.date(),
                "ad_spend": bolt.types.numerical(range=(0, 10000)),
                "sales_quantity": bolt.types.numerical(range=(0, 20)),
                "sales_performance": bolt.types.categorical(),
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
    >>> # Then we may configure UDT as follows:
    >>> config = deployment.UDTConfig(
            data_types={
                "user_id": bolt.types.categorical(),
                "timestamp": bolt.types.date(),
                "movie_id": bolt.types.categorical(),
                "hours_watched": bolt.types.numerical(range=(0, 25)),
            },
            temporal_tracking_relationships={
                "user_id": [
                    "movie_id", # autotuned movie temporal tracking
                    bolt.temporal.numerical(column_name="hours_watched", history_length="5") # track last 5 days of hours watched.
                ]
            },
            target="movie_id",
            n_target_classes=3000
        )

Notes:
    - Refer to the documentation bolt.types.ColumnType and bolt.temporal.TemporalConfig to better understand column types 
        and temporal tracking configurations.

)pbdoc";

const char* const UDT_CATEGORICAL_METADATA_CONFIG = R"pbdoc(
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
)pbdoc";

const char* const UDT_CATEGORICAL_TEMPORAL = R"pbdoc(
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
)pbdoc";

const char* const UDT_NUMERICAL_TEMPORAL = R"pbdoc(
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
)pbdoc";

const char* const UDT_CATEGORICAL_TYPE = R"pbdoc(
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
)pbdoc";

const char* const UDT_NUMERICAL_TYPE = R"pbdoc(
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
)pbdoc";

const char* const UDT_TEXT_TYPE = R"pbdoc(
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
)pbdoc";

const char* const UDT_DATE_TYPE = R"pbdoc(
Date column type. Use this object if a column contains date strings. 
Date strings must be in YYYY-MM-DD format.

Example:
    >>> deployment.UniversalDeepTransformer(
            data_types: {
                "timestamp": bolt.types.date()
            }
            ...
        )
)pbdoc";

}  // namespace thirdai::automl::python::docs