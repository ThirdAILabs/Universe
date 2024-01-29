#pragma once

namespace thirdai::automl::python::docs {

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

const char* const UDT_QUERY_REFORMULATION_INIT = R"pbdoc(
UniversalDeepTransformer (UDT) Constructor. 

Args:
    source_column (str): Optional. Column name specifying the source queries in the input 
        dataset. If provided then the model can use these queries to augment its training.
        If not provided then the model be trained from the target queries directly. If the 
        source column is specified the the model can be trained with in both a supervised 
        setting where (incorrect query, correct query) pairs are provided and in an 
        unsupervised setting where only correct queries are provided. If source is not specified
        then it can only be trained in an unsupervised setting.
    target_column (str): Column name specifying the target queries in the input dataset. 
        Queries in this column are the target that the UDT model learns to predict. 
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

const char* const UDT_TRAIN_BATCH = R"pbdoc(
Trains the model on the given training batch. 

Args:
    batch (List[Dict[str, str]]): The raw data comprising the training batch. This should 
        be in the form {"column_name": "column_value"} for each column the model expects.
    learning_rate (float): Optional, uses default if not provided.

Returns: 
    None
)pbdoc";

const char* const UDT_EVALUATE = R"pbdoc(
Evaluates the model on the given dataset using the provided metrics. 

Args:
    
)pbdoc";

const char* const UDT_PREDICT = R"pbdoc(
Performs inference on a single sample.

Args:
    input_sample (Dict[str, str]): The input sample as a dictionary 
        where the keys are column names and the values are the respective column values. 
    use_sparse_inference (bool) = False: Whether or not to use sparse inference.
    return_predicted_class (bool) = False: If true then the model will return the id of the 
        predicted class instead of the activations of the output layer. This argument is only 
        applicable to classification models. 
    top_k (Optional[int]) = None: If specified then the model will return the ids of the 
        top k predicted classes instead of the activations of the output layer. This argument
        is only applicable to classification models. 

Returns: 
    (np.ndarray, Tuple[np.ndarray, np.ndarray], List[int], or int): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (num_nonzeros_in_output, ). If return predicted class is specified
    then the class id (an integer) will be returned. If top_k is specified then a list of 
    integer class ids will be returned. You can map neuron ids back to target class names 
    by calling the `class_name()` method. If the target column is a sequence, UDT will 
    perform inference recursively and return a sequence in the same format as the target 
    column.

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

const char* const UDT_PREDICT_BATCH = R"pbdoc(
Performs inference on a batch of samples in parallel.

Args:
    input_samples (List[Dict[str, str]]): A list of input sample as dictionaries 
        where the keys are column names as specified in data_types and the 
        values are the respective column values. 
    use_sparse_inference (bool, default=False): Whether or not to use sparse inference.
    return_predicted_class (bool) = False: If true then the model will return the id of the 
        predicted class instead of the activations of the output layer. This argument is only 
        applicable to classification models. 
    top_k (Optional[int]) = None: If specified then the model will return the ids of the 
        top k predicted classes instead of the activations of the output layer. This argument
        is only applicable to classification models. 

Returns: 
    (np.ndarray, Tuple[np.ndarray, np.ndarray], List[List[int]], or List[int]): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (batch_size, num_nonzeros_in_output). If return predicted class is specified
    then the class id (an integer) will be returned. If top_k is specified then a list of 
    integer class ids will be returned. You can map neuron ids back to target class names 
    by calling the `class_name()` method. If the target column is a sequence, UDT will 
    perform inference recursively and return a sequence in the same format as the target 
    column.

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

const char* const UDT_ENTITY_EMBEDDING = R"pbdoc(
Returns an embedding representation for a given output entity, an entity being 
the name of a class predicted as output.

Args:
    label_id (Union[int, str]): The the name of the entity to get an embedding for.
    If integer_target=True, this function should take in an integer from 0 to 
    n_target_classes - 1 instead of a string.

Returns:
    A 1D numpy array of floats representing a dense embedding of that entity.
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
    List[Tuple[str, float]]:
    A list of explanations from the input features along with weights representing the 
    significance of that feature.

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

)pbdoc";

const char* const UDT_SAVE_CHECKPOINT = R"pbdoc(
Serializes an instance of UniversalDeepTransformer (UDT) into a file on disk. 
The serialized UDT includes its current temporal context. The `save` method just saves 
the model parameters, the `checkpoint` method saves additional information such as 
the optimizer state to use if training is resumed.

Args:
    filename (str): The file on disk to serialize this instance of UDT into.

Example:
    >>> model.save("udt_savefile.bolt")
    >>> model.checkpoint("udt_savefile.bolt")
)pbdoc";

const char* const UDT_LOAD = R"pbdoc(
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

const char* const UDT_INDEX_NODES = R"pbdoc(
Updates the graph that the UDT model is performing graph node classification on. The file 
should have the same node id, neighbors, and features columns as the model is configured to accept.

Args:
    filename (str): The filename to load the graph from.

Returns:
    None

)pbdoc";

const char* const UDT_CLEAR_GRAPH = R"pbdoc(
Clears all graph info that is being tracked by the model.

Returns:
    None

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
    >>> model = bolt.UniversalDeepTransformer(
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
    >>> model = bolt.UniversalDeepTransformer(
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
    >>> bolt.UniversalDeepTransformer(
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
    >>> bolt.UniversalDeepTransformer(
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
    tokenizer (str): Optional. Either "words", "words-punct" or 
        "char-k" (k is a number, e.g. "char-5"). Defaults to "words". 
    contextual_encoding (str): Optional. Either "local", "global", "ngram-N", or
        "none", defaults to "none". 

Example:
    >>> bolt.UniversalDeepTransformer(
            data_types: {
                "user_motto": bolt.types.text(),
                "user_bio": bolt.types.text(contextual_encoding="local")
            }
            ...
        )
)pbdoc";

const char* const UDT_DATE_TYPE = R"pbdoc(
Date column type. Use this object if a column contains date strings. 
Date strings must be in YYYY-MM-DD format.

Example:
    >>> bolt.UniversalDeepTransformer(
            data_types: {
                "timestamp": bolt.types.date()
            }
            ...
        )
)pbdoc";

const char* const UDT_SEQUENCE_TYPE = R"pbdoc(
 Sequence column type. Use this object if a column contains an ordered sequence 
 of strings delimited by a character. The delimiter must be different than the 
 delimiter between columns.

 When the target column is a sequence type, then UDT will perform inferences 
 recursively.

 Args:
     delimiter (str): Optional. The sequence delimiter. Defaults to " ".
     max_length (int): Required if the column is the target. The maximum length 
         of the sequence. If UDT sees longer sequences, elements beyond the provided
         upper bound will be ignored.

 Example:
     >>> bolt.UniversalDeepTransformer(
             data_types: {
                 "input_sequence": bolt.types.sequence(delimiter='\t')
                 "output_sequence": bolt.types.sequence(max_length=30) # max_length must be provided for target sequence.
             },
             target="output_sequence",
             n_target_classes=26
             ...
         )
 )pbdoc";

}  // namespace thirdai::automl::python::docs