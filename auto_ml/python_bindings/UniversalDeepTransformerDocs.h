#pragma once

namespace thirdai::automl::deployment::python::docs {

const char* const UDT_CLASS = R"pbdoc(
UniversalDeepTransformer (UDT) An all-purpose classifier for tabular datasets. 
In addition to learning from the columns of a single row, UDT can make use of 
"temporal context". For example, if used to build a movie recommender, UDT may 
use information about the last 5 movies that a user has watched to recommend the 
next movie. Similarly, if used to forecast the outcome of marketing campaigns, 
UDT may use several months' worth of campaign history for each product to make 
better forecasts.
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

Examples:
    >>> # Suppose each row of our data has the following columns: "product_id", "timestamp", "ad_spend", "sales_quantity", "sales_performance"
    >>> # We want to predict next week's sales performance for each product using temporal context.
    >>> # For each product ID, we would like to track both their ad spend and sales quantity over time.
    >>> model = deployment.UniversalDeepTransformer(
            data_types={
                "product_id": bolt.types.categorical(n_unique_classes=5000),
                "timestamp": bolt.types.date(),
                "ad_spend": bolt.types.numerical(),
                "sales_quantity": bolt.types.numerical(),
                "sales_performance": bolt.types.categorical(n_unique_classes=5),
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
            target="sales_performance"
            time_granularity="weekly",
            lookahead=2 # predict 2 weeks ahead
        )
    >>> # Alternatively suppose our data has the following columns: "user_id", "movie_id", "hours_watched", "timestamp"
    >>> # We want to build a movie recommendation system.
    >>> # Then we may configure UDT as follows:
    >>> model = deployment.UniversalDeepTransformer(
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
)pbdoc";

const char* const UDT_TRAIN = R"pbdoc(
Trains a UniversalDeepTransformer (UDT) on a given dataset using a file on disk.

Args:
    filename (str): Path to the dataset file.
    train_config (bolt.graph.TrainConfig): The training config specifies the number
        of epochs and learning_rate, and optionally allows for specification of a
        validation dataset, metrics, callbacks, and how frequently to log metrics 
        during training. 
    batch_size (Option[int]): This is an optional parameter indicating which batch
        size to use for training. If not specified, the batch size will be autotuned.
    max_in_memory_batches (Option[int]): The maximum number of batches to load in
        memory at a given time. If this is specified then the dataset will be processed
        in a streaming fashion.

Returns:
    None

Examples:
    >>> train_config = bolt.graph.TrainConfig.make(
            epochs=5, learning_rate=0.01
        ).with_metrics(["mean_squared_error"])
    >>> model.train(
            filename="./train_file", train_config=train_config , max_in_memory_batches=12
        )

Notes:
    - If temporal tracking relationships are provided, UDT can make better 
      predictions by taking temporal context into account. For example, UDT may 
      keep track of the last few movies that a user has watched to better 
      recommend the next movie. `model.train()` automatically updates UDT's 
      temporal context.
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

const char* const UDT_EVALUATE = R"pbdoc(
Evaluates the UniversalDeepTransformer (UDT) on the given dataset and returns a 
numpy array of the activations.

Args:
    filename (str): Path to the dataset file.
    predict_config (Option[bolt.graph.PredictConfig]): The predict config is optional.
        It specifies metrics to compute and whether to use sparse
        inference.

Returns:
    (np.ndarray or Tuple[np.ndarray, np.ndarray]): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (dataset_length, num_nonzeros_in_output). When the 
    `consecutive_integer_ids` argument of target column's categorical ColumnType
    object is set to False (as it is by default), UDT creates an internal 
    mapping between target class names and neuron ids. You can map neuron ids back to
    target class names by calling the `class_names()` method.

Examples:
    >>> predict_config = bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"])
    >>> activations = model.evaluate(filename="./test_file", predict_config=predict_config)

Notes: 
    - If temporal tracking relationships are provided, UDT can make better predictions 
      by taking temporal context into account. For example, UDT may keep track of 
      the last few movies that a user has watched to better recommend the next movie.
      `model.evaluate()` automatically updates UDT's temporal context.
)pbdoc";

const char* const UDT_PREDICT = R"pbdoc(
Performs inference on a single sample.

Args:
    input_sample (Dict[str, str]): The input sample as a dictionary 
        where the keys are column names and the values are the respective column values. 
    use_sparse_inference (bool, default=False): Whether or not to use sparse inference.

Returns: 
    (np.ndarray or Tuple[np.ndarray, np.ndarray]): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (num_nonzeros_in_output, ). When the 
    `consecutive_integer_ids` argument of target column's categorical ColumnType
    object is set to False (as it is by default), UDT creates an internal 
    mapping between target class names and neuron ids. You can map neuron ids back to
    target class names by calling the `class_names()` method.

Examples:
    >>> # Suppose we configure UDT as follows:
    >>> model = deployment.UniversalDeepTransformer(
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
Performs inference on a batch of samples samples in parallel.

Args:
    input_samples (List[Dict[str, str]]): A list of input sample as dictionaries 
        where the keys are column names as specified in data_types and the 
        values are the respective column values. 
    use_sparse_inference (bool, default=False): Whether or not to use sparse inference.

Returns: 
    (np.ndarray or Tuple[np.ndarray, np.ndarray]): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (batch_size, num_nonzeros_in_output). When the 
    `consecutive_integer_ids` argument of target column's categorical ColumnType
    object is set to False (as it is by default), UDT creates an internal 
    mapping between target class names and neuron ids. You can map neuron ids back to
    target class names by calling the `class_names()` method.

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
    >>> model = deployment.UniversalDeepTransformer(
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
    >>> model = deployment.UniversalDeepTransformer(
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
    input_samples (ListDict[str, str]): The input sample as a dictionary 
        where the keys are column names as specified in data_types and the "
        values are the respective column values. 

Example:
    >>> # Suppose we configure UDT to do movie recommendation as follows:
    >>> model = deployment.UniversalDeepTransformer(
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
    target (str): Optional. The desired target class. If provided, the
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
    >>> model = deployment.UniversalDeepTransformer(
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
    >>> # Make a single prediction
    >>> explanations = model.explain(
            input_sample={"user_id": "A33225", "timestamp": "2022-02-02", "special_event": "christmas"}, target=35
        )
    >>> print(explanations[0])
    column_number: 0 | column_name: "special_event" | keyword: "christmas" | percentage_significance: 25.2
    >>> print(explanations[1])
    column_number: 1 | column_name: "movie_title" | keyword: "Previously seen 'Die Hard'" | percentage_significance: -22.3
    
Notes: 
    - `percentage_significance` can be positive or negative depending on the 
      relationship between the responsible column and the prediction. In the above
      example, the `percentage_significance` associated with the explanation
      "Previously seen 'Die Hard'" is negative because recently watching "Die Hard" is 
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

const char* const UDT_LOAD = R"pbdoc(
Loads a serialized instance of UniversalDeepTransformer (UDT) from a file on 
disk. The loaded UDT includes the temporal context from before serialization.

Args:
    filename (str): The file on disk to load the instance of UDT from.

Returns:
    UniversalDeepTransformer: 
    The loaded instance of UDT.

Example:
    >>> model = deployment.UniversalDeepTransformer(...)
    >>> model = deployment.UniversalDeepTransformer.load("udt_savefile.bolt")
)pbdoc";

}  // namespace thirdai::automl::deployment::python::docs
