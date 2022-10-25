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
        - `bolt.types.categorical(n_unique_values: int, delimiter: str=None, consecutive_integer_ids: bool=False)`
        - `bolt.types.numerical()`
        - `bolt.types.text(average_n_words: int=None, embedding_size: str="m", use_attention: bool=False)`
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
        size to use for training. If not specified the default batch size from the 
        TrainEvalParameters is used.
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
    - `model.train()` resets UDT's temporal context at the start of training to 
      prevent unwanted information from leaking into the training routine.
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
    predict_config (Option[bolt.graph.PredictConfig]): The predict config is optional
        and allows for specification of metrics to compute and whether to use sparse
        inference.

Returns:
    (np.ndarray or Tuple[np.ndarray, np.ndarray]): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (dataset_length, num_nonzeros_in_output). When the 
    `consecutive_integer_ids` argument of target column's categorical ColumnType
    object is set to False (as it is by default), UDT creates an internal 
    mapping between target class names and neuron ids. This map is accessible 
    by calling the get_neuron_id_to_label_map() method.

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
        where the keys are column names as specified in data_types and the 
        values are the respective column values. 
    use_sparse_inference (bool, default=False): Whether or not to use sparse inference.

Returns: 
    (np.ndarray or Tuple[np.ndarray, np.ndarray]): 
    Returns a numpy array of the activations if the output is dense, or a tuple 
    of the active neurons and activations if the output is sparse. The shape of 
    each array will be (num_nonzeros_in_output, ). When the 
    `consecutive_integer_ids` argument of target column's categorical ColumnType
    object is set to False (as it is by default), UDT creates an internal 
    mapping between target class names and neuron ids. This map is accessible 
    by calling the get_neuron_id_to_label_map() method.

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
    - Only columns that are known at the time of inference need to be passed to
      `model.predict()`. For example, notice that while we have a "movie_title" 
      column in the `data_types` argument, we did not pass it to `model.predict()`. 
      This is because we do not know the movie title at the time of inference – that 
      is the target that we are trying to predict after all.
    - If temporal tracking relationships are provided, UDT can make better predictions 
      by taking temporal context into account. For example, UDT may keep track of 
      the last few movies that a user has watched to better recommend the next movie. 
      Thus, UDT is at its best when its internal temporal context gets updated with
      new true samples. `model.predict()` does not update UDT's temporal context.
      To do this, we need to use `model.index()` or `model.index_batch()`. Read 
      about `model.index()` and `model.index_batch()` for details.

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
    mapping between target class names and neuron ids. This map is accessible 
    by calling the get_neuron_id_to_label_map() method.

Examples:
    >>> activations = model.predict_batch([
            {"user_id": "A33225", "timestamp": "2022-12-25", "special_event": "christmas"},
            {"user_id": "A25978", "timestamp": "2022-12-25", "special_event": "christmas"}, 
            {"user_id": "A25978", "timestamp": "2022-12-26", "special_event": "christmas"}"
        ])


Notes: 
    - Only columns that are known at the time of inference need to be passed to
      `model.predict_batch()`. For example, notice that while we have a "movie_title" 
      column in the `data_types` argument, we did not pass it to `model.predict()`. 
      This is because we do not know the movie title at the time of inference – that 
      is the target that we are trying to predict after all.
    - If temporal tracking relationships are provided, UDT can make better predictions 
      by taking temporal context into account. For example, UDT may keep track of 
      the last few movies that a user has watched to better recommend the next movie. 
      Thus, UDT is at its best when its internal temporal context gets updated with
      new true samples. `model.predict_batch()` does not update UDT's temporal context.
      To do this, we need to use `model.index()` or `model.index_batch()`. Read 
      about `model.index()` and `model.index_batch()` for details.

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
    - Only columns that are known at the time of inference need to be passed to
      `model.predict()`. For example, notice that while we have a "movie_title" 
      column in the `data_types` argument, we did not pass it to `model.predict()`. 
      This is because we do not know the movie title at the time of inference – that 
      is the target that we are trying to predict after all.
    - If temporal tracking relationships are provided, UDT can make better predictions 
      by taking temporal context into account. For example, UDT may keep track of 
      the last few movies that a user has watched to better recommend the next movie. 
      Thus, UDT is at its best when its internal temporal context gets updated with
      new true samples. `model.predict()` does not update UDT's temporal context.
      To do this, we need to use `model.index()` or `model.index_batch()`. Read 
      about `model.index()` and `model.index_batch()` for details.


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
    `keyword` is a brief description of the value in this column, and
    `percentage_significance` represents this column's contribution to the
    predicted outcome. The list is sorted in descending order by the 
    absolute value of the `percentage_significance` field of each element.

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
    - If temporal tracking relationships are provided, UDT can make better predictions 
      by taking temporal context into account. For example, UDT may keep track of 
      the last few movies that a user has watched to better recommend the next movie. 
      Thus, UDT is at its best when its internal temporal context gets updated with
      new true samples. `model.explain()` does not update UDT's temporal context.
      To do this, we need to use `model.index()` or `model.index_batch()`. Read 
      about `model.index()` and `model.index_batch()` for details.

)pbdoc";

const char* const UDT_SAVE = R"pbdoc(
Serializes an instance of UniversalDeepTransformer (UDT) into a file on disk. 
The serialized UDT includes its current temporal context.

Args:
    filename (str): The file on disk to serialize this instance of UDT into.

Example:
    >>> model = deployment.UniversalDeepTransformer(...)
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
