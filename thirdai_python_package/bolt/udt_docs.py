udt_train_doc = """
    Trains a UniversalDeepTransformer (UDT) on a given dataset using a file on disk
    or in a cloud storage bucket, such as s3 or google cloud storage (GCS). If the
    file is on S3, it should be in the normal s3 form, i.e. s3://bucket/path/to/key.
    For files in GCS, the path should have the form gcs://bucket/path/to/filename.
    We currently support csv and parquet format files. If the file is parquet, it 
    should end in .parquet or .pqt. Otherwise, we will assume it is a csv file.

    Args:
        filename (str): Path to the dataset file. It Can be a path to a file on
            disk or an S3 or GCS resource identifier. If the file is on s3 or GCS,
            regular credentials files will be required for authentication. 
        learning_rate (float): Optional, uses default if not provided.
        epochs (int): Optional, uses default if not provided.
        validation (Optional[bolt.Validation]): This is an optional parameter that 
            specifies a validation dataset, metrics, and interval to use during 
            training.
        batch_size (Option[int]): This is an optional parameter indicating which batch
            size to use for training. If not specified, the batch size will be autotuned.
        max_in_memory_batches (Option[int]): The maximum number of batches to load in
            memory at a given time. If this is specified then the dataset will be processed
            in a streaming fashion.
        verbose (bool): Optional, defaults to True. Controls if additional information 
            is printed during training.
        callbacks (List[bolt.train.callbacks.Callback]): List of callbacks to use during 
            training. 
        metrics (List[str]): List of metrics to compute during training. These are
            logged if logging is enabled, and are accessible by any callbacks. 
        logging_interval (Optional[int]): How frequently to log training metrics,
            represents the number of batches between logging metrics. If not specified 
            logging is done at the end of each epoch. 

    Returns:
        (Dict[str, List[float]]): 
        The train method returns a dictionary providing the values of any metrics 
        computed during training. The format is: {"name of metric": [list of values]}.

    Examples:
        >>> model.train(
                filename="./train_file", epochs=5, learning_rate=0.01, max_in_memory_batches=12
            )
        >>> model.train(
                filename="s3://bucket/path/to/key"
            )

    Notes:
        - If temporal tracking relationships are provided, UDT can make better 
          predictions by taking temporal context into account. For example, UDT may 
          keep track of the last few movies that a user has watched to better 
          recommend the next movie. `model.train()` automatically updates UDT's 
          temporal context.
        - If the prediction task is binary classification then the model will attempt 
          to find an optimal threshold for predictions that will be used if `return_predicted_class=True`
          is passed to calls to evaluate, predict, and predict_batch. The optimal threshold
          will be selected based on what threshold maximizes the first validation metric
          on the validation data. If no validation data or metrics are passed in then 
          it will use the first 100 batches of the training data and the first training
          metric. If there is also no training metrics then it will not choose a prediction
          threshold. 
"""

udt_train_on_datasource_doc = """
Same as train except for arg `filename` is replaced by an arg `data_source` which accepts an DataSource object.
"""

udt_eval_doc = """
    Evaluates the UniversalDeepTransformer (UDT) on the given dataset and returns a 
    numpy array of the activations. We currently support csv and parquet format 
    files. If the file is parquet, it should end in .parquet or .pqt. Otherwise, 
    we will assume it is a csv file.

    Args:
        filename (str): Path to the dataset file. Like train, this can be a path
            to a local file or a path to a file that lives in an s3 or google cloud
            storage (GCS) bucket. 
        metrics (List[str]): List of metrics to compute during evaluation. 
        use_sparse_inference (bool): Optional, defaults to False, determines if 
            sparse inference is used during evaluation. 
        verbose (bool): Optional, defaults to True. Controls if additional information 
            is printed during training.
        top_k (Optional[int]): Optional, defaults to None. This parameter is only used 
            for query reformulation model to deterimine how many candidates to select
            before computing evaluation metrics.

    Returns:
        (Dict[str, float]): 
        Returns a list of values for the specified metrics, keyed by the metric names.

    Examples:
        >>> metrics = model.evaluate(filename="./test_file", metrics=["categorical_accuracy"])

    Notes: 
        - If temporal tracking relationships are provided, UDT can make better predictions 
          by taking temporal context into account. For example, UDT may keep track of 
          the last few movies that a user has watched to better recommend the next movie.
          `model.evaluate()` automatically updates UDT's temporal context.
 """

udt_eval_on_data_source_doc = """
Same as evaluate except for arg `filename` is replaced by an arg `data_source` which accepts an DataSource object.
"""

udt_cold_start_doc = """
    This method will perform cold start pretraining for UDT. This is a type of 
    pretraining for text classification models that is especially useful for query 
    to product recommendation models. It requires that the model takes in a single 
    text input and has a categorical/multi-categorical output.

    The cold start pretraining typically takes in an unsupervised dataset of objects
    where each object corresponds to one or more columns of textual metadata. This could 
    be something like a product catalog (with product ids as objects, and titles, 
    descriptions, and tags as metadata). The goal with cold start is to pre-train UDT
    on unsupervised data so in the future it may be able to answer text search queries 
    and return the relevant objects. The dataset it takes in should be a csv file that
    gives a class id column and some number of text columns, where for a given row 
    the text is related to the class also specified by that row.

    You may cold_start the model and train with supervised data afterwards, typically
    leading to faster convergence on the supervised data.

    Args:
        filename (str): Path to the dataset used for pretraining.
        strong_column_names (List[str]): The strong column names indicate which 
            text columns are most closely related to the output class. In this 
            case closely related means that all of the words in the text are useful
            in identifying the output class in that row. For example in the 
            case of a product catalog then a strong column could be the full title 
            of the product.
        weak_column_names (List[str]): The weak column names indicate which text 
            columns are either more loosely related to the output class. In 
            this case loosely related means that parts of the text are useful in 
            identifying the output class, but there may also be parts of the 
            text that contain more generic words or phrases that don't have as high 
            of a correlation. For example in a product catalog the description of
            the product could be a weak column because while there is a correlation,
            parts of the description may be fairly similar between products or be
            too general to completly identify which products the correspond to.
        learning_rate (float): Optional, uses default if not provided.
        epochs (int): Optional, uses default if not provided.
        batch_size (Option[int]): This is an optional parameter indicating which batch
            size to use for training. If not specified, the batch size will be autotuned.
        metrics (List[str]): List of metrics to compute during training. These are
            logged if logging is enabled, and are accessible by any callbacks. 
        validation (Optional[bolt.Validation]): This is an optional parameter that 
            specifies a validation dataset, metrics, and interval to use during 
            training.
        callbacks (List[bolt.train.callbacks.Callback]): List of callbacks to use during 
            training. 
        max_in_memory_batches (Option[int]): The maximum number of batches to load in
            memory at a given time. If this is specified then the dataset will be processed
            in a streaming fashion.
        verbose (bool): Optional, defaults to True. Controls if additional information 
            is printed during training.
        logging_interval (Optional[int]): How frequently to log training metrics,
            represents the number of batches between logging metrics. If not specified 
            logging is done at the end of each epoch. 

    Returns:
        (Dict[str, List[float]]): 
        The train method returns a dictionary providing the values of any metrics 
        computed during training. The format is: {"name of metric": [list of values]}.

    Examples:
        >>> model = bolt.UniversalDeepTransformer(
                data_types={
                    "query": bolt.types.text(), 
                    "product": bolt.types.categorical(n_classes=1000),
                }
                target="product",
            )
        >>> model.cold_start(
                filename="product_catalog.csv",
                strong_column_names=["title"],
                weak_column_names=["description", "bullet_points"],
                learning_rate=0.001,
                epochs=5,
                metrics=["f_measure(0.95)"]
            )
        >>> model.train(
                train_filename=supervised_query_product_data,
            )
        >>> result = model.predict({"QUERY": query})

"""

udt_cold_start_on_data_source_doc = """
Same as evaluate except for arg `filename` is replaced by an arg `data_source` which accepts an DataSource object.
"""
