classifier_train_doc = """
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
        callbacks (List[bolt.callbacks.Callback]): List of callbacks to use during 
            training. 
        metrics (List[str]): List of metrics to compute during training. These are
            logged if logging is enabled, and are accessible by any callbacks. 
        logging_interval (Optional[int]): How frequently to log training metrics,
            represents the number of batches between logging metrics. If not specified 
            logging is done at the end of each epoch. 

    Returns:
        None

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

classifier_eval_doc = """
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
        >>> activations = model.evaluate(filename="./test_file")

    Notes: 
        - If temporal tracking relationships are provided, UDT can make better predictions 
            by taking temporal context into account. For example, UDT may keep track of 
            the last few movies that a user has watched to better recommend the next movie.
        `   model.evaluate()` automatically updates UDT's temporal context.
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
        learning_rate (float): Learning rate used for pretraining. Cold start 
            can be very sensitive to this value. A good default is 0.001.
        epochs (int): Number of epochs to pre-train for.
        metrics (List[str]) = []: List of pre-training metric names to record.
        validation (Optional[bolt.Validation]) = None: Optional validation object. 
            Note that this validation data should have the format as specified in 
            the UDT constructor (ie text and categorical) and not the same format
            passed in to cold start. In the product catalog example this would 
            be query, product_id pairs. 
        callbacks (List[bolt.callbacks.Callback] = []: Callbacks to use during 
            pre-training.

    Returns:
        None.

    Examples:
        >>> model = bolt.UniversalDeepTransformer(
                data_types={
                    "query": bolt.types.text(), 
                    "product": bolt.types.categorical(),
                }
                target="product",
                n_target_classes=1000,
                integer_target=True,
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
