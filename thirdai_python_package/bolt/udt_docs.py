classifier_train_doc = """
    Trains a UniversalDeepTransformer (UDT) on a given dataset using a file on disk
    or on S3. If the file is on S3, it should be in the normal s3 form, i.e.
    s3://bucket/path/to/key.

    Args:
        filename (str): Path to the dataset file. Can be a path to a file on
            disk or an S3 resource identifier. If the filename is on S3 this
            function will use boto3 internally to load the file (normal boto3
            credential options apply). If multiple files match the bucket and
            prefix, then this will train on all of them.
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
        >>> train_config = bolt.TrainConfig(
                epochs=5, learning_rate=0.01
            ).with_metrics(["mean_squared_error"])
        >>> model.train(
                filename="./train_file", train_config=train_config , max_in_memory_batches=12
            )
        >>> model.train(
                filename="s3://bucket/path/to/key", train_config=train_config
            )

    Notes:
        - If temporal tracking relationships are provided, UDT can make better 
        predictions by taking temporal context into account. For example, UDT may 
        keep track of the last few movies that a user has watched to better 
        recommend the next movie. `model.train()` automatically updates UDT's 
        temporal context.
    """

classifier_eval_doc = """
    Evaluates the UniversalDeepTransformer (UDT) on the given dataset and returns a 
    numpy array of the activations.

    Args:
        filename (str): Path to the dataset file. Like train, this can be a path
            to a local file or a path to an S3 file.
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
        >>> eval_config = bolt.EvalConfig().with_metrics(["categorical_accuracy"])
        >>> activations = model.evaluate(filename="./test_file", eval_config=eval_config)

    Notes: 
        - If temporal tracking relationships are provided, UDT can make better predictions 
            by taking temporal context into account. For example, UDT may keep track of 
            the last few movies that a user has watched to better recommend the next movie.
        `   model.evaluate()` automatically updates UDT's temporal context.
    """
