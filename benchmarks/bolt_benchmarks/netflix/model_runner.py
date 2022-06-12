from mlflow_logger import FlexibleExperimentLogger
from thirdai import bolt
from thirdai.dataset import DataPipeline
import time

RMSE = "root_mean_squared_error"

def format_metrics(train_metrics, test_metrics):
    return {
        "train_rmse": train_metrics[RMSE][-1],
        "test_rmse": test_metrics[RMSE]
    }

def run_model(
    train_x,
    train_y,
    test_x,
    test_y,
    input_dim,
    batch_size,
    feature_description,
    data_pipeline_script_location,
    train_data_load_time,
    test_data_load_time,
):
    hidden_dim = 5000
    hidden_sparsity = 0.02
    learning_rate = 0.0001
    rehash = 6400
    rebuild = 128000

    with FlexibleExperimentLogger(
        experiment_name="Netflix Rating Prediction", 
        dataset="Netflix", 
        experiment_args={
            "feature_description": feature_description,
            "data_pipeline_script_location": data_pipeline_script_location,
            "hidden_dim": hidden_dim, 
            "hidden_sparsity": hidden_sparsity,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "input_dim": input_dim,
            "rehash": rehash,
            "rebuild": rebuild,
            "train_data_load_time": train_data_load_time,
            "test_data_load_time": test_data_load_time,
        }) as logger:
        layers = [
            bolt.FullyConnected(
                dim=hidden_dim,
                sparsity=hidden_sparsity,
                activation_function=bolt.ActivationFunctions.ReLU,
            ),
            bolt.FullyConnected(
                dim=1, activation_function=bolt.ActivationFunctions.Linear
            ),
        ]

        network = bolt.Network(layers=layers, input_dim=input_dim)

        epochs = 20
        last_metrics = {}

        logger.log_start_training(last_metrics)

        for i in range(epochs):
            train_metrics = network.train(
                train_data=train_x, 
                train_labels=train_y, 
                loss_fn=bolt.MeanSquaredError(), 
                learning_rate=learning_rate, 
                epochs=1, 
                rehash=rehash, 
                rebuild=rebuild, 
                metrics=[RMSE])
            test_metrics, _ = network.predict(
                test_data=test_x,
                test_labels=test_y,
                batch_size=batch_size,
                metrics=[RMSE],
                verbose=True,
            )
            last_metrics = format_metrics(train_metrics, test_metrics)
            logger.log_epoch(last_metrics)
        
        logger.log_final_metrics(last_metrics)


def make_pipeline(filename, input_blocks, label_blocks, batch_size):
    return DataPipeline(
        filename=filename, 
        input_blocks=input_blocks,
        label_blocks=label_blocks,
        batch_size=batch_size)

def run_experiment(input_blocks, label_blocks):
    batch_size = 2048
    train_pipeline = make_pipeline("/share/data/netflix/date_sorted_data_train.csv", input_blocks, label_blocks, batch_size)
    test_pipeline = make_pipeline("/share/data/netflix/date_sorted_data_test.csv", input_blocks, label_blocks, batch_size)

    print("Loading train data...")
    train_load_start = time.time()
    train_x, train_y = train_pipeline.load_in_memory()
    train_load_end = time.time()
    print("Loaded train data in", train_load_end - train_load_start, "seconds.")

    print("Loading test data...")
    test_load_start = time.time()
    test_x, test_y = test_pipeline.load_in_memory()
    test_load_end = time.time()
    print("Loaded test data in", test_load_end - test_load_start, "seconds.")

    print("DataPipeline produced a dataset with", train_pipeline.get_input_dim(), "dimensional vectors.")

    run_model(
        train_x,
        train_y,
        test_x,
        test_y,
        train_pipeline.get_input_dim(),
        batch_size=batch_size,
        feature_description="baseline",
        data_pipeline_script_location=__file__,
        train_data_load_time=train_load_end - train_load_start,
        test_data_load_time=test_load_end - test_load_start,
    )