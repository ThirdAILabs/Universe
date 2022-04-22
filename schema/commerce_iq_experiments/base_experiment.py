import sys
sys.path.insert(1, sys.path[0] + "/../../logging/")
from mlflow_logger import DemandForecastingLogger
from thirdai import bolt
# This is where we have the logging, the generic model definition

num_hash_tables = 64
hashes_per_table = 5
sparsity = 0.02
reservoir_size = 128

def write_outputs_to_file(outputs, file):
    for batch in outputs:
        for vec in batch:
            file.write(str(vec[0][1]) + '\n')

def run_experiment(
    dataset,
    hidden_dim,
    get_train_and_test_fn,
    input_dim,
    epochs,
    params={},
    config_uri=None,
    log_results_every_n_epochs=50,
    learning_rate=0.0001,
    error_fn="wmape",
    num_hash_tables=64,
    hashes_per_table=4,
    sparsity=0.1,
    reservoir_size=128,
    ):
    message = ""
    while len(message) < 15:
        message = input("Please enter a message / short description of this run (at least 15 chars): ")
    
    with DemandForecastingLogger(
        run_name=message,
        dataset=dataset,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        metric_name=error_fn,
        num_hash_tables=num_hash_tables,
        hashes_per_table=hashes_per_table,
        sparsity=sparsity,
        reservoir_size=reservoir_size,
        algorithm="bolt",
        feature_config=params
    ) as mlflow_logger:
        if config_uri is not None:
            mlflow_logger.log_script(config_uri)

        train_dataset, test_dataset = get_train_and_test_fn()
        mlflow_logger.log_finished_feature_extraction()

        layers = [
            bolt.LayerConfig(
                dim=hidden_dim, 
                load_factor=sparsity,
                activation_function="ReLU",
                sampling_config=bolt.SamplingConfig(
                    hashes_per_table=hashes_per_table,
                    num_tables=num_hash_tables,
                    range_pow=hashes_per_table * 3,
                    reservoir_size=reservoir_size,
                ),
            ),
            bolt.LayerConfig(dim=1, activation_function="None")
        ]
        

        network = bolt.Network(layers=layers, input_dim=input_dim, loss_function=error_fn, metric=error_fn)

        mlflow_logger.log_start_training()

        last_e_logged = 0
        for e in range(1, epochs + 1):
                
            train_out, train_res = network.train(
            train_dataset,
            learning_rate=learning_rate, 
            epochs=1, 
            rehash=5000, 
            rebuild=15000
            )
            test_out, test_res = network.predict(test_dataset)
            mlflow_logger.log_epoch(train_res[-1].metric, test_res.metric)
            
            if (e - last_e_logged) == log_results_every_n_epochs:
                prefix = "results-" + mlflow_logger.get_run_id() + "-" + str(e)
                with open(prefix + "-train.csv", "w") as f:
                    write_outputs_to_file(train_out, f)
                with open(prefix + "-test.csv", "w") as f:
                    write_outputs_to_file(test_out, f)
                last_e_logged = e
        

