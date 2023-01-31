class BenchmarkConfig:
    learning_rate = 0.01
    num_epochs = 5
    target = "label"
    n_target_classes = 2
    delimiter = ","
    metric_type = "categorical_accuracy"
    model_config = None
    model_config_path = None
    callbacks = []