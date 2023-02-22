from thirdai import bolt


def get_train_and_eval_configs(benchmark_config, callbacks=None):
    learning_rate = benchmark_config.learning_rate
    metrics = benchmark_config.metrics

    train_config = bolt.TrainConfig(epochs=1, learning_rate=learning_rate)

    if callbacks is not None:
        train_config.with_callbacks(callbacks)

    if hasattr(benchmark_config, "reconstruct_hash_functions"):
        train_config.with_reconstruct_hash_functions(
            benchmark_config.reconstruct_hash_functions
        )

    if hasattr(benchmark_config, "rebuild_hash_tables"):
        train_config.with_rebuild_hash_tables(benchmark_config.rebuild_hash_tables)

    eval_config = bolt.EvalConfig().with_metrics(metrics)
    if hasattr(benchmark_config, "compute_roc_auc"):
        eval_config.return_activations()

    return train_config, eval_config


def fix_mlflow_metric_name(original_key):
    # Mlflow can't handle parentheses in metric names.
    # This maps "f_measure(0.95)" to "f_measure_0.95"
    key = original_key.replace("(", "_")
    key = key.replace(")", "")
    key = key.replace("@", "_")
    return key
