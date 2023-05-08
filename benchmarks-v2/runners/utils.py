from thirdai import bolt, bolt_v2


def get_train_config(benchmark_config, learning_rate):
    train_config = bolt.TrainConfig(epochs=1, learning_rate=learning_rate)

    if hasattr(benchmark_config, "reconstruct_hash_functions"):
        train_config.with_reconstruct_hash_functions(
            benchmark_config.reconstruct_hash_functions
        )

    if hasattr(benchmark_config, "rebuild_hash_tables"):
        train_config.with_rebuild_hash_tables(benchmark_config.rebuild_hash_tables)

    return train_config


def get_eval_config(benchmark_config):
    metrics = benchmark_config.metrics

    eval_config = bolt.EvalConfig().with_metrics(metrics)
    if hasattr(benchmark_config, "compute_roc_auc"):
        eval_config.return_activations()

    return eval_config


def get_fc_layer(
    config, input_dim, rebuild_hash_tables, reconstruct_hash_functions, batch_size
):
    return bolt_v2.nn.FullyConnected(
        dim=config["dim"],
        input_dim=input_dim,
        sparsity=config.get("sparsity", 1.0),
        activation=config["activation"],
        sampling=config.get("sampling_config"),
        rebuild_hash_tables=rebuild_hash_tables // batch_size,
        reconstruct_hash_functions=reconstruct_hash_functions // batch_size,
    )
