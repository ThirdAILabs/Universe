from thirdai import bolt


def add_sparsity_to_first_non_sparse_layer(
    model, start_layer=1, experimental_autotune=False
):
    try:
        layer = model.__getitem__(f"fc_{start_layer}")
        if layer.get_sparsity() == 1:
            autotune_sparsity_for_layer(layer, experimental_autotune)
            return start_layer
        else:
            add_sparsity_to_first_non_sparse_layer(
                model,
                start_layer=start_layer + 1,
                experimental_autotune=experimental_autotune,
            )

    except Exception as e:
        print(f"An error occurred: {e}")


def autotune_sparsity_for_layer(layer, experimental_autotune):

    dim = layer.dim()

    if dim <= 512:
        sparsity = 0.2
    elif dim <= 2048:
        sparsity = 0.1
    else:
        sparsity = 0.05

    add_sparsity_to_layer(layer, experimental_autotune, sparsity)


def add_sparsity_to_layer(layer, experimental_autotune, sparsity):
    layer.set_sparsity(
        sparsity, rebuild_tables=True, experimental_autotune=experimental_autotune
    )
    

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
