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
