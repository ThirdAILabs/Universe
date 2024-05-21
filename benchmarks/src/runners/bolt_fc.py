from thirdai import bolt

from ..configs.bolt_configs import BoltBenchmarkConfig
from .runner import Runner


class BoltFullyConnectedRunner(Runner):
    config_type = BoltBenchmarkConfig

    def run_benchmark(config: BoltBenchmarkConfig, path_prefix, mlflow_logger):
        model = define_fully_connected_bolt_model(config)
        train_x, train_y, test_x, test_y = config.load_datasets(path_prefix)

        train_data = (
            bolt.train.convert_dataset(train_x, dim=config.input_dim),
            bolt.train.convert_dataset(train_y, dim=config.output_node["dim"]),
        )

        val_data = (
            bolt.train.convert_dataset(test_x, dim=config.input_dim),
            bolt.train.convert_dataset(test_y, dim=config.output_node["dim"]),
        )

        trainer = bolt.train.Trainer(model)

        if isinstance(config.learning_rate, list):
            assert len(config.learning_rate) == config.num_epochs
            learning_rates = config.learning_rate
        else:
            learning_rates = [config.learning_rate] * config.num_epochs

        for epoch, learning_rate in enumerate(learning_rates):
            metrics = trainer.train(
                train_data=train_data,
                learning_rate=learning_rate,
                epochs=1,
                validation_data=val_data,
                validation_metrics=config.metrics,
            )

            if mlflow_logger:
                for k, v in metrics.items():
                    mlflow_logger.log_additional_metric(key=k, value=v[-1], step=epoch)


def define_fully_connected_bolt_model(config: BoltBenchmarkConfig):
    input_layer = bolt.nn.Input(dim=config.input_dim)

    for node_id, hidden_node in enumerate(config.hidden_node):
        if node_id == 0:
            hidden_layer = get_fc_layer(
                config=hidden_node,
                input_dim=config.input_dim,
                rebuild_hash_tables=config.rebuild_hash_tables,
                reconstruct_hash_functions=config.reconstruct_hash_functions,
                batch_size=config.batch_size,
            )(input_layer)
            hidden_layer = bolt.nn.LayerNorm()(hidden_layer)
        else:
            hidden_layer = get_fc_layer(
                config=hidden_node,
                input_dim=config.hidden_node[node_id - 1]["dim"],
                rebuild_hash_tables=config.rebuild_hash_tables,
                reconstruct_hash_functions=config.reconstruct_hash_functions,
                batch_size=config.batch_size,
            )(hidden_layer)
            hidden_layer = bolt.nn.LayerNorm()(hidden_layer)

    output_layer = get_fc_layer(
        config=config.output_node,
        input_dim=config.hidden_node[-1]["dim"],
        rebuild_hash_tables=config.rebuild_hash_tables,
        reconstruct_hash_functions=config.reconstruct_hash_functions,
        batch_size=config.batch_size,
    )(hidden_layer)

    labels = bolt.nn.Input(dim=config.output_node["dim"])

    if config.loss_fn == "CategoricalCrossEntropyLoss":
        loss = bolt.nn.losses.CategoricalCrossEntropy(output_layer, labels)
    elif config.loss_fn == "BinaryCrossEntropyLoss":
        loss = bolt.nn.losses.BinaryCrossEntropy(output_layer, labels)
    else:
        raise ValueError("Invalid loss function in config.")

    model = bolt.nn.Model(
        inputs=[input_layer],
        outputs=[output_layer],
        losses=[loss],
        use_torch_initialization=True,
    )
    model.summary()

    return model


def get_fc_layer(
    config, input_dim, rebuild_hash_tables, reconstruct_hash_functions, batch_size
):
    return bolt.nn.FullyConnected(
        dim=config["dim"],
        input_dim=input_dim,
        sparsity=config.get("sparsity", 1.0),
        activation=config["activation"],
        sampling_config=config.get("sampling_config"),
        rebuild_hash_tables=rebuild_hash_tables // batch_size,
        reconstruct_hash_functions=reconstruct_hash_functions // batch_size,
    )
