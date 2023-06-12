from thirdai import bolt, bolt_v2

from ..configs.bolt_configs import BoltBenchmarkConfig
from .runner import Runner
from .utils import get_train_config


class BoltFullyConnectedRunner(Runner):
    config_type = BoltBenchmarkConfig

    def run_benchmark(config: BoltBenchmarkConfig, path_prefix, mlflow_logger):
        model = define_fully_connected_bolt_model(config)
        train_set, train_labels, test_set, test_labels = config.load_datasets(
            path_prefix
        )

        if isinstance(config.learning_rate, list):
            assert len(config.learning_rate) == config.num_epochs
            learning_rates = config.learning_rate
        else:
            learning_rates = [config.learning_rate] * config.num_epochs

        for epoch, learning_rate in enumerate(learning_rates):
            train_metrics = model.train(
                train_data=train_set,
                train_labels=train_labels,
                train_config=get_train_config(config, learning_rate),
            )

            if mlflow_logger:
                for k, v in train_metrics.items():
                    mlflow_logger.log_additional_metric(key=k, value=v[0], step=epoch)

            predict_output = model.evaluate(
                test_data=test_set,
                test_labels=test_labels,
                eval_config=bolt.EvalConfig().with_metrics(config.metrics),
            )

            if mlflow_logger:
                for k, v in predict_output[0].items():
                    mlflow_logger.log_additional_metric(
                        key="val_" + k, value=v, step=epoch
                    )


def define_fully_connected_bolt_model(config: BoltBenchmarkConfig):
    input_node = bolt.nn.Input(dim=config.input_dim)
    hidden_node = bolt.nn.FullyConnected(**config.hidden_node)(input_node)
    output_node = bolt.nn.FullyConnected(**config.output_node)(hidden_node)

    model = bolt.nn.Model(inputs=[input_node], output=output_node)
    model.compile(
        loss=bolt.nn.losses.get_loss_function(name=config.loss_fn),
        print_when_done=False,
    )
    model.summary(detailed=True)

    return model


class BoltV2FullyConnectedRunner(Runner):
    config_type = BoltBenchmarkConfig

    def run_benchmark(config: BoltBenchmarkConfig, path_prefix, mlflow_logger):
        model = define_fully_connected_bolt_v2_model(config)
        train_x, train_y, test_x, test_y = config.load_datasets(path_prefix)

        train_data = (
            bolt_v2.train.convert_dataset(train_x, dim=config.input_dim),
            bolt_v2.train.convert_dataset(train_y, dim=config.output_node["dim"]),
        )

        val_data = (
            bolt_v2.train.convert_dataset(test_x, dim=config.input_dim),
            bolt_v2.train.convert_dataset(test_y, dim=config.output_node["dim"]),
        )

        trainer = bolt_v2.train.Trainer(model)

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


def define_fully_connected_bolt_v2_model(config: BoltBenchmarkConfig):
    input_layer = bolt_v2.nn.Input(dim=config.input_dim)

    hidden_layer = get_fc_layer(
        config=config.hidden_node,
        input_dim=config.input_dim,
        rebuild_hash_tables=config.rebuild_hash_tables,
        reconstruct_hash_functions=config.reconstruct_hash_functions,
        batch_size=config.batch_size,
    )(input_layer)

    output_layer = get_fc_layer(
        config=config.output_node,
        input_dim=config.hidden_node["dim"],
        rebuild_hash_tables=config.rebuild_hash_tables,
        reconstruct_hash_functions=config.reconstruct_hash_functions,
        batch_size=config.batch_size,
    )(hidden_layer)

    labels = bolt_v2.nn.Input(dim=config.output_node["dim"])

    if config.loss_fn == "CategoricalCrossEntropyLoss":
        loss = bolt_v2.nn.losses.CategoricalCrossEntropy(output_layer, labels)
    elif config.loss_fn == "BinaryCrossEntropyLoss":
        loss = bolt_v2.nn.losses.BinaryCrossEntropy(output_layer, labels)
    else:
        raise ValueError("Invalid loss function in config.")

    model = bolt_v2.nn.Model(
        inputs=[input_layer], outputs=[output_layer], losses=[loss]
    )
    model.summary()

    return model


def get_fc_layer(
    config, input_dim, rebuild_hash_tables, reconstruct_hash_functions, batch_size
):
    return bolt_v2.nn.FullyConnected(
        dim=config["dim"],
        input_dim=input_dim,
        sparsity=config.get("sparsity", 1.0),
        activation=config["activation"],
        sampling_config=config.get("sampling_config"),
        rebuild_hash_tables=rebuild_hash_tables // batch_size,
        reconstruct_hash_functions=reconstruct_hash_functions // batch_size,
    )
