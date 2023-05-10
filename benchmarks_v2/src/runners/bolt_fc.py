from thirdai import bolt

from ..configs.bolt_configs import BoltBenchmarkConfig
from .runner import Runner
from .utils import get_train_and_eval_configs


class BoltFullyConnectedRunner(Runner):
    config_type = BoltBenchmarkConfig

    def run_benchmark(config: BoltBenchmarkConfig, path_prefix, mlflow_logger):
        model = define_fully_connected_bolt_model(config)
        train_set, train_labels, test_set, test_labels = config.load_datasets(
            path_prefix
        )

        train_config, eval_config = get_train_and_eval_configs(
            benchmark_config=config, callbacks=config.callbacks
        )

        for epoch in range(config.num_epochs):
            train_metrics = model.train(
                train_data=train_set,
                train_labels=train_labels,
                train_config=train_config,
            )

            if mlflow_logger:
                for k, v in train_metrics.items():
                    mlflow_logger.log_additional_metric(key=k, value=v[0], step=epoch)

            predict_output = model.evaluate(
                test_data=test_set, test_labels=test_labels, eval_config=eval_config
            )

            if mlflow_logger:
                for k, v in predict_output[0].items():
                    mlflow_logger.log_additional_metric(key=k, value=v, step=epoch)


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
