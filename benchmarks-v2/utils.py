import numpy as np
from sklearn.metrics import roc_auc_score
from thirdai import bolt
from thirdai.experimental import MlflowCallback


def get_mlflow_callback(run_name, mlflow_uri, experiment_name, dataset_name):
    mlflow_callback = MlflowCallback(
        tracking_uri=mlflow_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        dataset_name=dataset_name,
        experiment_args={},
    )
    return mlflow_callback


def define_fully_connected_bolt_model(config):
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


def define_dlrm_model(config):
    input_node = bolt.nn.Input(dim=config.input_dim)
    token_input = bolt.nn.TokenInput(**config.token_input)

    first_hidden_node = bolt.nn.FullyConnected(**config.first_hidden_node)(input_node)
    second_hidden_node = bolt.nn.FullyConnected(**config.second_hidden_node)(
        first_hidden_node
    )

    embedding_node = bolt.nn.FullyConnected(**config.embedding_node)(token_input)
    concat_node = bolt.nn.Concatenate()[second_hidden_node, embedding_node]

    third_hidden_node = bolt.nn.FullyConnected(**config.third_hidden_node)(concat_node)
    output_node = bolt.nn.FullyConnected(**config.output_node)(third_hidden_node)

    model = bolt.nn.Model(inputs=[input_node, token_input], output=output_node)
    model.compile(
        loss=bolt.nn.losses.get_loss_function(name=config.loss_fn),
        print_when_done=False,
    )
    model.summary(detailed=True)
    return model


def get_train_and_eval_configs(benchmark_config, callbacks=None):
    learning_rate = benchmark_config.learning_rate
    metrics = [benchmark_config.metric_type]

    train_config = bolt.TrainConfig(epochs=1, learning_rate=learning_rate).with_metrics(
        metrics
    )
    if callbacks is not None:
        train_config.with_callbacks(callbacks)

    if hasattr(benchmark_config, "reconstruct_hash_functions"):
        train_config.with_reconstruct_hash_functions(
            benchmark_config.reconstruct_hash_functions
        )

    if hasattr(benchmark_config, "rebuild_hash_tables"):
        train_config.with_rebuild_hash_tables(benchmark_config.rebuild_hash_tables)

    eval_config = bolt.EvalConfig().with_metrics(metrics)
    if benchmark_config.compute_roc_auc == True:
        eval_config.return_activations()

    return train_config, eval_config


def compute_roc_auc(predict_output, test_labels_path, mlflow_callback=None):
    with open(test_labels_path) as file:
        test_labels = [np.array([int(line[0]) for line in file.readlines()])]

    if len(predict_output) != 2:
        raise ValueError("Cannot compute the AUC without dense activations")

    activations = predict_output[1]
    if len(activations) != len(test_labels):
        raise ValueError(f"Length of activations must match the length of test labels")
    # If there are two output neurons then the true scores are activations of the second neuron.
    if len(activations.shape) == 2 and activations.shape[1] == 2:
        scores = activations[:, 1]
    # If there is a single output neuron the it is the true score.
    elif len(activations.shape) == 2 and activations.shape[1] == 1:
        scores = activations[:, 0]
    else:
        raise ValueError(
            "Activations must have shape (n,1) or (n,2) to compute the AUC"
        )
    auc = roc_auc_score(test_labels, scores)
    print(f"AUC : {auc}")

    if mlflow_callback is not None:
        mlflow_callback.log_additional_metric(key="roc_auc", value=auc)
