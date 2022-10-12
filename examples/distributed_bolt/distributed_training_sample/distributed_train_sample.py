import thirdai.distributed_bolt as db
from thirdai import bolt, dataset
import thirdai


def get_mnist_model():
    input_layer = bolt.graph.Input(dim=100000)

    hidden_layer = bolt.graph.FullyConnected(dim=256, sparsity=0.5, activation="Relu")(
        input_layer
    )

    output_layer = bolt.graph.FullyConnected(dim=30522, activation="softmax")(hidden_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model


if __name__ == "__main__":
    model = get_mnist_model()
    dataset_paths = ["/share/pratik/mnist_a", "/share/pratik/mnist_b"]
    train_config = (
        bolt.graph.TrainConfig.make(learning_rate=0.0001, epochs=1)
        .with_metrics(["mean_squared_error"])
        .with_log_loss_frequency(32)
    )
    cluster_config = db.RayTrainingClusterConfig(
        num_workers=2, requested_cpus_per_node=1, communication_type="linear"
    )
    wrapped_model = db.DistributedDataParallel(
        cluster_config=cluster_config,
        model=model,
        train_config=train_config,
        data_loader_config="data_loader_config.txt",
        batch_size=256,
    )
    wrapped_model.train()

    model = wrapped_model.get_model()

    predict_config = (
        bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    )
    test_data, test_labels = dataset.load_bolt_svm_dataset(
        "/share/pratik/mnist.t", batch_size=256
    )
    metrics = model.predict(
        test_data=test_data, test_labels=test_labels, predict_config=predict_config
    )
    print(metrics[0]["categorical_accuracy"])
