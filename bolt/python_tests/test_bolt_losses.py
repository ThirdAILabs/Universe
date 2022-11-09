from thirdai import bolt
from utils import gen_numpy_training_data, get_simple_dag_model

N_CLASSES = 10


def test_bolt_losses():
    model = get_simple_dag_model(
        input_dim=N_CLASSES,
        hidden_layer_dim=2000,
        hidden_layer_sparsity=1.0,
        output_dim=N_CLASSES,
        output_activation="sigmoid",
        loss=bolt.nn.losses.BinaryCrossEntropy(),
    )

    train_data, train_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=10, noise_std=0.3
    )

    train_config = bolt.TrainConfig(learning_rate=0.01, epochs=1)

    for i in range(20):
        res = model.train(train_data, train_labels, train_config)
        print("Loss from epoch", i, res["last_epoch_losses"])


test_bolt_losses()
