from thirdai import bolt, dataset


layers = [
    bolt.FullyConnected(
        dim=1000, activation_function=bolt.ActivationFunctions.ReLU
    ),

    bolt.FullyConnected(
        dim=2, activation_function=bolt.ActivationFunctions.Softmax
    )
]

network = bolt.Network(
    layers=layers, input_dim=10000
)
train_x, train_y = dataset.load_bolt_svm_dataset("test.svm", 3)

# network = bolt.Network(
#     layers=layers, input_dim=7
# )
# train_x, train_y = dataset.load_bolt_csv_dataset("test.csv", 3)


network.train(train_data=train_x,
            train_labels=train_y,
            loss_fn=bolt.CategoricalCrossEntropyLoss(),
            learning_rate=0.01,
            epochs=1)

network.train(train_data=train_x,
            train_labels=train_y,
            loss_fn=bolt.CategoricalCrossEntropyLoss(),
            learning_rate=0.01,
            epochs=1)

network.predict(test_data=train_x,
            test_labels=train_y,
            batch_size=2,
            metrics=["categorical_accuracy"])