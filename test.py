from thirdai import bolt, dataset

input_layer = bolt.graph.Input(dim=784)

hidden_layer = bolt.graph.FullyConnected(
    bolt.FullyConnected(
        dim=1000, sparsity=0.1, activation_function=bolt.ActivationFunctions.ReLU
    )
)

hidden_layer(input_layer)

output_layer = bolt.graph.FullyConnected(
    bolt.FullyConnected(dim=10, activation_function=bolt.ActivationFunctions.Softmax)
)

output_layer(hidden_layer)

model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

model.compile(loss=bolt.CategoricalCrossEntropyLoss())


train_data, train_labels = dataset.load_bolt_svm_dataset(
    "/Users/nmeisburger/ThirdAI/data/mnist/mnist", 250
)

test_data, test_labels = dataset.load_bolt_svm_dataset(
    "/Users/nmeisburger/ThirdAI/data/mnist/mnist.t", 250
)


for _ in range(5):
    model.train(
        train_data=train_data,
        train_labels=train_labels,
        learning_rate=0.0001,
        epochs=1,
        rehash=3000,
        rebuild=10000,
    )

    model.predict(
        test_data=test_data, test_labels=test_labels, metrics=["categorical_accuracy"]
    )
