from thirdai import bolt, dataset
from thirdai.bolt import callbacks

train_data, train_labels = dataset.load_bolt_svm_dataset(
    "/Users/ben/Documents/ThirdAI/Demos/train_shuf.svm", 256
)

layers = [
    bolt.FullyConnected(
        dim=10000,
        sparsity=0.005,
        activation_function=bolt.ActivationFunctions.ReLU,
        sampling_config=bolt.SamplingConfig(
            hashes_per_table=4, num_tables=64, range_pow=13, reservoir_size=32
        ),
    ),
    bolt.FullyConnected(
        dim=151, sparsity=1.0, activation_function=bolt.ActivationFunctions.Softmax
    ),
]

def schedule(epoch, lr):
    if epoch < 3:
        return lr
    else:
        return lr / epoch

callback = callbacks.LearningRateScheduler(schedule=schedule)

network = bolt.Network(layers=layers, input_dim=5512)

network.train(
    train_data=train_data,
    train_labels=train_labels,
    loss_fn=bolt.BinaryCrossEntropyLoss(),
    learning_rate=0.001,
    epochs=10,
    rehash=3000,
    rebuild=10000,
    verbose=True,
    callbacks=[callback]
)
