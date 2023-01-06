import numpy as np
from thirdai import bolt as old_bolt
from thirdai import bolt_v2 as bolt
from thirdai import dataset
from tqdm import tqdm

####################
### Define Model ###
####################
input_layer = bolt.nn.Input(dim=784)
hidden = bolt.nn.FullyConnected(
    dim=1_000,
    sparsity=0.1,
    activation="relu",
    sampling_config=old_bolt.nn.DWTASamplingConfig(
        num_tables=64, hashes_per_table=3, reservoir_size=32
    ),
    rebuild_hash_tables=12,
    reconstruct_hash_functions=40,
)(input_layer)
output = bolt.nn.FullyConnected(dim=10, activation="softmax")(hidden)

loss = bolt.nn.losses.CategoricalCrossEntropy(output)

model = bolt.nn.Model(
    inputs=[input_layer], outputs=[output], losses=[loss]
)

model.summary()


#################
### Load Data ###
#################
TRAIN = "/Users/nmeisburger/ThirdAI/data/mnist/mnist"
TEST = "/Users/nmeisburger/ThirdAI/data/mnist/mnist.t"

train_x, train_y = dataset.load_bolt_svm_dataset(TRAIN, 250)
test_x, test_y = dataset.load_bolt_svm_dataset(TEST, 250)

#############
### Train ###
#############
trainer = bolt.train.Trainer(model)

history = trainer.train(
    train_data=(train_x, train_y),
    epochs=3,
    learning_rate=0.0001,
    train_metrics={
        "act_2": [bolt.train.metrics.LossMetric(loss)],
    },
    validation_data=(test_x, test_y),
    validation_metrics={
        "act_2": [bolt.train.metrics.LossMetric(loss)],
    },
    steps_per_validation=None,
    callbacks=[],
)

print(history)

################
### Evaluate ###
################
labels = [int(x.split(" ")[0]) for x in open(TEST).readlines()]

predictions = []
for x in tqdm(test_x):
    model.forward(x, False)
    predictions.append(np.argmax(output.activations, axis=1))


acc = np.mean(np.array(labels) == np.concatenate(predictions, axis=0))

print("Output Accuracy := ", acc)
