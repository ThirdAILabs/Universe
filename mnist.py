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
    dim=1000,
    sparsity=0.2,
    activation="relu",
    sampling_config=old_bolt.nn.RandomSamplingConfig(),
)(input_layer)
output_1 = bolt.nn.FullyConnected(dim=10, activation="softmax")(hidden)
output_2 = bolt.nn.FullyConnected(dim=10, activation="softmax")(hidden)

loss_1 = bolt.nn.losses.CategoricalCrossEntropy(output_1)
loss_2 = bolt.nn.losses.CategoricalCrossEntropy(output_2)

model = bolt.nn.Model(
    inputs=[input_layer], outputs=[output_1, output_2], losses=[loss_1, loss_2]
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
for (x, y) in tqdm(list(zip(train_x, train_y))):
    model.train_on_batch([x], [y, y])
    model.update_parameters(0.0001)

################
### Evaluate ###
################
labels = [int(x.split(" ")[0]) for x in open(TEST).readlines()]

predictions_1 = []
predictions_2 = []
for x in tqdm(test_x):
    model.forward(x, False)
    predictions_1.append(np.argmax(output_1.activations, axis=1))
    predictions_2.append(np.argmax(output_2.activations, axis=1))


acc_1 = np.mean(np.array(labels) == np.concatenate(predictions_1, axis=0))
acc_2 = np.mean(np.array(labels) == np.concatenate(predictions_2, axis=0))

print("Output 1 Accuracy := ", acc_1)
print("Output 2 Accuracy := ", acc_2)
