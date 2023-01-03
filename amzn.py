import numpy as np
from thirdai import bolt_v2 as bolt
from thirdai import dataset
from tqdm import tqdm

####################
### Define Model ###
####################
input_layer = bolt.nn.Input(dim=135909)
hidden = bolt.nn.FullyConnected(dim=256)(input_layer)
output = bolt.nn.FullyConnected(
    dim=670091,
    sparsity=0.005,
    activation="softmax",
    rebuild_hash_tables=25,
    reconstruct_hash_functions=500,
)(hidden)

loss = bolt.nn.losses.CategoricalCrossEntropy(output)

model = bolt.nn.Model(inputs=[input_layer], outputs=[output], losses=[loss])

model.summary()


#################
### Load Data ###
#################
TRAIN = "/share/data/amazon-670k/train_shuffled_noHeader.txt"
TEST = "/share/data/amazon-670k/test_shuffled_noHeader_sampled.txt"

train_x, train_y = dataset.load_bolt_svm_dataset(TRAIN, 256)
test_x, test_y = dataset.load_bolt_svm_dataset(TEST, 256)


for _ in range(5):
    #############
    ### Train ###
    #############
    for i in tqdm(range(len(train_x))):
        model.train_on_batch(train_x[i], train_y[i])
        model.update_parameters(0.0001)

    ################
    ### Evaluate ###
    ################
    labels = [
        [int(y) for y in x.split(" ")[0].split(",")] for x in open(TEST).readlines()
    ]

    predictions = []
    for x in tqdm(test_x):
        model.forward(x, use_sparsity=False)
        predictions.append(np.argmax(output.activations, axis=1))

    predictions = np.concatenate(predictions)

    correct = np.zeros(len(predictions))
    for i in range(len(predictions)):
        if predictions[i] in labels[i]:
            correct[i] = 1

    print(f"p@1={np.mean(correct)}")
