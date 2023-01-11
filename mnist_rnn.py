from thirdai import bolt_v2 as bolt
from thirdai import dataset
import numpy as np
import tqdm


def read_data(filename):
    lines = open(filename).readlines()

    data = np.zeros(shape=(len(lines), 784), dtype=np.float32)
    labels = []

    for i, line in enumerate(lines):
        items = line.split(' ')
        labels.append(int(items[0]))

        for kv in items[1:]:
            split_kv = kv.split(':')
            key = int(split_kv[0])
            value = float(split_kv[1])
            data[i, key] = value

    data /= 255    
    return data, np.array(labels, dtype=np.uint32)


def load_data(filename, batchsize, labels_as_np=False):
    x_np, labels = read_data(filename)

    x_nps = np.split(x_np, 28, axis=1)

    x_nps = [np.zeros((len(x_nps[0]), 100), dtype=np.float32)] + x_nps

    inputs = [dataset.from_numpy(x, batchsize) for x in x_nps]

    inputs = list(zip(*inputs))

    if not labels_as_np:
        labels = dataset.from_numpy(labels, batchsize)

    return (inputs, labels)


train_x, train_y = load_data("./build/mnist", 250)
test_x, test_y = load_data("./build/mnist.t", 250, labels_as_np=True)


def create_model(hidden_size):
    placeholder = bolt.nn.Input(dim=hidden_size, sparse_nonzeros=hidden_size)
    hidden_state = placeholder

    inputs = [bolt.nn.Input(dim=28, sparse_nonzeros=28) for _ in range(28)]

    hidden_layer = bolt.nn.FullyConnected(dim=hidden_size, input_dim=128)
     
    for timestep in inputs:
        concat = bolt.nn.Concatenate()([hidden_state, timestep])
        hidden_state = hidden_layer(concat)

    output = bolt.nn.FullyConnected(dim=10, input_dim=100, activation="softmax")(hidden_state)

    loss = bolt.nn.losses.CategoricalCrossEntropy(output)

    model = bolt.nn.Model(inputs=[placeholder] + inputs, outputs=[output], losses=[loss])

    model.summary()

    return model, output


model, output = create_model(hidden_size=100)

for (x,y) in tqdm.tqdm(list(zip(train_x, train_y))):
    model.train_on_batch(x,y)
    model.update_parameters(0.0001)


predictions=[]
for (x,y) in tqdm.tqdm(list(zip(test_x, test_y))):
    model.forward(x)
    predictions.append(np.argmax(output.activations), axis=1)


acc = np.mean(np.array(test_y) == np.concatenate(predictions, axis=0))
print(acc)