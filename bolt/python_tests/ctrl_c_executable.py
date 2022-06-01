from thirdai import bolt
import numpy as np

def get_random_dataset_as_numpy():
    no_of_training_examples = 100000
    dimension_of_input = 5

    train_data = []
    train_labels = []
    for i in range(no_of_training_examples):
        datapoints = []
        for j in range(dimension_of_input):
            datapoints.append(np.random.randint(1,high=10000))
        train_labels.append(np.random.randint(1,high=5))
        train_data.append(datapoints)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    train_data = np.float32(train_data.reshape((no_of_training_examples, dimension_of_input)))
    train_labels = np.uint32(train_labels)
    return train_data, train_labels


def train_using_random_numpy():

    train_data, train_labels = get_random_dataset_as_numpy() 
    layers = [
        
        bolt.FullyConnected(
            dim=100, 
            load_factor=0.2, 
            activation_function=bolt.ActivationFunctions.ReLU),
            
        bolt.FullyConnected(
            dim=5,
            load_factor=1.0, 
            activation_function=bolt.ActivationFunctions.Softmax)     
    ]

    network = bolt.Network(
        layers=layers, 
        input_dim=5)

    # print('Done First Training!!')
    network.train(
        train_data=train_data,
        train_labels = train_labels,
        batch_size=10,
        loss_fn=bolt.CategoricalCrossEntropyLoss(), 
        learning_rate=0.0001, 
        epochs=20, 
        verbose=True)

if  __name__ == "__main__":
    train_using_random_numpy()