from cookie_monster import *

# Initialize the model
layers = [bolt.FullyConnected(dim=2000, sparsity=0.1, activation_function=bolt.ActivationFunctions.ReLU),
            bolt.FullyConnected(dim=2,  activation_function=bolt.ActivationFunctions.Softmax)]

cookie_model = CookieMonster(layers)

# Step One: Download a set of weights&biases from MLflow or go to step two from scratch
# cookie_model.download_hidden_weights(link_to_weights_mlflow, link_to_biases_mlflow)

# Step Two: Train the model on a corpus. Specify a directory containing config files for each dataset to be trained on.
#           The model will iterate through each file and train on each dataset.
cookie_model.train_corpus("/home/henry/cookie_train/", mlflow=True)

# Step Three: Evaluate the model on a set of benchmark datasets.