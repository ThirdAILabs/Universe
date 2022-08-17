# Bolt Benchmarks Guide

## Usage

The Bolt benchmarks can be run with the `bolt.py` script in this directory. This script can be used as follows:

```shell
usage: bolt.py [-h] [--disable_mlflow] [--run_name RUN_NAME] config_file

Runs creates and trains a bolt network on the specified config.

positional arguments:
  config_file          Name of the config file to use to run experiment.

optional arguments:
  -h, --help           show this help message and exit
  --disable_mlflow     Disable mlflow logging for the current run.
  --run_name RUN_NAME  The name of the run to use in mlflow, if mlflow is not disabled this is required.
```

## Config File Usage

The config file is a toml file that contains information about the dataset, model architecture, and other training parameters. These files are located in `benchmarks/bolt_benchmarks/configs` and are saved with the .txt extension so that they can be uploaded as artifacts to mlflow. These config files can specify both fully connected networks and DLRM models. An example script for a fully connected network is `configs/mnist_sh.txt` and an example DLRM scrip tis `configs/criteo_dlrm.txt`.

When the `bolt.py` script is invoked with a config file it will construct the network and perform the specified training. It will run the test dataset after each epoch, if the test dataset is large then the `max_test_batches` parameter can be specified in the config file that will limit how many test batches will run after each epoch. Note that even with this option it will run on the whole test dataset at the end of training. 

## Config File Format
The config file is organized into a few main sections. 

* `job` - this field simply specifies the name which is used as the name of the experiment in mlflow. This is not the same as the run name command line argument which helps delineate the different runs of the same experiment. Example:
```toml
job = "Amazon670k Product Recommendation"

```
* `[dataset]` - this section specifies information about the dataset, including the path, input dimension to the network, and the format if the dataset is svm or csv. Example:
```toml
[dataset]
train_data = "amazon-670k/train_shuffled_noHeader.txt"
test_data = "amazon-670k/test_shuffled_noHeader_sampled.txt"
format = "svm"
input_dim = 135909
max_test_batches = 20
```
* `[params]` - this section provides information about various parameters for training such as the loss function, number of epochs, metrics, learning rate, batch size, etc. Example:
```toml
[params]
loss_fn = "CategoricalCrossEntropyLoss"
train_metrics = []
test_metrics = ["categorical_accuracy"]
learning_rate = 0.0001
epochs = 5
batch_size = 256
rehash = 6400
rebuild = 128000
```
* __Network Architecture__ - the way the architecture is specified in the config file depends on if it is a fully connected or DLRM model.
  * __Fully Connected Model__ - for a fully connnected model architecture the layers are specified by the `[[layers]]` sections which give the dimension, activation function, sparsity, and any other sampling paramters (or indicate that it should be autotuned). Example:
  ```toml
  [[layers]]
  dim = 256
  activation = 'ReLU'

  # With autotuning
  [[layers]]
  dim = 670091
  activation = 'Softmax'
  sparsity = 0.005
  use_default_sampling = true

  # Without autotuning
  [[layers]]
  dim = 670091
  activation = 'Softmax'
  sparsity = 0.005
  hashes_per_table = 5
  num_tables = 128
  reservoir_size = 128
  range_pow = 15
  ```
* __DLRM__ - for a DLRM model there are several sections for the architecture. The `[embedding_layer]` section gives the architecture of the embedding layer, its size, number of lookups, etc. The remaining information is in the sections `[[bottom_mlp_layers]]` and `[[top_mlp_layers]]` which have the same format of the `[[layers]]` section for a fully connected model and indicate the architecture of the two fully connected networks that are used in the DLRM model. Example:
  ```toml
  [embedding_layer]
  num_embedding_lookups = 8
  lookup_size = 16
  log_embedding_block_size = 10
  [[bottom_mlp_layers]]
  # ...
  [[top_mlp_layers]]
  # ...
  ```
## Authentication on S3 Bucket for Uploading Artifacts

TODO(vihan)
