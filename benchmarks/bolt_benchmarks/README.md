# Bolt Benchmarks Guide

The Bolt benchmarks can be run with the `bolt.py` script in this directory. This script can be used as follows:

```python
usage: bolt.py [-h] [--disable_mlflow] [--run_name RUN_NAME] config_file

Runs creates and trains a bolt network on the specified config.

positional arguments:
  config_file          Name of the config file to use to run experiment.

optional arguments:
  -h, --help           show this help message and exit
  --disable_mlflow     Disable mlflow logging for the current run.
  --run_name RUN_NAME  The name of the run to use in mlflow, if mlflow is not disabled this is required.
```

The config file is a toml file that contains information about the dataset, model architecture, and other training parameters. This can specify both fully connected networks and DLRM models. An example script for a fully connected network is `configs/mnist_sh.txt` and an example DLRM scrip tis `configs/criteo_dlrm.txt`. 

When the `bolt.py` script is invoked with a config file it will construct the network and perform the specified training. It will run the test dataset after each epoch, if the test dataset is large then the `max_test_batches` parameter can be specified in the config file that will limit how many test batches will run after each epoch. Note that even with this option it will run on the whole test dataset at the end of training. 

The config file is organized into a few main sections. 

* job - this field simply specifies the name which is used as the name of the experiment in mlflow. This is not the same as the run name command line argument which helps delineate the different runs of the same experiment. 
* [dataset] - this section specifies information about the dataset, including the path, input dimension to the network, and the format if the dataset is svm or csv. 
* [params] - this section provides information about various parameters for training such as the loss function, number of epochs, metrics, learning rate, batch size, etc. 
* Network Architecture - the way the architecture is specified in the config file depends on if it is a fully connected or DLRM model.  
  * Fully Connected Model - for a fully connnected model architecture the layers are specified by the [[layers]] sections which give the dimension, activation function, sparsity, and any other sampling paramters (or indicate that it should be autotuned).
  * DLRM - for a DLRM model there are several sections for the architectore. The [embedding_layer] section gives the architecture of the embedding layer, its size, number of lookups, etc. The remaining information is in the sections [[bottom_mlp]] and [[top_mlp]] which have the same format of the [[layers]] section for a fully connected model and indicate the architecture of the two fully connected networks that are used in the DLRM model. 