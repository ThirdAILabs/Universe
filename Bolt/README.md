# Bolt
Big Ol' Layer Training

## Instructions 

### Running in C++
1. Run `$ make` to compile the executable `bolt`.
2. The bolt executable takes in a single argument which is the name of the config file to use. e.g `$ ./bolt ./configs/mnist.cfg`

### Config File Format 
The format of the config files are `key = <value1>, ... , <value n>` where the values are comma separated. String values must be enclosed in single or double quotations. Lines starting with // are ignored.

Example: 

```c
batch_size = 128
dims = 64, 128, 256

train_data = "/usr/data/train.txt"

// Comment 
sparsity = 0.05, 1.0
```

### Compiling Python Bindings 
1. Run `$ make bindings` to compile the python bindings and generate the library `bolt.so`
2. Run `$ export PYTHONPATH=$(pwd):$PYTHONPATH` to allow python to find the `bolt.so` file. Note that this step is not necessary if the python file is located in the same directory as the `bolt.so` file. 

### A Note on the Libraries Required for the Python Bindings
This library is compiled using pybind11. This can be installed via pip, e.g `$ pip3 install pybind11`. Compiling the library also requires linking it with the python library and including python header files. On some linux systems these library and header files are in the link and include paths automatically and everything works nicely. However on Mac, and some linux systems you will need to specify the correct include and link paths and libraries. For example I am using Mac and downloaded python3 via homebrew. I had to add the link path `-L/usr/local/Cellar/python@3.9/3.9.6/Frameworks/Python.framework/Versions/3.9/lib` as will as the flag `-lpython3.9` for it to link with python library and compile successfully. If you are trying to run this on Windows, best of luck. 


### Using the Python Bindings

If you have made it this far and have managed to get the python bindings to compile, congratulations. To use the python bindings simply import the module `bolt`. The interface that is exposed allows you to construct a network and provide it a dataset to train on. After this there are methods on the network that you can use to obtain a numpy array of the weight and bias matrices for each layer. 

Example (Dense hidden layer and a sparse output layer):

```python
from thirdai import bolt

layers = [
    bolt.LayerConfig(dim=256, activation_function="ReLU"),
    bolt.LayerConfig(dim=10, sparsity=0.4, activation_function="Softmax",
                      sampling_config=bolt.SamplingConfig(K=1, L=64, 
                                                           range_pow=3, reservoir_size=10))
]

network = bolt.Network(layers=layers, input_dim=780)

network.Train(batch_size=250, train_data="/usr/data/mnist",
              test_data="/usr/data/mnist.t",
              learning_rate=0.0001, epochs=10, rehash=5000, rebuild=10000)
```

Example (Dense hidden layer, 1 sparse hidden layer and a dense output layer):

```python
from thirdai import bolt

layers = [
    bolt.LayerConfig(dim=128, activation_function="ReLU"),
    bolt.LayerConfig(dim=1024, sparsity=0.01, activation_function="ReLU",
                      sampling_config=bolt.SamplingConfig(K=3, L=128, 
                                                           range_pow=9, reservoir_size=32)),
    bolt.LayerConfig(dim=10, activation_function="Softmax")
]

network = bolt.Network(layers=layers, input_dim=780)

network.Train(batch_size=250, train_data="/usr/data/mnist",
              test_data="/usr/data/mnist.t",
              learning_rate=0.0001, epochs=10, rehash=5000, rebuild=10000)
```
