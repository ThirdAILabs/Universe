import string
import random
import time
from core import Dataset, Schema
from parsers import CsvIterable
from sources import LocalFileSystem
from thirdai.dataset import Text, Categorical, BoltTokenizer, OneHot
from thirdai import bolt


# TODO: rename embedding to encoding? or something else? vectorization? tokenization? featurization?
# TODO: Play with huggingface

schema = Schema(
    input_blocks=[
        Text(
            col=1, 
            embedding=BoltTokenizer(
                seed=10, 
                dim=100_000))
    ], 
    target_blocks=[
        Categorical(
            col=0,
            from_string=False,
            embedding=OneHot(dim=2)
        )
    ])

# CONCISE VERSION
# Schema seems redundant
schema = Schema(
    input_blocks=[Text(col=1, dim=100_000)], 
    target_blocks=[Categorical(col=0, dim=2)])

######## PUT TOGETHER ########

dataset = Dataset(
    source=LocalFileSystem("/Users/benitogeordie/Desktop/thirdai_datasets/amazon_polarity_train.txt"),
    parser=CsvIterable(delimiter='\t'),
    schema=schema,
    batch_size=1024
)

dataset = Dataset(
    source=LocalFileSystem("/Users/benitogeordie/Desktop/thirdai_datasets/amazon_polarity_train.txt"),
    parser=CsvIterable(delimiter='\t'),
    schema=Schema(
        input_blocks=[Text(col=1, dim=100_000)], 
        target_blocks=[Categorical(col=0, dim=2)]),
    batch_size=1024
)

print("Starting")
start = time.time()
train_data = dataset.processInMemory()
end = time.time()
print("That took", end - start, "seconds.")

layers = [  
    bolt.FullyConnected(
        dim=2000, 
        load_factor=0.2, 
        activation_function=bolt.ActivationFunctions.ReLU,
        sampling_config=bolt.SamplingConfig(
            hashes_per_table=5,
            num_tables=128,
            reservoir_size=128,
            range_pow=15
        )),
        
    bolt.FullyConnected(
        dim=2,
        load_factor=1.0, 
        activation_function=bolt.ActivationFunctions.Softmax)     
]

network = bolt.Network(
    layers=layers, 
    input_dim=dataset._schema.input_dim())

network.train(
    train_data=train_data,
    loss_fn=bolt.CategoricalCrossEntropyLoss(), 
    learning_rate=0.0001, 
    epochs=20, 
    metrics=["categorical_accuracy"],
    verbose=True)
