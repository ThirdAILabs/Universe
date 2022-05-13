from thirdai.dataset import Loader, Schema, parsers, sources, blocks, text_encodings

def load_text_classification_dataset(file: str, delim: str='\t', labeled: bool=True, label_dim: int=2, batch_size: int=1024, encoding_dim: int=100_000):
    
    ######## DEFINE SOURCE ########
    source = sources.LocalFileSystem(file)
    parser = parsers.CsvIterable(delimiter='\t')

    ######## DEFINE SCHEMA ########
    text_block = blocks.Text(col=1, dim=100000)
    label_block = blocks.Categorical(col=0, dim=2)
    schema = Schema(input_blocks=[text_block], target_blocks=[label_block])

    ######## PUT TOGETHER ########
    dataset = Loader(source, parser, schema, batch_size)

    return dataset.processInMemory(), dataset.input_dim()

import time
from thirdai import bolt

print("Starting")
start = time.time()
train_data, input_dim = load_text_classification_dataset("/Users/benitogeordie/Desktop/thirdai_datasets/amazon_polarity_train_100k.txt")
end = time.time()
print("That took", end - start, "seconds.")

print(input_dim)

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
    input_dim=input_dim)

# network.train(
#     train_data=train_data,
#     loss_fn=bolt.CategoricalCrossEntropyLoss(), 
#     learning_rate=0.0001, 
#     epochs=20, 
#     metrics=["categorical_accuracy"],
#     verbose=True)