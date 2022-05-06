import string
import random
import time
from core import Dataset, Schema
from parsers import CsvIterable
from sources import LocalFileSystem
from thirdai.dataset import Text, Categorical
# from models import TextOneHotEncoding
from thirdai.dataset import BoltTokenizer
from thirdai import bolt

######## DEFINE SOURCE ########

source = LocalFileSystem("/Users/benitogeordie/Desktop/thirdai_datasets/amazon_polarity_train.txt")
parser = CsvIterable(delimiter='\t')

######## DEFINE SCHEMA ########

map_lower = lambda str_list: [str.lower(s) for s in str_list]

text_embed = BoltTokenizer(seed=10, feature_dim=100000)
text_block = Text(
    col=1, embedding=text_embed
)

label_block = Categorical(col=0, dim=2)

schema = Schema(input_blocks=[text_block], target_blocks=[label_block])

######## PUT TOGETHER ########

dataset = (
    Dataset().set_source(source).set_parser(parser).set_schema(schema).set_batch_size(1024)
)

print("Starting")
start = time.time()
train_data = dataset.processInMemory()
end = time.time()
print("That took", end - start, "seconds.")

# layers = [
    
    # bolt.FullyConnected(
    #     dim=2000, 
    #     load_factor=0.2, 
    #     activation_function=bolt.ActivationFunctions.ReLU,
    #     sampling_config=bolt.SamplingConfig(
    #         hashes_per_table=5,
    #         num_tables=128,
    #         reservoir_size=128,
    #         range_pow=15
    #     )),
        
#     bolt.FullyConnected(
#         dim=2,
#         load_factor=1.0, 
#         activation_function=bolt.ActivationFunctions.Softmax)     
# ]

# network = bolt.Network(
#     layers=layers, 
#     input_dim=dataset._schema.input_dim())

# network.train(
#     train_data=train_data,
#     loss_fn=bolt.CategoricalCrossEntropyLoss(), 
#     learning_rate=0.0001, 
#     epochs=20, 
#     metrics=["categorical_accuracy"],
#     verbose=True)
