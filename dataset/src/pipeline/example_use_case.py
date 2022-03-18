from pipeline import Pipeline
from feature_schema import Schema
from example_concrete_components import InMemoryCollection, CsvPythonList, TextOneHotEncoding, TextBlock

# from thirdai.data.core import Dataset, Schema
# from thirdai.data.blocks import TextBlock
# from thirdai.data.source import S3Source, etc...
# from thirdai.data.models import TextOneHotEncoding, CategoricalOneHotEncoding
# from thirdai.engine.bolt import Bolt

######## EXAMPLE "VIRTUAL FILE" ########

list_of_passages = [
  "Hello World",
  "I love ThirdAI",
  "i love thirdai so much",
  "Fantastic Bugs and Where to Find them"
]

######## DEFINE SOURCE ########

source_location = InMemoryCollection(list_of_passages)
source_format = CsvPythonList()

######## DEFINE SCHEMA ########

map_lower = lambda str_list: [str.lower(s) for s in str_list]
map_word_unigram = lambda str_list: [item for s in str_list for item in s.split(' ')]
text_preprocessing_pipeline = [map_lower, map_word_unigram]

text_embed = TextOneHotEncoding(seed=10, output_dim=1000)
text_block = TextBlock(column=0, pipeline=text_preprocessing_pipeline, embedding_model=text_embed)

schema = Schema(input_feature_blocks=[text_block])

######## PUT TOGETHER ########

# TODO: "Dataset"
dataset = Pipeline() \
  .set_source(source_location, source_format) \
  .set_schema(schema=schema) \
  .set_batch_size(2) \
  .shuffle()

dataset = (
  Pipeline()
    .set_source(source_location, source_format)
    .set_schema(schema=schema)
    # TODO: Implement these
    .set_batch_size(2)
    .shuffle()
)

for batch in dataset.process():
  print(batch.to_string())

# Integrate with colbert
# Get the packaging right (also, pybind doesnt give autocomplete and documentation)
# Talk to nick about passing generator to BOLT (if in memory, that should be handled by dataset)
# Also talk to nick about n tower models and making things efficient. And parallel.




