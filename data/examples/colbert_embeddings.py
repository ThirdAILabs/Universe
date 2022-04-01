"""
Our dataset API is flexible enough to be used with external libraries
and embedding models. In this example, we will use ColBERT embeddings
in our dataset preprocessing pipeline.
"""

# Import modules from dataset API
from data.core import Dataset, Schema
from data.sources import LocalFileSystem, CsvIterable
from data.blocks import TextBlock
from data.models import CustomDenseTextEmbedding

# Import ColBERT module
from colbert.wrapper import ColBERT

######## DEFINE SOURCE ########

source_location = LocalFileSystem("~/Desktop/Universe/data/examples/colbert_test_file.csv")
source_format = CsvIterable()

######## DEFINE SCHEMA ########

# TODO(GEORDIE): ASK THARUN: HOW DOES THIS WORK
colbert_model = ColBERT()
text_embed = CustomDenseTextEmbedding(embed_fn=colbert_model.doc, out_dim=colbert_model.output_dim())
text_block = TextBlock(column=0, embedding_model=text_embed)

schema = Schema(input_feature_blocks=[text_block])

######## PUT TOGETHER ########

dataset = Dataset() \
  .set_source(source_location, source_format) \
  .set_schema(schema=schema) \
  .set_batch_size(3)


for batch in dataset.process():
  print(batch.to_string())


