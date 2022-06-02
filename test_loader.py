from thirdai.dataset import sources
from thirdai.dataset import parsers
from thirdai.dataset import blocks
from thirdai.dataset import text_encodings
from thirdai.dataset import Schema
from thirdai.dataset import Loader

# Define source
source = sources.LocalFileSystem("/Users/benitogeordie/Desktop/thirdai_datasets/dev_auto_classifier_4.csv")
parser = parsers.CsvIterable(delimiter=",")

# Label is categorical
label = blocks.Categorical(col=0, dim=931)

# Unigrams for first 4 words
unigram_segments = [
    blocks.Text(
        col=1, 
        encoding=text_encodings.UniGram(dim=50_000, start_pos=i, end_pos=i+1))
    for i in range(4)
]
# Pairgrams for all words
pairgrams = blocks.Text(col=1, encoding=text_encodings.PairGram(dim=500_000))

schema = Schema(input_blocks=[*unigram_segments, pairgrams], target_blocks=[label])

# Assemble
loader = Loader(source, parser, schema, batch_size=2048, est_num_elems=32840248)

loader.processInMemory()