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
unigram_0 = blocks.Text(1, text_encodings.UniGram(dim=50_000, start_pos=0, end_pos=1))
unigram_1 = blocks.Text(1, text_encodings.UniGram(dim=50_000, start_pos=1, end_pos=2))
unigram_2 = blocks.Text(1, text_encodings.UniGram(dim=50_000, start_pos=2, end_pos=3))
unigram_3 = blocks.Text(1, text_encodings.UniGram(dim=50_000, start_pos=3, end_pos=4))
# Pairgrams for all words
pairgrams = blocks.Text(1, text_encodings.PairGram(dim=500_000))

schema = Schema(input_blocks=[unigram_0, unigram_1, unigram_2, unigram_3, pairgrams], target_blocks=[label])

# Assemble
loader = Loader(source, parser, schema, batch_size=2048)

loader.processInMemory()