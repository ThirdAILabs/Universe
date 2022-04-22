import string
import random
import time
from core import Dataset, Schema
from parsers import CsvIterable
from sources import InMemoryCollection
from blocks import TextBlock
from models import TextOneHotEncoding

######## EXAMPLE "VIRTUAL FILE" ########

letters = string.ascii_letters


def generate_list_of_passages(
    n_passages: int, min_words_per_passage: int, max_words_per_passage: int
):
    def random_word():
        return "".join(random.choice(letters) for _ in range(random.randint(3, 12)))

    def random_passage(min_length: int, max_length: int):
        return " ".join(
            random_word() for _ in range(random.randint(min_length, max_length))
        )

    return [
        random_passage(min_words_per_passage, max_words_per_passage)
        for _ in range(n_passages)
    ]


start_generate = time.time()
long_list_of_passages = generate_list_of_passages(
    n_passages=10, min_words_per_passage=20, max_words_per_passage=200
)
end_generate = time.time()
print(f"Generated long list of passages in {end_generate - start_generate} seconds.")

######## DEFINE SOURCE ########

source = InMemoryCollection(long_list_of_passages)
parser = CsvIterable()

######## DEFINE SCHEMA ########

map_lower = lambda str_list: [str.lower(s) for s in str_list]
map_word_unigram = lambda str_list: [item for s in str_list for item in s.split(" ")]
text_preprocessing_pipeline = [map_lower, map_word_unigram]

text_embed = TextOneHotEncoding(seed=10, output_dim=1000)
text_block = TextBlock(
    column=0, pipeline=text_preprocessing_pipeline, embedding_model=text_embed
)

schema = Schema(input_blocks=[text_block])

######## PUT TOGETHER ########

dataset = (
    Dataset().set_source(source).set_parser(parser).set_schema(schema).set_batch_size(3)
)

start = time.time()
count = 0
for batch in dataset.process():
    print(batch.to_string())
    count += batch.size()
end = time.time()
print(
    f"Done in {end - start} seconds. Throughput: {count / (end - start)} rows per second"
)
