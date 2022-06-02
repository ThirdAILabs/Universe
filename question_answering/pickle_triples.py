from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)

import numpy as np
import tensorflow as tf


def tokenize(input_tokens):
    # Truncate start and end token
    return tf.keras.preprocessing.sequence.pad_sequences(
        [t.ids[1:-1] for t in tokenizer.encode_batch(input_tokens)], padding="post"
    )


with open("triples.train.small.tsv") as f:
    queries = []
    positives = []
    negatives = []
    number_of_examples = 10**6
    lines = [f.readline() for _ in range(number_of_examples)]
    lines = [l.split("\t") for l in lines if l != None]
    queries = tokenize([l[0] for l in lines])
    positives = tokenize([l[1] for l in lines])
    negatives = tokenize([l[2] for l in lines])

    np.save("tokenized_queries.npy", queries)
    np.save("tokenized_positives.npy", positives)
    np.save("tokenized_negatives.npy", negatives)
