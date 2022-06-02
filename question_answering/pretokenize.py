from tokenizers import BertWordPieceTokenizer
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description="Pretokenize a qa file.")
parser.add_argument("type", help='one of "train" or "test"')
args = parser.parse_args()


tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)


def tokenize(input_tokens, maxlen):
    # Truncate start and end token
    return tf.keras.preprocessing.sequence.pad_sequences(
        [t.ids[1:-1] for t in tokenizer.encode_batch(input_tokens)],
        padding="post",
        maxlen=maxlen,
    )


def tokenize_file(
    input_file_name,
    columns_to_tokenize,
    max_sizes,
    batch_size,
    max_num_lines,
    output_file_names,
):
    results = [[] for _ in range(len(columns_to_tokenize))]
    with open(input_file_name) as f:
        for _ in tqdm(range(max_num_lines // batch_size)):
            lines = [f.readline() for _ in range(batch_size)]
            lines = [l.split("\t") for l in lines if l != ""]
            if len(lines) == 0:
                break
            for i, (column_id, max_size) in enumerate(
                zip(columns_to_tokenize, max_sizes)
            ):
                results[i] += [tokenize([l[column_id] for l in lines], max_size)]

    for result, output_file_name in zip(results, output_file_names):
        np.save(output_file_name, np.concatenate(result))


if args.type == "train":

    tokenize_file(
        input_file_name="triples.train.small.tsv",
        columns_to_tokenize=(0, 1, 2),
        max_sizes=(32, 256, 256),
        batch_size=10**4,
        max_num_lines=10**7,
        output_file_names=(
            "tokenized_queries_train.npy",
            "tokenized_positives_train.npy",
            "tokenized_negatives_train.npy",
        ),
    )

elif args.type == "test":

    tokenize_file(
        input_file_name="/share/josh/msmarco/collection.tsv",
        columns_to_tokenize=(1,),
        max_sizes=(256,),
        batch_size=10**4,
        max_num_lines=10**7,  # This is an overestimate
        output_file_names=("tokenized_documents_test.npy",),
    )

    tokenize_file(
        input_file_name="/share/josh/msmarco/queries.dev.small.tsv",
        columns_to_tokenize=(1,),
        max_sizes=(32,),
        batch_size=10**4,
        max_num_lines=10**7,
        output_file_names=("tokenized_queries_test.npy",),
    )
