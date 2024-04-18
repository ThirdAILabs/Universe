import pytest

from thirdai import bolt, data, dataset
from conftest import download_bert_base_uncased

import os
import json

pytestmark = [pytest.mark.unit]
VOCAB_SIZE = 30522


def build_model(vocab_size):
    input = bolt.nn.Input(dim=vocab_size)

    emb = bolt.nn.Embedding(dim=64, input_dim=vocab_size, activation="relu")(input)

    output = bolt.nn.FullyConnected(
        dim=100,
        input_dim=emb.dim(),
        sparsity=0.02,
        activation="softmax",
    )(emb)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output, labels=bolt.nn.Input(dim=output.dim())
    )

    return bolt.nn.Model(inputs=[input], outputs=[output], losses=[loss])


def build_index(vocab_size, mach_index_seed):
    index = dataset.MachIndex(
        output_range=100, num_hashes=1, num_elements=vocab_size, seed=mach_index_seed
    )
    return index


@pytest.fixture()
def create_simple_dataset():
    def to_json_sample(text):
        return json.dumps({"target": text}) + "\n"

    filename = f"nwp.txt"
    with open(filename, "w") as file:
        file.writelines(
            [
                to_json_sample("0 1 2 3 4 5 6"),
                to_json_sample("7 8 9 10 11"),
                to_json_sample("12 13 14 15 16 17"),
            ]
        )

    yield filename

    os.remove(filename)


def test_test_mach_pretrained(create_simple_dataset, download_bert_base_uncased):
    train_file = create_simple_dataset
    num_models = 4

    models = [build_model(VOCAB_SIZE) for _ in range(num_models)]
    indexes = [build_index(VOCAB_SIZE, i) for i in range(num_models)]
    BERT_VOCAB_PATH = download_bert_base_uncased
    tokenizer = dataset.WordpieceTokenizer(BERT_VOCAB_PATH)

    pretrained_mach_models = bolt.MachPretrained(
        "target", models, indexes, tokenizer, VOCAB_SIZE
    )

    json_column_data = data.JsonIterator(dataset.FileDataSource(train_file), ["target"])

    pretrained_mach_models.train(json_column_data, 1, 10, 0.001, json_column_data)

    top_buckets = pretrained_mach_models.get_top_hash_buckets(
        ["Hello! How are you!"], 10
    )

    top_tokens = pretrained_mach_models.get_top_tokens(["Hello! How are you!"], 10)

    pretrained_mach_models.save("./pretrained_mach_model")

    pretrained_mach_model_load = bolt.MachPretrained.load("./pretrained_mach_model")

    os.remove("/pretrained_mach_model")
