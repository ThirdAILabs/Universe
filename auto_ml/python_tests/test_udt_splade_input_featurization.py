import os

import pytest
from thirdai import bolt, dataset

from conftest import download_bert_base_uncased

pytestmark = [pytest.mark.unit]
VOCAB_SIZE = 30522
NUM_CLASSES = 3


def build_model(vocab_size):
    input = bolt.nn.Input(dim=vocab_size)

    emb = bolt.nn.Embedding(dim=64, input_dim=vocab_size, activation="relu")(input)

    output = bolt.nn.FullyConnected(
        dim=NUM_CLASSES,
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
        output_range=NUM_CLASSES,
        num_hashes=1,
        num_elements=vocab_size,
        seed=mach_index_seed,
    )
    return index


@pytest.fixture()
def create_simple_dataset():

    filename = f"splade_data.csv"
    with open(filename, "w") as file:
        file.writelines(
            [
                "text,category\n",
                "water is blue.,0\n",
                "sky is red.,1\n",
                "ground is green.,2\n",
            ]
        )

    yield filename

    os.remove(filename)


def test_splade_mach(create_simple_dataset, download_bert_base_uncased):
    train_file = create_simple_dataset
    num_models = 4

    models = [build_model(VOCAB_SIZE) for _ in range(num_models)]
    indexes = [build_index(VOCAB_SIZE, i) for i in range(num_models)]
    BERT_VOCAB_PATH = download_bert_base_uncased
    tokenizer = dataset.WordpieceTokenizer(BERT_VOCAB_PATH)

    pretrained_mach_model = bolt.SpladeMach("target", models, indexes, tokenizer)

    udt_model = bolt.UniversalDeepTransformer(
        data_types={
            "text": bolt.types.text(
                tokenizer=dataset.WordpieceTokenizer(BERT_VOCAB_PATH)
            ),
            "category": bolt.types.categorical(),
        },
        n_target_classes=NUM_CLASSES,
        integer_target=True,
        pretrained_model=pretrained_mach_model,
        options={},
    )

    udt_model.train(train_file)
    udt_model.evaluate(train_file)
    udt_model.predict({"text": "water is blue"})
