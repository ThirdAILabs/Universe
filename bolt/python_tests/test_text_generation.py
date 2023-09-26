import json
import os

import pytest
from thirdai import bolt, dataset

VOCAB_SIZE = 20


def create_dyadic_backend():
    N_INTERVALS = 5
    inputs = [bolt.nn.Input(dim=VOCAB_SIZE) for _ in range(N_INTERVALS)]

    embeddings = [
        bolt.nn.Embedding(dim=10, input_dim=VOCAB_SIZE, activation="relu")(inp)
        for inp in inputs
    ]

    embedding = bolt.nn.Concatenate()(embeddings)

    hidden = bolt.nn.FullyConnected(
        dim=20, input_dim=embedding.dim(), activation="relu"
    )(embedding)

    norm = bolt.nn.LayerNorm()(hidden)

    output = bolt.nn.FullyConnected(
        dim=VOCAB_SIZE,
        input_dim=norm.dim(),
        sparsity=0.5,
        activation="softmax",
        rebuild_hash_tables=4,
        reconstruct_hash_functions=25,
    )(norm)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        output, labels=bolt.nn.Input(dim=output.dim())
    )

    model = bolt.nn.Model(inputs=inputs, outputs=[output], losses=[loss])

    return bolt.DyadicModel(model)


def create_contextual_backend(with_prompt=False):
    LRC_LEN = 20
    IRC_LEN = 6
    SRC_LEN = 4
    lrc_input = bolt.nn.Input(dim=VOCAB_SIZE)
    irc_input = bolt.nn.Input(dim=(2**32) - 1)
    src_input = bolt.nn.Input(dim=VOCAB_SIZE)
    prompt_input = bolt.nn.Input(dim=VOCAB_SIZE)

    small_emb = bolt.nn.RobeZ(
        num_embedding_lookups=4,
        lookup_size=3,
        log_embedding_block_size=10,
        reduction="avg",
    )

    small_ebm_src = small_emb.duplicate_with_new_reduction(
        reduction="concat", num_tokens_per_input=SRC_LEN
    )

    large_emb = bolt.nn.RobeZ(
        num_embedding_lookups=8,
        lookup_size=4,
        log_embedding_block_size=10,
        reduction="avg",
    )

    computations = [
        small_emb(lrc_input),
        large_emb(lrc_input),
        small_emb(irc_input),
        large_emb(irc_input),
        small_ebm_src(src_input),
    ]

    if with_prompt:
        computations += [small_emb(prompt_input)]

    concat = bolt.nn.Concatenate()(computations)

    hidden = bolt.nn.FullyConnected(
        dim=20,
        input_dim=concat.dim(),
        activation="relu",
    )(concat)
    hidden = bolt.nn.LayerNorm()(hidden)
    output = bolt.nn.FullyConnected(
        dim=VOCAB_SIZE,
        input_dim=hidden.dim(),
        sparsity=0.4,
        activation="softmax",
        rebuild_hash_tables=4,
        reconstruct_hash_functions=25,
    )(hidden)

    labels = bolt.nn.Input(dim=VOCAB_SIZE)
    loss = bolt.nn.losses.CategoricalCrossEntropy(activations=output, labels=labels)

    inputs = [lrc_input, irc_input, src_input]

    if with_prompt:
        inputs += [prompt_input]

    model = bolt.nn.Model(inputs=inputs, outputs=[output], losses=[loss])

    featurizer = dataset.TextGenerationFeaturizer(
        lrc_len=LRC_LEN,
        irc_len=IRC_LEN,
        src_len=SRC_LEN,
        vocab_size=VOCAB_SIZE,
        include_position=False,
    )

    return bolt.ContextualModel(model, featurizer)


@pytest.mark.unit
@pytest.mark.parametrize("backend", [create_dyadic_backend, create_contextual_backend])
def test_generation(backend):
    model = bolt.GenerativeModel(
        backend(), allowed_repeats=set(), punctuation_tokens=set()
    )

    gen_1 = model.generate(
        input_tokens=list(range(20)), beam_width=5, max_predictions=20, temperature=0.4
    )

    stream = model.streaming_generate(
        input_tokens=list(range(20)),
        beam_width=5,
        max_predictions=20,
        prediction_chunk_size=6,
        temperature=0.4,
    )

    for res in stream:
        gen_2 = res

    assert gen_1 == gen_2


@pytest.fixture()
def create_simple_dataset():
    def to_json_sample(text):
        return json.dumps({"target": text}) + "\n"

    filename = "nwp.txt"
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


@pytest.mark.unit
@pytest.mark.parametrize("backend", [create_dyadic_backend, create_contextual_backend])
def test_nwp_training(backend, create_simple_dataset):
    model = bolt.GenerativeModel(
        backend(), allowed_repeats=set(), punctuation_tokens=set()
    )

    filename = create_simple_dataset

    train_data = dataset.FileDataSource(filename)
    val_data = dataset.FileDataSource(filename)

    model.train(
        train_data=train_data,
        epochs=3,
        learning_rate=0.0001,
        train_metrics=["loss"],
        val_data=val_data,
        val_metrics=["loss", "categorical_accuracy"],
    )
