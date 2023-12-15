import numpy as np
import pytest
from thirdai import data, dataset
from transformers import BertTokenizer

from conftest import download_bert_base_uncased

pytestmark = [pytest.mark.unit]


def test_text_tokenizer_unigrams():
    featurizer = data.transformations.Text(
        input_column="input", output_indices="output", dim=0xFFFFFFFF
    )

    input_col = data.columns.StringColumn(["aa bb cc dd", "dd cc bb aa", "xx aa bb cc"])
    columns = data.ColumnMap({"input": input_col})

    columns = featurizer(columns)

    tokens = np.array(columns["output"].data())

    assert tokens.shape == (3, 4)

    # Reversed sentence should give same tokens.
    assert set(tokens[0]) == set(tokens[1])

    # 3 of the 4 words are the same and should have the same tokens
    assert len(set(tokens[0]).intersection(set(tokens[2]))) == 3


TEXT_SAMPLES = [
    "popularity      of threading has increased around 2003, as the growth of the cpu frequency was replaced with the growth of number of cores, in turn requiring concurrency to utilize multiple cores.",
    "arkansas highway 59 business is a business route in gentry.",
    "before joining city of hope, rosen worked at northwestern university.",
    "the third section (8 km) was the bypass of szczuczyn opened in november 2015 as a single carriageway road.",
    "in may 2021, the company left the swedish market due to a “unfair and discriminatory treatment” by the swedish tax agency.",
    "since 2018, the winner of the bonaire kampionato, the top tier of football on the island qualifies for the caribbean club shield, a tertiary knockout tournament for developing caribbean football nations.",
    "the tone of linlithgow's warnings to amery grew increasingly serious over the first half of 1943, as did amery's requests to the war cabinet; on 4august 1943 amery noted the spread of famine, and specifically stressed the effect upon calcutta and the potential effect on the morale of european troops.",
    "during the first years of japanese occupation following the first sino-japanese war, the political divisions of the island were changed frequently.",
    "october 27 – martin mullen, 63, right fielder in one game for the 1872 cleveland forest citys of the national association.",
    "morata de tajuna is a municipality of the community of madrid, spain.",
    "from the hebrew name × ö¸×¢ö³×ö´× na omiy meaning pleasantness in the old testament this is the name of the mother in law of ruth after",
]


def test_text_tokenizer_wordpiece(download_bert_base_uncased):
    huggingface_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    BERT_VOCAB_PATH = download_bert_base_uncased
    tokenizer = dataset.WordpieceTokenizer(BERT_VOCAB_PATH)

    featurizer = data.transformations.Text(
        input_column="input",
        output_indices="output",
        tokenizer=tokenizer,
        dim=0xFFFFFFFF,
    )

    input_col = data.columns.StringColumn(TEXT_SAMPLES)
    columns = data.ColumnMap({"input": input_col})

    columns = featurizer(columns)

    tokens = [np.array(x) for x in columns["output"].data()]

    for text, tokens in zip(TEXT_SAMPLES, tokens):
        hf_tokens = huggingface_tokenizer.encode(text, add_special_tokens=False)
        assert np.array_equal(np.array(hf_tokens), tokens)


def test_text_tokenizer_hybrid_wordpiece(download_bert_base_uncased):
    huggingface_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    BERT_VOCAB_PATH = download_bert_base_uncased
    tokenizer = dataset.WordpieceTokenizer(BERT_VOCAB_PATH)

    featurizer = data.transformations.Text(
        input_column="input",
        output_indices="output",
        tokenizer=tokenizer,
        dim=0xFFFFFFFF,
    )

    input_col = data.columns.StringColumn(TEXT_SAMPLES)
    columns = data.ColumnMap({"input": input_col})

    columns = featurizer(columns)

    tokens = [np.array(x) for x in columns["output"].data()]

    for text, tokens in zip(TEXT_SAMPLES, tokens):
        hf_tokens = huggingface_tokenizer.encode(text, add_special_tokens=False)
        num_wordpiece_tokens = len(hf_tokens)
        assert np.array_equal(np.array(hf_tokens), tokens[-num_wordpiece_tokens:])


def test_token_deduplication():
    columns = data.ColumnMap(
        {"input": data.columns.StringColumn(["a", "b", "c", "a b a a c b", ""])}
    )

    transformation = data.transformations.Text(
        input_column="input",
        output_indices="indices",
        output_values="values",
        dim=0xFFFFFFFF,
    )

    columns = transformation(columns)

    indices = columns["indices"].data()

    values = columns["values"].data()

    a_token = indices[0][0]
    b_token = indices[1][0]
    c_token = indices[2][0]

    counts = {k: v for k, v in zip(indices[3], values[3])}
    expected_counts = {a_token: 3.0, b_token: 2.0, c_token: 1.0}

    assert counts == expected_counts

    assert len(indices[-1]) == 0
    assert len(values[-1]) == 0
