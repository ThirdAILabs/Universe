import os

import pytest
from cookie_monster import CookieMonster
from thirdai import dataset
from thirdai.dataset import FixedVocabulary

pytestmark = [pytest.mark.unit]

BERT_TAG = "bert-base-uncased"
BERT_VOCAB_PATH = "{}.vocab".format(BERT_TAG)
BERT_VOCAB_URL = "https://huggingface.co/{}/resolve/main/vocab.txt".format(BERT_TAG)

BERT_SAMPLES = [
    "popularity of thread ##ing has increased around 2003 , as the growth of the cpu frequency was replaced with the growth of number of cores , in turn requiring concurrency to utilize multiple cores .",
    "arkansas highway 59 business is a business route in gentry .",
    "before joining city of hope , rosen worked at northwestern university .",
    "the third section ( 8 km ) was the bypass of s ##zcz ##uc ##zyn opened in november 2015 as a single carriage ##way road .",
    "in may 2021 , the company left the swedish market due to a “ unfair and disc ##rim ##inatory treatment ” by the swedish tax agency .",
    "since 2018 , the winner of the bon ##aire kam ##pio ##nat ##o , the top tier of football on the island qu ##ali ##fies for the caribbean club shield , a tertiary knockout tournament for developing caribbean football nations .",
    "the tone of lin ##lit ##hg ##ow ' s warnings to am ##ery grew increasingly serious over the first half of 1943 , as did am ##ery ' s requests to the war cabinet ; on 4a ##ug ##ust 1943 am ##ery noted the spread of famine , and specifically stressed the effect upon calcutta and the potential effect on the morale of european troops .",
    "during the first years of japanese occupation following the first sino - japanese war , the political divisions of the island were changed frequently .",
    "october 27 – martin mu ##llen , 63 , right fielder in one game for the 1872 cleveland forest city ##s of the national association .",
    "mora ##ta de ta ##jun ##a is a municipality of the community of madrid , spain .",
]


def setup_module():
    with open("string_mlm.txt", "w") as f:
        f.write("This is a test sentence\n")
        f.write("To train the model on mlm tasks\n")
    with open("string_classification.csv", "w") as f:
        f.write("0,This is a positive sentence\n")
        f.write("1,To train the model on classification tasks\n")

    if not os.path.exists(BERT_VOCAB_PATH):
        import urllib.request

        response = urllib.request.urlopen(BERT_VOCAB_URL)
        with open(BERT_VOCAB_PATH, "wb+") as bert_vocab_file:
            bert_vocab_file.write(response.read())


def test_new_dimension():
    vocab = FixedVocabulary.make(BERT_VOCAB_PATH)
    model = CookieMonster(
        vocab,
        input_dimension=100000,
        hidden_dimension=2000,
        hidden_sparsity=0.1,
        mlflow_enabled=False,
    )
    model.set_output_dimension(10)
    assert model.output_layer.get_dim() == 10
    model.set_output_dimension(30224)
    assert model.output_layer.get_dim() == 30224


def test_train():
    vocab = FixedVocabulary.make(BERT_VOCAB_PATH)
    model = CookieMonster(
        vocab,
        input_dimension=784,
        hidden_dimension=2000,
        hidden_sparsity=0.1,
        mlflow_enabled=False,
    )
    model.eat_corpus("test_dir", False, True)
    model.evaluate("test_dir")


def test_vocab():
    vocab = FixedVocabulary.make(BERT_VOCAB_PATH)

    with open(BERT_VOCAB_PATH) as vocab_file:
        lines = vocab_file.read().splitlines()
        assert len(lines) == vocab.size()

    for sample in BERT_SAMPLES:
        pieces = vocab.encode(sample)

        # Nothing maps to unknown, as the above is taken from BERT (thirdai)
        assert vocab.unk_id() not in pieces
        tokens = sample.split()
        if "THIRDAI_TEST_DEBUG" in os.environ:
            for piece, token in zip(pieces, tokens):
                print("{}: {}".format(token, piece), end=" ")
            print()

        assert len(pieces) == len(tokens)


def test_dataset_with_vocab(tmp_path):
    vocab = FixedVocabulary.make(BERT_VOCAB_PATH)
    mlm_loader = dataset.MLMDatasetLoader(vocab, pairgram_range=100)

    dataset_path = tmp_path / "bert.head.10.txt"
    dataset_path = str(dataset_path)
    with open(dataset_path, "w+") as dataset_file:
        dataset_file.write("\n".join(BERT_SAMPLES))

    batch_size = 2
    features, tokens, labels = mlm_loader.load(dataset_path, batch_size=batch_size)
    for features_batch, tokens_batch, labels_batch in zip(features, tokens, labels):
        assert len(features_batch) <= batch_size
        assert len(tokens_batch) <= batch_size
        assert len(labels_batch) <= batch_size
