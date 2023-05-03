import pytest
from thirdai.dataset import Wordpiece

from conftest import download_bert_tokenizer

BERT_TOKENIZED_SAMPLES = [
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

BERT_RAW_SAMPLES = [
    "popularity of threading has increased around 2003, as the growth of the cpu frequency was replaced with the growth of number of cores, in turn requiring concurrency to utilize multiple cores.",
    "arkansas highway 59 business is a business route in gentry.",
    "before joining city of hope, rosen worked at northwestern university.",
    "the third section (8 km) was the bypass of szczuczyn opened in november 2015 as a single carriageway road.",
    "in may 2021, the company left the swedish market due to a “unfair and discriminatory treatment” by the swedish tax agency.",
    "since 2018, the winner of the bonaire kampionato, the top tier of football on the island qualifies for the caribbean club shield, a tertiary knockout tournament for developing caribbean football nations.",
    "the tone of linlithgow's warnings to amery grew increasingly serious over the first half of 1943, as did amery's requests to the war cabinet; on 4august 1943 amery noted the spread of famine, and specifically stressed the effect upon calcutta and the potential effect on the morale of european troops.",
    "during the first years of japanese occupation following the first sino-japanese war, the political divisions of the island were changed frequently.",
    "october 27 – martin mullen, 63, right fielder in one game for the 1872 cleveland forest citys of the national association.",
    "morata de tajuna is a municipality of the community of madrid, spain.",
]


@pytest.mark.unit
def test_wordpiece_vocab(download_bert_tokenizer):
    BERT_VOCAB_PATH = download_bert_tokenizer
    vocab = Wordpiece(BERT_VOCAB_PATH)

    with open(BERT_VOCAB_PATH) as vocab_file:
        lines = vocab_file.read().splitlines()
        assert len(lines) == vocab.size()

    for raw, tokenized in zip(BERT_RAW_SAMPLES, BERT_TOKENIZED_SAMPLES):
        token_ids = vocab.tokenize(raw)

        assert vocab.unk_id() not in token_ids

        tokens = tokenized.split()
        assert len(token_ids) == len(tokens)

        # Assert piece level reconstruction.
        for token_id, token in zip(token_ids, tokens):
            assert vocab.id(token) == token_id

        # Assert sentence-level reconstruction
        decoded = vocab.decode(token_ids)
        assert decoded == tokenized.replace(" ##", "")
