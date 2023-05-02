import os

import pytest
from thirdai.dataset import FixedVocabulary, Wordpiece

pytestmark = [pytest.mark.unit]

BERT_TAG = "bert-base-uncased"
BERT_VOCAB_PATH = "{}.vocab".format(BERT_TAG)
BERT_VOCAB_URL = "https://huggingface.co/{}/resolve/main/vocab.txt".format(BERT_TAG)

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


# These list failures on wikipedia dataset from /share1. These are not strictly
# failures - as the tokenization works, but behaves differently from
# HuggingFace tokenizer. These are 37 failures out of 40M samples - so should
# not affect out general training as much.
WIKIPEDIA_CORNER_CASES = [
    "The Unicode symbols for a Quran verse, including U+06DD (۝), and U+08E2 (࣢  ).",
    "1925: Tönnies’s major writings collected in Soziologische Studien und Kritiken (  vols.).",
    "Industry in Skorupy appeared already in the 1860s together with the construction of a textile plant by Albrecht Adolfai Scharlotta Zoi from Riedlów Reich from the Polish Kingdom.",
    "1,186,060,307,891,929,990 takes 261 iterations to reach the 119-digit palindrome 44562665878976437622437848976653870388884783662598425855963436955852489526638748888307835667984873422673467987856626544, which was a former world record for the Most Delayed Palindromic Number.",
    'Publishing Triangle named Bastard Out of Carolina one of "The Triangle’s 100 Best " novels of the 1990s.',
    "In his comedy Assemblywomen (c. 392 BC), Aristophanes coined the 175-letter word  (lopado­temacho­selacho­galeo­kranio­leipsano­drim­hypo­trimmato­silphio­karabo­melito­katakechy­meno­kichl­epi­kossypho­phatto­perister­alektryon­opte­kephallio­kigklo­peleio­lagoio­siraio­baphe­tragano­                            pterygon), a fictional food dish consisting of a combination of fish and other meat.",
    "1994 – [SEP] Sephardic Songs in the Hispano-Arabic tradition of medieval Spain. Ballads of the Sephardic Jews).",
    "1, 105263157894736842, 1034482758620689655172413793, 102564, 102040816326530612244897959183673469387755, 1016949152542372881355932203389830508474576271186440677966, 1014492753623188405797, 1012658227848, 10112359550561797752808988764044943820224719, 10, 100917431192660550458715596330275229357798165137614678899082568807339449541284403669724770642201834862385321, 100840336134453781512605042016806722689075630252, ...",
    "He is also the  Chairman of the Narotam Sekhsaria Foundation, the philanthropic arm of his family office, which funds and supports individuals and organisations working in health, education, livelihoods, governance, art and culture.",
    " TA/DA during Inter University Tournaments.",
    " Free Sports Kit and Track Suits to the players participating in Inter University Tournaments.",
    " Affordable Fee for the students.",
    " Bilingual programmes in Hindi and English with study material in English except otherwise intimated.",
    " Distance Education by using technology in selected subjects.",
    " Standard study material.",
    " The University Centre for Distance Learning holds Personal Contact Programmes (PCP).",
    " To provide the syllabus and study material after admission of the students.",
    " Punctuality in the conduct of examinations and declaration of results.",
    " No Migration Certificate is required for taking admission in Open and Distance Learning programme of CDLU, Sirsa and no Migration Certificate will be issued by the University after the completion of the course.",
    " The UCDL has appointed well qualified teachers and they remains available in UCDL Library for student‘s related queries/doubts on all working days.",
    " The University College Library has a total number of collection of 393 books.",
    " Water coolers with RO system installed.",
    " All rooms are well furnished with furniture, lecture stand, white board, tube lights and ceiling fans.",
    " Secure and Safe environment.",
    " Lush green campus.",
    " Fully Wi-Fi Campus.",
    "Sovereign rulers (both Emperors and Kings) in general are referred to in Vietnamese as Vua (君, 𢁨, 𢂜, 𢃊, 𤤰, 𪻟, 𪼀, 󰅫, 󰅻). classical Chinese).",
    'Publishing Triangle named Aquamarine one of "The Triangle’s 100 Best " gay and lesbian novels of the 1990s.',
    "       General Orthopedic Rehab (Total Joint, Post-Shoulder Surgery, Post-Spine Surgery, etc.)",
    "All five vowels found in the Mixtepec Mixtec language have nasalized differentiating counterparts however, the mid vowels /õ/ and /ε􏰀/ are rare.",
    " The work of the early nationalists had exposed the economic exploitation of India by the British.",
    "In particular, each element in  can be written uniquely as 􏰐, where , and the product of any two secondaries is uniquely given by , where .",
    "Honda Y, Rogers L, Nakata K, Zhao B, Pine R, Nakai Y, Kurosu K, Rom WN, Weiden M.  Type I interferon induces inhibitory 16 kD CCAAT/Enhancer Binding Protein (C/EBP), repressing the HIV-1 long terminal repeat in macrophages: pulmonary tuberculosis alters C/EBP expression, enhancing HIV-1 replication.",
    " The winners must agree to take part in publicity generated by the IYC.",
    "   Used in the subcontinent, this indicates a difference of opinion on the pause.",
    "Lu demonstrated that growth factor receptor activation induces translocation of PKM2 into the nucleus, where it binds to and activates tyrosine-phosphorylated -catenin and c-Myc, resulting in expression of glycolytic genes and enhanced glucose uptake and lactate production.",
    "Hala Feb Festival  Tuesday Feb 9th 2016.",
]


def setup_module():
    if not os.path.exists(BERT_VOCAB_PATH):
        import urllib.request

        response = urllib.request.urlopen(BERT_VOCAB_URL)
        with open(BERT_VOCAB_PATH, "wb+") as bert_vocab_file:
            bert_vocab_file.write(response.read())


def test_fixed_vocab():
    vocab = FixedVocabulary.make(BERT_VOCAB_PATH)

    with open(BERT_VOCAB_PATH) as vocab_file:
        lines = vocab_file.read().splitlines()
        assert len(lines) == vocab.size()

    for sample in BERT_TOKENIZED_SAMPLES:
        pieces = vocab.encode(sample)

        # Nothing maps to unknown, as the above is taken from BERT (thirdai)
        assert vocab.unk_id() not in pieces

        # Assert piece level reconstruction.
        tokens = sample.split()
        for piece, token in zip(pieces, tokens):
            # Assert reconstruction works
            if "THIRDAI_TEST_DEBUG" in os.environ:
                print("{}: {}".format(token, piece), end=" ")

        # Assert sentence-level reconstruction
        assert vocab.decode(pieces) == sample

        if "THIRDAI_TEST_DEBUG" in os.environ:
            print()

        assert len(pieces) == len(tokens)


def test_wordpiece_vocab():
    vocab = Wordpiece.make(BERT_VOCAB_PATH)

    with open(BERT_VOCAB_PATH) as vocab_file:
        lines = vocab_file.read().splitlines()
        assert len(lines) == vocab.size()

    for raw, tokenized in zip(BERT_RAW_SAMPLES, BERT_TOKENIZED_SAMPLES):
        # sample = sample.replace(" ##", "")
        pieces = vocab.encode(raw)

        # Nothing maps to unknown, as the above is taken from BERT (thirdai)
        assert vocab.unk_id() not in pieces
        tokens = tokenized.split()

        # Assert piece level reconstruction.
        for piece, token in zip(pieces, tokens):
            # Assert reconstruction works
            if "THIRDAI_TEST_DEBUG" in os.environ:
                print("{}: {}".format(token, piece), end=" ")

            # Assert the piece matches the token from huggingface.
            assert vocab.id(token) == piece

        # Assert sentence-level reconstruction
        decoded = vocab.decode(pieces)
        assert decoded == tokenized.replace(" ##", "")

        if "THIRDAI_TEST_DEBUG" in os.environ:
            print()

        # assert len(pieces) == len(tokens)
