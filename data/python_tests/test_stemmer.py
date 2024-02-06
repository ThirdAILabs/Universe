import pandas as pd
import pytest
from download_dataset_fixtures import download_scifact_dataset
from nltk.stem import PorterStemmer
from thirdai import data


def test_stemmer_different_suffixes():
    # These cases are from the examples mentioned in the paper to explain the
    # different types of stemming that can be applied.
    cases = (
        "caresses ponies ties flies caress cats feed agreed plastered bled "
        "motoring sing conflated troubled sized hopping tanned falling hissing "
        "fizzed failing filing died lied happy die say relational conditional "
        "rational valency hesitancy digitizer conformably radically differently "
        "vilely analogousli vietnamization predication operator feudalism "
        "decisiveness hopefulness callousness formality sensitivity sensibility "
        "triplicate formative formalize electricity electrical hopeful goodness "
        "revival allowance inference airliner gyroscopic adjustable defensible "
        "irritant replacement adjustment dependent adoption communism activate "
        "angularities homologous effective bowdlerize probate rate cease control "
        "roll skies sky dying lying tying news innings inning outings inning "
        "cannings canning howe proceed exceed succeed"
    )

    stemmer = PorterStemmer()
    for word in cases.split():
        assert stemmer.stem(word) == data.stem(word)


@pytest.mark.unit
@pytest.mark.parametrize("lowercase", [True, False])
def test_stemmer_lowercase(lowercase):
    stemmer = PorterStemmer()

    for word in ["Running", "Glided", "Differently"]:
        assert stemmer.stem(word, to_lowercase=True) == data.stem(word, lowercase=True)


@pytest.mark.unit
def test_stemmer_scifact(download_scifact_dataset):
    docs, _, _, _ = download_scifact_dataset

    df = pd.read_csv(docs)
    df["TEXT"] = df["TEXT"].str.lower().map(lambda x: x.split())

    stemmer = PorterStemmer()
    nltk_stemmer = df["TEXT"].map(lambda s: [stemmer.stem(w) for w in s])

    thirdai_stemmer = df["TEXT"].map(lambda s: data.stem(s))

    for ns, ts in zip(nltk_stemmer, thirdai_stemmer):
        assert ns == ts
