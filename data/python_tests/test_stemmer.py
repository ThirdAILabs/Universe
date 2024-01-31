import pandas as pd
import pytest
from download_dataset_fixtures import download_scifact_dataset
from nltk.stem import PorterStemmer
from thirdai import data


@pytest.mark.unit
def test_stemmer(download_scifact_dataset):
    docs, _, _, _ = download_scifact_dataset

    df = pd.read_csv(docs)
    df["TEXT"] = df["TEXT"].map(lambda x: x.split())

    stemmer = PorterStemmer()
    nltk_stemmer = df["TEXT"].map(lambda s: [stemmer.stem(w, False) for w in s])

    thirdai_stemmer = df["TEXT"].map(lambda s: data.stem(s, False))

    for ns, ts in zip(nltk_stemmer, thirdai_stemmer):
        assert ns == ts
