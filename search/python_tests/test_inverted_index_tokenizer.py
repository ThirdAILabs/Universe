import pytest
from thirdai.search import IndexConfig, InvertedIndex, WordKGrams


@pytest.mark.unit
def test_inverted_index_kgram_tokenizer():
    """This is a sanity check to make sure nothing breaks. Detailed behavioral
    tests are written in C++.
    """

    index = InvertedIndex(IndexConfig(tokenizer=WordKGrams()))
    index.index(ids=[0, 1], docs=["Numero uno", "Numero dos"])
    assert index.query(query="Numero uno", k=1)[0][0] == 0
