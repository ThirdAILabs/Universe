import pytest
from thirdai.search import InvertedIndex, KgramTokenizer


@pytest.mark.unit
def test_inverted_index_kgram_tokenizer():
    """This is a sanity check to make sure nothing breaks. Detailed behavioral
    tests are written in C++.
    """

    index = InvertedIndex(tokenizer=KgramTokenizer())
    index.index(ids=[0, 1], docs=["Numero uno", "Numero dos"])
    assert index.query(query="Numero uno", k=1)[0][0] == 0


