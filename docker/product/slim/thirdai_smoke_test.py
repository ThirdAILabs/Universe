import pytest


def test_smoke():
    import thirdai
    from thirdai import bolt, search
    from thirdai.search import MagSearch, DocRetrieval

    with pytest.raises(Exception):
        import thisshouldfail
