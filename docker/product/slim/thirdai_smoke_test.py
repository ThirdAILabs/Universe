import pytest


def test_bolt_smoke():
    from thirdai import bolt, search

    bolt.graph.Model(inputs=[], output=None)


def test_docsearch_smoke():
    from thirdai import search

    search.DocRetrieval(
        centroids=[[0.0]], hashes_per_table=1, num_tables=1, dense_input_dimension=1
    )


def test_should_fail():
    with pytest.raises(Exception):
        import thisshouldfail
