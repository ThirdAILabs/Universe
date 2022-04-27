import pytest


def test_bolt_smoke():
    from thirdai import bolt, search

    bolt.Network(
        layers=[
            bolt.FullyConnected(dim=256, activation_function=bolt.ActivationFunctions.ReLU)
        ],
        input_dim=10,
    )


def test_docsearch_smoke():
    from thirdai import search

    search.DocRetrieval(
        centroids=[[0.0]], hashes_per_table=1, num_tables=1, dense_input_dimension=1
    )


def test_should_fail():
    with pytest.raises(Exception):
        import thisshouldfail
