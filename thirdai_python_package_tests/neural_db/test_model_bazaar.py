import pytest


@pytest.mark.unit
def test_model_bazaar():
    from thirdai import neural_db as ndb

    bazaar = ndb.Bazaar()

    bazaar_models = bazaar.list_model_names()

    model_name = "Yash/ICML"

    assert model_name in bazaar_models

    db_model = bazaar.get_neuraldb(model_name="ICML", author_username="Yash")

    results = db_model.search(
        query="some query",
        top_k=10,
    )

    assert len(results) == 10
