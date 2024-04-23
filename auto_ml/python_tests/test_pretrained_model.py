import pytest
from thirdai import bolt, dataset

pytestmark = [pytest.mark.unit]


def build_dummy_pretrained_model(two_layer):
    input_ = bolt.nn.Input(dim=500)
    out = bolt.nn.Embedding(dim=100, input_dim=input_.dim(), activation="relu")(input_)

    if two_layer:
        out = bolt.nn.FullyConnected(
            dim=200, input_dim=out.dim(), activation="softmax"
        )(out)

    loss = bolt.nn.losses.CategoricalCrossEntropy(out, bolt.nn.Input(out.dim()))
    model = bolt.nn.Model(inputs=[input_], outputs=[out], losses=[loss])

    index = dataset.MachIndex(output_range=100, num_hashes=1, num_elements=20, seed=14)

    tokenizer = dataset.CharKGramTokenizer(4)

    return bolt.PretrainedBase(
        "text", models=[model], indexes=[index], tokenizer=tokenizer, lowercase=True
    )


@pytest.mark.parametrize("two_layer", [True, False])
@pytest.mark.parametrize("emb_only", [True, False])
def test_udt_with_pretrained(two_layer, emb_only):
    model = bolt.UniversalDeepTransformer(
        data_types={
            "category": bolt.types.categorical(),
            "text": bolt.types.text(),
        },
        target="category",
        pretrained_model=build_dummy_pretrained_model(two_layer=two_layer),
        integer_target=True,
        n_target_classes=25,
        options={"emb_only": emb_only},
    )

    assert len(model._get_model().ops()) == (3 if two_layer and not emb_only else 2)

    model.predict({"text": "hello"})
