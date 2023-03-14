import pytest
from thirdai import bolt

pytestmark = [pytest.mark.unit]

# integration test on scifact dataset with coldstart reaches good accuracy


model = bolt.UniversalDeepTransformer(
    data_types={
        "category": bolt.types.categorical(),
        "text": bolt.types.text(),
    },
    target="category",
    n_target_classes=150,
    integer_target=True,
)

# test integer and string target?
# test the decoding -> c++ test?
# test save load
# test embedding
# test entity embedding
