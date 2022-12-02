import thirdai
thirdai.activate("94TN-9LUT-KXWK-K4VE-CPEW-3U9K-3R7H-HREL")

from thirdai import bolt

bolt.UniversalDeepTransformer(
    data_types={"col": bolt.types.categorical()}, target="col", n_target_classes=1
)

thirdai.activate("blarg")

bolt.UniversalDeepTransformer(
    data_types={"col": bolt.types.categorical()}, target="col", n_target_classes=1
)