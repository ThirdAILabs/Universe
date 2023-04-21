import os
import random
from collections import defaultdict

import pytest
from thirdai import bolt


def test_udt_load_old_version():

    model = bolt.UniversalDeepTransformer(
        data_types={
            "Sample": bolt.types.text(),
            "Target": bolt.types.categorical(),
        },
        target="Target",
        n_target_classes=10,
    )

    model.save("./udt_models/udt_base_version_neg1.model")


test_udt_load_old_version()