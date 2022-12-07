def this_should_require_a_license_bolt():

    from thirdai import bolt

    bolt.UniversalDeepTransformer(
        data_types={"col": bolt.types.categorical()}, target="col", n_target_classes=1
    )
