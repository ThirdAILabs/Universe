import os

import pandas as pd


def this_should_require_a_full_license_udt(
    test_load_save=True, n_target_classes=2, num_data_points=2
):
    from thirdai import bolt

    model = bolt.UniversalDeepTransformer(
        data_types={
            "col_1": bolt.types.categorical(),
            "col_2": bolt.types.categorical(),
        },
        target="col_1",
        n_target_classes=n_target_classes,
    )

    df = pd.DataFrame(
        {
            "col_1": [i % n_target_classes for i in range(num_data_points)],
            "col_2": [i % n_target_classes for i in range(num_data_points)],
        }
    )
    df.to_csv("temp_training.csv")

    model.train("temp_training.csv")

    if test_load_save:
        model.save("temp_save_loc")

        bolt.UniversalDeepTransformer.load("temp_save_loc")

        os.remove("temp_save_loc")

    os.remove("temp_training.csv")


LOCAL_HEARTBEAT_SERVER = f"http://localhost:50421"
