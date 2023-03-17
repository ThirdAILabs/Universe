import pandas as pd
import os

def this_should_require_a_full_license_udt():
    from thirdai import bolt

    model = bolt.UniversalDeepTransformer(
        data_types={
            "col_1": bolt.types.categorical(),
            "col_2": bolt.types.categorical(),
        },
        target="col_1",
        n_target_classes=2,
    )

    df = pd.DataFrame({'col_1': [0.0, 1.0], 'col_2': [0, 1]})
    df.to_csv("temp_training.csv")

    model.train("temp_training.csv")

    model.save("temp_save_loc")

    bolt.UniversalDeepTransformer.load("temp_save_loc")

    os.remove("temp_training.csv")
    os.remove("temp_save_loc")


LOCAL_HEARTBEAT_SERVER = f"http://localhost:50421"
