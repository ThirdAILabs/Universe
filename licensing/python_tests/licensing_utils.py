import os

import pandas as pd
import pytest


# This function builds a UDT model, trains the model, saves it, and run a
# prediction on it. It is primarily used for testing licenses, and the arguments
# allow the training to be tweaked to fit within more restrictive licenses.
def run_udt_training_routine(do_save_load=True, n_classes=2, num_data_points=2):
    from thirdai import bolt

    model = bolt.UniversalDeepTransformer(
        data_types={
            "col_1": bolt.types.categorical(n_classes=n_classes),
            "col_2": bolt.types.categorical(),
        },
        target="col_1",
    )

    df = pd.DataFrame(
        {
            "col_1": [i % n_classes for i in range(num_data_points)],
            "col_2": [i % n_classes for i in range(num_data_points)],
        }
    )
    df.to_csv("temp_training.csv")

    model.train("temp_training.csv")

    if do_save_load:
        model.save("temp_save_loc")

        model = bolt.UniversalDeepTransformer.load("temp_save_loc")

        os.remove("temp_save_loc")

    model.predict({"col_2": "0"})

    os.remove("temp_training.csv")


@pytest.fixture(autouse=True)
def deactivate_license_at_start_of_demo_test():
    import thirdai

    thirdai.licensing.deactivate()


LOCAL_HEARTBEAT_SERVER = f"http://localhost:50421"
