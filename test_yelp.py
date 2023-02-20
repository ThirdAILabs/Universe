import pandas as pd
from thirdai import bolt

all_data_udt = pd.read_csv("all.csv")
numerical_col_ranges = all_data_udt.agg([min, max]).T.values.tolist()
numerical_col_names = ["col_" + str(i) for i in range(32)]

model = bolt.models.UDTGraphNetwork(
    data_types={
        "node_id": bolt.types.node_id(),
        **{
            col_name: bolt.types.numerical(col_range)
            for col_range, col_name in zip(numerical_col_ranges, numerical_col_names)
        },
        "target": bolt.types.categorical(),
        "neighbors": bolt.types.neighbors(),
    },
    target="target",
    n_target_classes=2,
    integer_target=True,
)

model.index("all_no_labels.csv")

model.train("train.csv", learning_rate=0.001, metrics=["categorical_accuracy"])
