import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from thirdai import bolt

all_data = pd.read_csv("yelp_all.csv")
numerical_col_names = ["col_" + str(i) for i in range(32)]
numerical_col_ranges = all_data[numerical_col_names].agg([min, max]).T.values.tolist()

train_data, test_data = train_test_split(all_data, test_size=0.5)

train_data = train_data.sample(frac=1)
train_data.to_csv("yelp_train.csv", index=False)

ground_truth = test_data["target"].to_numpy()
test_data["target"] = np.zeros(len(ground_truth))
test_data.to_csv("yelp_test.csv", index=False)

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

model.index("yelp_test.csv")

for epoch in range(10):
    model.train("yelp_train.csv", learning_rate=0.001, epochs=1)
    activations = model.evaluate("yelp_test.csv")
    print(metrics.roc_auc_score(ground_truth, activations[:, 1]))
