import pdb
import pickle

import pandas as pd
from sklearn.metrics import roc_auc_score
from thirdai import bolt

df = pd.read_csv("/share/data/capillary/processed_test.csv")
y_test = df["target"].tolist()

cols = []
for i in range(32):
    cols.append("col_{}".format(i))

mins = []
maxs = []
total_df = pd.read_csv("/share/data/capillary/processed_32.csv")
for i in cols:
    mins.append(total_df[i].min())
    maxs.append(total_df[i].max())

dic = {}

for i in range(len(cols)):
    dic[cols[i]] = bolt.types.numerical(range=(mins[i], maxs[i]))


data = pd.read_pickle(
    "/share/shubh/current_project/experiments-misc-shubh/swiggy_graphs/graph_datasets/yelp_fraud/yelpcareprocessed/yelp_homo_adjlists.pickle"
)


adj = {}
for key in data.keys():
    data[key].remove(key)
    adj[str(key)] = list(map(str, list(data[key])))

model = bolt.UDTGraph(
    data_types={
        "node_id": bolt.types.categorical(),
        **dic,
        "target": bolt.types.categorical(),
    },
    graph_file_name="/share/data/capillary/processed_32.csv",
    source="node_id",
    target="target",
    n_target_classes=2,
    max_neighbours=1,
    numerical_context=True,
    k_hop=1,
    adj_list=adj,
)

for i in range(5):
    model.train(
        filename="/share/data/capillary/processed_train.csv",
        epochs=1,
        learning_rate=0.001,
        metrics=["categorical_accuracy"],
    )

    activations = model.evaluate(
        filename="/share/data/capillary/processed_test_0.csv",
        metrics=["categorical_accuracy"],
    )

    print(activations)

    print(roc_auc_score(y_test, activations[:, 0]))

    print(roc_auc_score(y_test, activations[:, 1]))
