from thirdai import bolt
import pandas as pd
from sklearn.metrics import roc_auc_score
import pickle
import pdb

graph_file = "/share/data/capillary/tfinance.csv"
train_file = "/share/data/capillary/tfinance_train.csv"
test_file = "/share/data/capillary/tfinance_test.csv"
num_numerical = 10
adj_pickle_file = "/share/shubh/data/tfinance/adjlist.pickle"
target = "label"

# graph_file = "/share/data/capillary/processed_32.csv"
# train_file = "processed_train.csv"
# test_file = "processed_test.csv"
# num_numerical = 32
# adj_pickle_file = "/share/shubh/current_project/experiments-misc-shubh/swiggy_graphs/graph_datasets/yelp_fraud/yelpcareprocessed/yelp_homo_adjlists.pickle"

df = pd.read_csv(test_file)
y_test = df[target].tolist()

# model = bolt.UniversalDeepTransformer(
#     data_types={
#         "node_id": bolt.types.categorical(),
#         "rel_1": bolt.types.categorical(),
#         "rel_2": bolt.types.categorical(),
#         "target": bolt.types.categorical(),
#         # "n1": bolt.types.numerical(range=(0, 48)),
#         # "n2": bolt.types.numerical(range=(0, 44)),
#         # "n3": bolt.types.numerical(range=(0, 42)),
#         "n1": bolt.types.numerical(range=(0, 78)),
#         "n2": bolt.types.numerical(range=(0, 170)),
#         "n3": bolt.types.numerical(range=(0, 84)),
#         "n4": bolt.types.categorical(),
#     },
#     graph_file_name="/share/data/capillary/graph2.csv",
#     source="node_id",
#     target="target",
#     relationship_columns=["rel_1", "rel_2"],
#     n_target_classes=2,
#     max_neighbours=10,
#     numerical_context=True,
#     k_hop=1,
# )

cols = []
for i in range(num_numerical):
    cols.append("feature_{}".format(i))

mins = []
maxs = []
total_df = pd.read_csv(graph_file)
for i in cols:
    mins.append(total_df[i].min())
    maxs.append(total_df[i].max())

dic = {}

for i in range(len(cols)):
    dic[cols[i]] = bolt.types.numerical(range=(mins[i],maxs[i]))


data = pd.read_pickle(adj_pickle_file)


adj = {}
for key in data.keys():
    # data[key].remove(key)
    adj[str(key)] = list(map(str,list(data[key])))
    # break
# adj = {0:[1,2],1:[2,0]}

# print(adj)

model = bolt.UDTGraph(
    data_types={
        "node_id": bolt.types.categorical(),
        **dic,
        target: bolt.types.categorical(),
    },
    graph_file_name=graph_file,
    source="node_id",
    target=target,
    n_target_classes=2,
    max_neighbours=25,
    numerical_context=True,
    k_hop=1,
    adj_list=adj,
)
# # pdb.set_trace()
for i in range(15):
    model.train(
        filename=train_file, epochs=1,learning_rate=0.001,metrics=["categorical_accuracy"],
    )

    activations = model.evaluate(filename=test_file, metrics=["categorical_accuracy"])

    print(activations)

    print(roc_auc_score(y_test,activations[:,0]))

    print(roc_auc_score(y_test,activations[:,1]))



