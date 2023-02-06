from thirdai import bolt
import pandas as pd
from sklearn.metrics import roc_auc_score
import pickle
import pdb
df = pd.read_csv('processed_test.csv')
y_test = df['target'].tolist()

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
#     graph_file_name="/share/data/capillary/graph_total.csv",
#     source="node_id",
#     target="target",
#     relationship_columns=["rel_1", "rel_2"],
#     n_target_classes=2,
#     neighbourhood_context=True,
#     kth_neighbourhood=1,
#     label_context=True,
# )

cols = []
for i in range(32):
    cols.append("col_{}".format(i))

mins = []
maxs = []
total_df = pd.read_csv('/share/data/capillary/processed_32.csv')
for i in cols:
    mins.append(total_df[i].min())
    maxs.append(total_df[i].max())

dic = {}

for i in range(len(cols)):
    dic[cols[i]] = bolt.types.numerical(range=(mins[i],maxs[i]))


data = pd.read_pickle('/share/shubh/current_project/experiments-misc-shubh/swiggy_graphs/graph_datasets/yelp_fraud/yelpcareprocessed/yelp_homo_adjlists.pickle')


adj = {}
for key in data.keys():
    data[key].remove(key)
    adj[key] = list(data[key])

model = bolt.UniversalDeepTransformer(
    data_types = {
        "node_id":bolt.types.categorical(),
        **dic,
        "target":bolt.types.categorical(),
    },
    graph_file_name='/share/data/capillary/processed_32.csv',
    source="node_id",
    target="target",
    n_target_classes=2,
    neighbourhood_context=True,
    kth_neighbourhood=1,
    adj_list=adj,
)
# pdb.set_trace()

# model.train(
#     filename="/share/data/capillary/train_total.csv", epochs=1,
# )

# activations = model.evaluate(filename="/share/data/capillary/test_total.csv", metrics=["categorical_accuracy"])

# print(activations)

# print(roc_auc_score(y_test,activations[:,0]))

# print(roc_auc_score(y_test,activations[:,1]))



