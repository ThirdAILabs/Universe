from thirdai import bolt, deployment

tabular_train_file = "/Users/david/Documents/data/new_Train.csv"
tabular_test_file = "/Users/david/Documents/data/new_Test.csv"

import pandas as pd

train_df = pd.read_csv("/Users/david/Documents/data/new_Train.csv")
test_df = pd.read_csv("/Users/david/Documents/data/new_Train.csv")

numeric_column_names = [
    "age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week", 
]

for colname in numeric_column_names:
    print(f"MAX {colname}", train_df[colname].max())
    print(f"MIN {colname}", train_df[colname].min(), "\n")

# exit()

with open(tabular_train_file) as f:
    for _ in range(5):
        print(next(f))

tabular_model = deployment.UniversalDeepTransformer(
    data_types={
        # "age":bolt.types.numerical(range=(17, 90)),
        "workclass":bolt.types.categorical(n_unique_classes=9),
        # "fnlwgt":bolt.types.numerical(range=(12285, 1484705)),
        "education":bolt.types.categorical(n_unique_classes=16),
        "education-num":bolt.types.categorical(n_unique_classes=16),
        "marital-status":bolt.types.categorical(n_unique_classes=7),
        "occupation":bolt.types.categorical(n_unique_classes=15),
        "relationship":bolt.types.categorical(n_unique_classes=6),
        "race":bolt.types.categorical(n_unique_classes=5),
        "sex":bolt.types.categorical(n_unique_classes=2),
        # "capital-gain":bolt.types.numerical(range=(0, 99999)),
        # "capital-loss":bolt.types.numerical(range=(0, 4356)),
        # "hours-per-week":bolt.types.numerical(range=(1, 99)),
        "native-country":bolt.types.categorical(n_unique_classes=42),
        "label":bolt.types.categorical(n_unique_classes=2),
    },
    target="label",
)

train_config = (bolt.graph.TrainConfig
                    .make(epochs=5, learning_rate=0.001)
                    .with_metrics(["categorical_accuracy"]))

tabular_model.train(tabular_train_file, train_config)

test_config = (bolt.graph.PredictConfig.make()
                   .with_metrics(["categorical_accuracy"]))

tabular_model.evaluate(tabular_test_file, test_config)