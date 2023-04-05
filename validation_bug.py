from thirdai import bolt
from thirdai.demos import download_beir_dataset

dataset = "scifact"
# unsup_file, sup_train_file, sup_test_file, n_target_classes = download_beir_dataset(dataset)


print("STARTING TEST\n\n\n\n")

model = bolt.UniversalDeepTransformer(
    data_types={
        "QUERY": bolt.types.text(),
        "DOC_ID": bolt.types.categorical(delimiter=':'),
    },
    target="DOC_ID",
    n_target_classes=5183,
    integer_target=True,
    # tanh config
)

validation = bolt.Validation("scifact/tst_supervised.csv", metrics=["categorical_accuracy", "recall@100"])

model.cold_start(
    filename="scifact/unsupervised.csv",
    strong_column_names=["TITLE"],
    weak_column_names=["TEXT"],
    learning_rate=0.001,
    epochs=1,
    validation=validation,
    metrics=["categorical_accuracy", "recall@100"],
)

print("EVALUATE \n\n")
activations = model.evaluate("scifact/tst_supervised.csv", metrics=['categorical_accuracy','recall@100'])
print("\n\n")

model.train(
    filename="scifact/trn_supervised.csv",
    learning_rate=0.001,
    epochs=1,
    validation=validation,
    metrics=["categorical_accuracy", "recall@100"],
)

activations = model.evaluate("scifact/tst_supervised.csv", metrics=['categorical_accuracy','recall@100'])