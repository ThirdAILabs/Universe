from thirdai import bolt
from thirdai.demos import download_beir_dataset

dataset = "scifact"
unsup_file, sup_train_file, sup_test_file, n_target_classes = download_beir_dataset(dataset)

model = bolt.UniversalDeepTransformer(
    data_types={
        "QUERY": bolt.types.text(),
        "DOC_ID": bolt.types.categorical(delimiter=':'),
    },
    target="DOC_ID",
    n_target_classes=n_target_classes,
    integer_target=True,
    # tanh config
)

validation = bolt.Validation(sup_test_file, metrics=["categorical_accuracy", "recall@100"])

model.cold_start(
    filename=unsup_file,
    strong_column_names=["TITLE"],
    weak_column_names=["TEXT"],
    learning_rate=0.001,
    epochs=2,
    validation=validation,
    metrics=["categorical_accuracy", "recall@100"],
)

activations = model.evaluate(sup_test_file, metrics=['categorical_accuracy','recall@100'])

model.train(
    filename=sup_train_file,
    learning_rate=0.001,
    epochs=5,
    validation=validation,
    metrics=["categorical_accuracy", "recall@100"],
)

activations = model.evaluate(sup_test_file, metrics=['categorical_accuracy','recall@100'])