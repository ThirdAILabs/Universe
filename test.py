from thirdai import bolt

model = bolt.UniversalDeepTransformer(
    data_types={
        "text": bolt.types.categorical(delimiter=" "),
        "category": bolt.types.categorical(),
    },
    target="category",
    n_target_classes=150,
    integer_target=True,
)

model.train("./test/clinc_train.csv", learning_rate=0.01, epochs=5)

model.evaluate("./test/clinc_test.csv", metrics=["categorical_accuracy"])
