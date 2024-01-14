from thirdai import bolt, demos

model = bolt.UniversalDeepTransformer(
    data_types={
        "category": bolt.types.categorical(),
        "text": bolt.types.text(),
    },
    target="category",
    n_target_classes=150,
    integer_target=True,
)


train_data, test_data, _ = demos.download_clinc_dataset()

model.train(
    train_data,
    epochs=5,
    learning_rate=0.01,
    validation=bolt.Validation(test_data, ["categorical_accuracy"]),
)


model.save_cpp_classifier("./udt_classifier")
