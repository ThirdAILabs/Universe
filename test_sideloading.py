def generate_text_classification_dataset(filename, delim):
    with open(filename, "w") as f:
        for i in range(15_000):
            sentiment = i % 3
            if sentiment == 0:
                f.write(f"1{delim}good stuff\n")
            elif sentiment == 1:
                f.write(f"0{delim}bad stuff\n")
            else:
                f.write(f"2{delim}neutral stuff\n")


from thirdai.dataset import (
    BlockList,
    DatasetLoader,
    FileDataSource,
    TabularFeaturizer,
    blocks,
)

filename = "test_text_classification.csv"
text_block = blocks.TextPairGram(col=1)
delim = ","
generate_text_classification_dataset(filename, delim)
featurizer = TabularFeaturizer(
    block_lists=[
        BlockList([text_block]),
        BlockList([blocks.NumericalId(col=0, n_classes=3)]),
    ],
    delimiter=delim,
)
pipeline = DatasetLoader(
    data_source=FileDataSource(filename), featurizer=featurizer, shuffle=True
)
[data, labels] = pipeline.load_all(batch_size=256)

from thirdai import bolt

first_input_layer = bolt.nn.Input(dim=pipeline.get_input_dim())
first_hidden_layer = bolt.nn.FullyConnected(dim=1000, sparsity=0.1, activation="relu")(
    first_input_layer
)
first_output_layer = bolt.nn.FullyConnected(dim=3, activation="softmax")(
    first_hidden_layer
)

first_model = bolt.nn.Model(inputs=[first_input_layer], output=first_output_layer)
first_model.compile(bolt.nn.losses.CategoricalCrossEntropy())


second_input_layer = bolt.nn.Input(dim=pipeline.get_input_dim())
second_hidden_layer = first_model.get_layer("fc_1").clone_for_param_sharing()(
    second_input_layer
)
second_output_layer = first_model.get_layer("fc_2").clone_for_param_sharing()(
    second_hidden_layer
)

second_model = bolt.nn.Model(inputs=[second_input_layer], output=second_output_layer)
second_model.compile(bolt.nn.losses.CategoricalCrossEntropy())

train_cfg = bolt.TrainConfig(learning_rate=0.001, epochs=1).silence()
eval_config = (
    bolt.EvalConfig()
    .with_metrics(["categorical_accuracy"])
    .return_activations()
    .silence()
)

first_model.train(data, labels, train_cfg)
first_results = first_model.evaluate(data, labels, eval_config)[1]

second_results = second_model.evaluate(data, labels, eval_config)[1]
assert first_results != second_results

second_model.get_layer("fc_1").use_params(first_model.get_layer("fc_1"))
second_model.get_layer("fc_2").use_params(first_model.get_layer("fc_2"))

second_results_after_param_sharing = second_model.evaluate(data, labels, eval_config)[1]
assert all(first_results == second_results_after_param_sharing)
