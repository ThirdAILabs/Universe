Using the DeploymentConfig and ModelPipeline
============================================

The `ModelPipeline` is a wrapper around a Bolt DAG model and a dataset pipeline that 
abstracts the process of loading and featurizing a dataset and training a model into
a single interface that hides the complexity from the user. Rather than take in a
`BoltDataset`, it takes in raw files/DataLoaders and hides the data processing step.

The `DeploymentConfig` is a way of specifying the architecture and parameters of a 
`ModelPipeline` while allowing us to hide architecture decisions from customers, 
while also still allowing the user to input some hyperparameters to the model and/or 
dataset pipeline. 

Defining a DeploymentConfig
---------------------------
1. First define the config for the model:

>>> model_config = deployment.ModelConfig(
        input_names=["input"],
        nodes=[
            deployment.FullyConnectedNodeConfig(
                name="hidden",
                dim=deployment.OptionMappedParameter(
                    option_name="size", values={"small": 100, "large": 200}
                ),
                activation=deployment.ConstantParameter("relu"),
                predecessor="input",
            ),
            deployment.FullyConnectedNodeConfig(
                name="output",
                dim=deployment.UserSpecifiedParameter("output_dim", type=int),
                sparsity=deployment.ConstantParameter(1.0),
                activation=deployment.ConstantParameter("softmax"),
                predecessor="hidden",
            ),
        ],
        loss=bolt.CategoricalCrossEntropyLoss(),
    )

2. Then define the config for the dataset loader:

>>> dataset_config = deployment.SingleBlockDatasetFactory(
        data_block=deployment.TextBlockConfig(use_pairgrams=True),
        label_block=deployment.NumericalCategoricalBlockConfig(
            n_classes=deployment.UserSpecifiedParameter("output_dim", type=int),
            delimiter=deployment.ConstantParameter(","),
        ),
        shuffle=deployment.ConstantParameter(False),
        delimiter=deployment.UserSpecifiedParameter("delimiter", type=str),
    )

3. Then define the `TrainEvalParameters` (these are additional parameters that the 
   model uses for training/evaluation):

>>> train_eval_params = deployment.TrainEvalParameters(
        rebuild_hash_tables_interval=None,
        reconstruct_hash_functions_interval=None,
        default_batch_size=256,
        freeze_hash_tables=True,
    )

4. Now we can construct the complete deployment config:

>>> config = deployment.DeploymentConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        train_eval_parameters=train_eval_params,
    )

Creating a ModelPipeline
------------------------

The `ModelPipeline` can be constructed by passing in the path to a serialized `DeploymentConfig`
config or by passing in the DeploymentConfig directly. 

1. Save the `DeploymentConfig`.

>>> config.save("./saved_config")

2. To construct the `ModelPipeline` we need to pass in the saved config as well as
   the values needed for any `UserSpecifiedParameter` or `OptionMappedParameter`.

>>> model = bolt.models.Pipeline(
        config_path="./saved_config",
        parameters={"size": "large", "output_dim": num_classes, "delimiter": ","},
    ) 

Training with the ModelPipeline
-------------------------------

Training is similar to the regular Bolt API where you pass in `TrainConfig` that allows
specification of various hyperparameters and options. 

>>> train_config = bolt.TrainConfig(epochs=5, learning_rate=0.01)

Optionally you can also specify options like metrics or validation. The `ModelPipeline`
has a method to load a validation dataset since it contains the dataset loader functionality.

>>> train_config = train_config.with_metrics(["mean_squared_error"])

>>> val_data, val_labels = trained_text_classifier.load_validation_data("./validation_data")
>>> validation_config = (
        bolt.EvalConfig()
        .with_metrics(["categorical_accuracy"])
    )
>>> train_config = train_config.with_validation(
        validation_data=val_data,
        validation_labels=val_labels,
        eval_config=validation_config,
        validation_frequency=10,
    )

The model will default to the `batch_size` provided in the `TrainEvalParameters` if 
not specified in the call to train. The parameter `max_in_memory_batches` is also
optional, providing it means the `ModelPipeline` will load the dataset it chunks with 
up to the specified number of batches. 

>>> model.train(
        filename="./train_data",
        train_config=train_config,
        batch_size=256,
        max_in_memory_batches=12,
    )

Evaluating with the ModelPipeline
---------------------------------

Evaulate just requires an evaluation dataset or optionally a Bolt `EvalConfig` 
if you would like to specify metrics or sparse inference. It returns the activations
from the final layer of the model.

>>> eval_config = (
        bolt.EvalConfig()
        .with_metrics(["categorical_accuracy"])
        .enable_sparse_inference()
    )
>>> activations = model.evaluate(
        filename="./test_data", eval_config=eval_config
    )

Inference with the ModelPipeline
--------------------------------

The model pipeline has methods for `predict`, `predict_batch`, and `predict_tokens`. 
These methods take in unlabeled samples and return activations. Whether to use sparse 
inference is an optional parameter (default is False). 

>>> output = model.predict("some single sample", use_sparse_inference=True)

>>> outputs = model.predict_batch(["some sample", "another sample"])

>>> output = model.precict_tokens([14, 92, 74, 46])

Saving & Loading the ModelPipeline
----------------------------------

The model pipeline supports save/load functions just like most of our library. 

>>> model.save("./saved_model_pipeline")
>>> new_model = bolt.models.Pipeline.load("./saved_model_pipeline")