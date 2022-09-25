from utils import decide_datasets_to_run, get_col_datatypes
from thirdai import bolt, dataset
from thirdai.dataset import TabularDataType, DataPipeline, blocks
from pandas.api.types import is_numeric_dtype
import pandas as pd
import time
import os
from dataclasses import dataclass
from typing import Dict, Any

# TODO try freeze hash tables after N epochs
# TODO try varying the input dimension
# TODO try going to the max int token dimension for embedding layer and not modding the pairgrams
# TODO try not deduplicating pairgrams and concatenating the output (since we'll then have fixed number of pairgrams)


def get_tabular_col_datatypes(data_dir):
    tabular_dtypes = []
    for dtype in get_col_datatypes(data_dir):
        if dtype == "label":
            tabular_dtypes.append(TabularDataType.Label)
        if dtype == "categorical":
            tabular_dtypes.append(TabularDataType.Categorical)
        if dtype == "numeric":
            tabular_dtypes.append(TabularDataType.Numeric)
    return tabular_dtypes


def get_pandas_datasets(data_dir, tabular_dtypes):
    train_data = pd.read_csv(data_dir + "/Train.csv", header=None)
    label_col_index = tabular_dtypes.index(TabularDataType.Label)
    train_label_col = train_data.columns[label_col_index]
    xtrain, ytrain = (
        train_data.drop(train_label_col, axis=1),
        train_data[train_label_col],
    )

    test_data = pd.read_csv(data_dir + "/Test.csv", header=None)
    num_test_rows = test_data.shape[0]
    num_classes = ytrain.nunique()

    return xtrain, ytrain, num_test_rows, num_classes


def make_tabular_metadata(pandas_xtrain, pandas_ytrain, tabular_dtypes):
    col_min_maxes = {}
    for i, colname in enumerate(pandas_xtrain.columns):
        if is_numeric_dtype(pandas_xtrain[colname]):
            col_min_maxes[i] = [
                pandas_xtrain[colname].min(),
                pandas_xtrain[colname].max(),
            ]

    class_name_to_id = {}
    uid = 0
    for val in pandas_ytrain:
        if str(val) not in class_name_to_id:
            class_name_to_id[str(val)] = uid
            uid += 1

    return dataset.TabularMetadata(
        column_dtypes=tabular_dtypes,
        col_min_maxes=col_min_maxes,
        class_name_to_id=class_name_to_id,
    )


def featurize_bolt_tabular_data(
    pandas_xtrain, pandas_ytrain, tabular_dtypes, input_dim, data_dir
):
    metadata = make_tabular_metadata(pandas_xtrain, pandas_ytrain, tabular_dtypes)

    data_labels = []
    for filename in ["/Train.csv", "/Valid.csv", "/Test.csv"]:
        pipeline = DataPipeline(
            filename=data_dir + filename,
            input_blocks=[blocks.TabularPairGram(metadata, input_dim)],
            label_blocks=[
                blocks.StringId(
                    col=tabular_dtypes.index(TabularDataType.Label),
                    vocab=metadata.get_class_to_id_map(),
                )
            ],
            batch_size=256,
        )
        data, labels = pipeline.load_in_memory()
        data_labels.append(data)
        data_labels.append(labels)

    return data_labels


def generate_run_configs():
    return [
        {
            "name": "tiny_model",
            "num_embedding_lookups": 8,
            "lookup_size": 8,
            "log_embedding_block_size": 10,
            "reduction": "sum",
        },
        {
            "name": "small_model",
            "num_embedding_lookups": 8,
            "lookup_size": 8,
            "log_embedding_block_size": 15,
            "reduction": "sum",
        },
        {
            "name": "medium_model",
            "num_embedding_lookups": 8,
            "lookup_size": 8,
            "log_embedding_block_size": 20,
            "reduction": "sum",
        },
        {
            "name": "large_model",
            "num_embedding_lookups": 8,
            "lookup_size": 8,
            "log_embedding_block_size": 25,
            "reduction": "sum",
        },
        {
            "name": "medium_higher_lookup_size",
            "num_embedding_lookups": 8,
            "lookup_size": 16,
            "log_embedding_block_size": 20,
            "reduction": "sum",
        },
    ]


def make_bolt_embedding_model(input_dim, run_config, num_classes):
    input_layer = bolt.graph.Input(dim=input_dim)

    embedding_layer = bolt.graph.Embedding(
        num_embedding_lookups=run_config["num_embedding_lookups"],
        lookup_size=run_config["lookup_size"],
        log_embedding_block_size=run_config["log_embedding_block_size"],
        reduction=run_config["reduction"],
    )(input_layer)

    output_layer = bolt.graph.FullyConnected(dim=num_classes, activation="softmax")(
        embedding_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model


def execute_run(
    run_config,
    xtrain,
    ytrain,
    xtest,
    ytest,
    dataset_name,
    base_train_config,
    base_predict_config,
    num_test_rows,
    input_dim,
    num_classes,
):
    mlflow_callback = bolt.MlflowCallback(
        tracking_uri="http://deplo-mlflo-15qe25sw8psjr-1d20dd0c302edb1f.elb.us-east-1.amazonaws.com",
        experiment_name="Tabular Architecture Sweep",
        run_name=run_config["name"],
        dataset_name=dataset_name,
        experiment_args=run_config,
    )

    train_config = base_train_config.with_callbacks(
        [
            mlflow_callback,
            bolt.graph.callbacks.EarlyStopCheckpoint(
                monitored_metric="categorical_accuracy",
                model_save_path="saveLoc",
                patience=3,
                min_delta=0,
            ),
        ]
    )

    model = make_bolt_embedding_model(input_dim, run_config, num_classes=num_classes)

    start_train = time.perf_counter()
    model.train(xtrain, ytrain, train_config)
    end_train = time.perf_counter()
    mlflow_callback.log_additional_metric(
        "total_training_time", end_train - start_train
    )

    model = bolt.graph.Model.load("saveLoc")
    start_predict = time.perf_counter()
    best_test_accuracy = model.predict(xtest, ytest, base_predict_config)[0][
        "categorical_accuracy"
    ]
    end_predict = time.perf_counter()
    mlflow_callback.log_additional_metric("final_test_accuracy", best_test_accuracy)
    mlflow_callback.log_additional_metric(
        "average_inference_time", (end_predict - start_predict) / num_test_rows
    )

    size = os.path.getsize("saveLoc")
    mlflow_callback.log_additional_metric("serialized_model_size", size)

    mlflow_callback.end_run()


def main():
    datasets = decide_datasets_to_run()

    base_dir = "/share/data/tabular_benchmarks/"
    for dataset_name in datasets:
        data_dir = base_dir + dataset_name
        tabular_dtypes = get_tabular_col_datatypes(data_dir)

        pandas_xtrain, pandas_ytrain, num_test_rows, num_classes = get_pandas_datasets(
            data_dir, tabular_dtypes
        )

        print(
            f"\nTraining on dataset: {dataset_name} with {pandas_xtrain.shape[0]} rows, {pandas_xtrain.shape[1]} features, and {pandas_ytrain.nunique()} categories\n",
        )

        input_dim = 100000

        xtrain, ytrain, xvalid, yvalid, xtest, ytest = featurize_bolt_tabular_data(
            pandas_xtrain, pandas_ytrain, tabular_dtypes, input_dim, data_dir
        )

        base_predict_config = bolt.graph.PredictConfig.make().with_metrics(
            ["categorical_accuracy"]
        )

        base_train_config = (
            bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=15)
            .with_metrics(["categorical_accuracy"])
            .with_validation(
                validation_data=[xvalid],
                validation_labels=yvalid,
                predict_config=base_predict_config,
            )
        )

        for run_config in generate_run_configs():
            execute_run(
                run_config,
                xtrain,
                ytrain,
                xtest,
                ytest,
                dataset_name,
                base_train_config,
                base_predict_config,
                num_test_rows,
                input_dim,
                num_classes,
            )


if __name__ == "__main__":
    main()
