import argparse
import mlflow
import os
import pandas as pd
import sys
import toml
import pandas as pd

from thirdai import bolt
from benchmarks.utils import add_mlflow_args, verify_mlflow_args

sys.path.append(os.path.dirname(__file__) + "/..")
from utils import (
    add_mlflow_args,
    config_get, 
    config_get_or, 
    find_full_filepath,
    load_config,
    mlflow_is_enabled,
    start_mlflow,
    verify_mlflow_args
)


def compute_accuracy(test_file, pred_file):
    with open(pred_file) as pred:
        predictions = pred.read().splitlines()

    test_csv = pd.read_csv(test_file, dtype=str)
    labels = test_csv.category.tolist()

    correct = 0
    total = 0

    if len(predictions) != len(labels):
        raise ValueError(
            f"The number of predictions ({len(predictions)}) does not match the number of test examples ({len(labels)})"
        )
    for (prediction, answer) in zip(predictions, labels):
        if prediction == answer:
            correct += 1
        total += 1

    print("Accuracy = {} / {} = {}".format(correct, total, correct / total))
    return correct / total


def train_classifier(train_dataset, n_classes, model_size, epochs, learning_rate):
    classifier = bolt.TextClassifier(model_size=model_size, n_classes=n_classes)

    classifier.train(
        train_file=train_dataset, epochs=epochs, learning_rate=learning_rate
    )

    return classifier


def evaluate_classifier(classifier, test_dataset, output_file):
    classifier.predict(
        test_file=test_dataset,
        output_file=output_file,
    )
    accuracy = compute_accuracy(test_dataset, output_file)
    mlflow.log_metric("accuracy", accuracy)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Trains and evaluates a generic bolt text classifier "
    )
    add_mlflow_args(parser)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    
    verify_mlflow_args(parser, args)

    config = load_config(args)

    if mlflow_is_enabled(args):
        start_mlflow(config, mlflow_args=args)

    classifier = train_classifier(
        find_full_filepath(config_get(config, "train_dataset_path")),
        config_get(config, "n_classes"),
        config_get_or(config, "model_size", default="small"),
        config_get_or(config, "epochs", default=5),
        config_get_or(config, "learning_rate", default=0.01),
    )
    if mlflow_is_enabled(args):
        mlflow.log_params(classifier.get_hyper_parameters())
    evaluate_classifier(
        classifier,
        find_full_filepath(config_get(config, "test_dataset_path")),
        output_file=config_get_or(
            config, "prediction_file_path", default="predictions.txt"
        ),
    )


if __name__ == "__main__":
    main()
