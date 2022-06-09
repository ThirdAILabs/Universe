import argparse
import mlflow
import os
import platform
import pandas as pd
import psutil
import socket
import toml
import pandas as pd

from thirdai import bolt
from util import log_machine_info, start_mlflow


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
    parser.add_argument(
        "--train_dataset", required=True, help="Dataset on which to train classifier"
    )
    parser.add_argument(
        "--test_dataset",
        required=True,
        help="Dataset on which to evaluate the classifier",
    )
    parser.add_argument(
        "--disable_mlflow",
        action="store_true",
        help="Disable mlflow logging for the current run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of cycles through the data on which to train",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        required=True,
        help="The number of output classes in the dataset",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="The learning rate used for training",
    )
    parser.add_argument(
        "--model_size",
        default="small",
        choices=["small", "medium", "large"],
        help="The desired model size. BOLT will automatically configure the parameters based on the choice of size",
    )
    parser.add_argument(
        "--prediction_file_path",
        default="predictions.txt",
        help="Path to write out classifier predictions on test dataset",
    )
    parser.add_argument("--experiment_name", help="Name of experiment for mlflow")
    parser.add_argument(
        "--run_name",
        type=str,
        help="The name of the run to use in mlflow, if mlflow is not disabled this is required.",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    mlflow_enabled = not args.disable_mlflow
    if mlflow_enabled and not args.run_name:
        parser.print_usage()
        raise ValueError("Error: --run_name is required when using mlflow logging.")

    if mlflow_enabled:
        start_mlflow(args.experiment_name, args.run_name, args.train_dataset)
        log_machine_info()
        mlflow.log_params(vars(args))

    classifier = train_classifier(
        args.train_dataset,
        args.n_classes,
        args.model_size,
        args.epochs,
        args.learning_rate,
    )
    evaluate_classifier(
        classifier, args.test_dataset, output_file=args.prediction_file_path
    )


if __name__ == "__main__":
    main()
