import argparse
import mlflow
import pandas as pd
import toml
import pandas as pd

from thirdai import bolt
from benchmarks.text_classifier_benchmarks.utils import find_full_filepath
from utils import log_machine_info, start_mlflow, config_get, config_get_or


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
        "config_path",
        type=str,
        help="Path to a config file containing the dataset, experiment, and model configs.",
    )
    parser.add_argument(
        "--disable_mlflow",
        action="store_true",
        help="Disable mlflow logging for the current run.",
    )
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

    config = toml.load(args.config_path)

    if mlflow_enabled:
        start_mlflow(
            config_get(config, "experiment_name"),
            args.run_name,
            config_get(config, "dataset_name"),
            model_name="text_classifier",
        )
        log_machine_info()
        mlflow.log_params({"run_name": args.run_name, **vars(config)})

    classifier = train_classifier(
        find_full_filepath(config_get(config, "train_dataset_path")),
        config_get(config, "n_classes"),
        config_get_or(config, "model_size", default="small"),
        config_get_or(config, "epochs", default=5),
        config_get_or(config, "learning_rate", default=0.01),
    )
    if mlflow_enabled:
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
