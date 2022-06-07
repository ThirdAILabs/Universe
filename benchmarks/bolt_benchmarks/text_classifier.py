import argparse
import mlflow
import os
import platform
import pandas as pd
import psutil
import socket
import time
import toml
import pandas as pd
import time

from thirdai import bolt


def start_mlflow(run_name, dataset):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(file_dir, "../config.toml")
    with open(file_name) as f:
        parsed_config = toml.load(f)
        mlflow.set_tracking_uri(parsed_config["tracking"]["uri"])

    mlflow.set_experiment("Generic Classifier")
    mlflow.start_run(
        run_name=run_name,
        tags={"dataset": dataset},
    )


def log_machine_info():
    machine_info = {
        "load_before_experiment": os.getloadavg()[2],
        "platform": platform.platform(),
        "platform_version": platform.version(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "ram_gb": round(psutil.virtual_memory().total / (1024.0**3)),
        "num_cores": psutil.cpu_count(logical=True),
    }

    mlflow.log_params(machine_info)


def compute_accuracy(test_file, pred_file):
    with open(pred_file) as pred:
        predictions = pred.readlines()

    test_csv = pd.read_csv(test_file)
    labels = test_csv.category.tolist()

    correct = 0
    total = 0

    if len(predictions) != len(labels):
        raise ValueError(
            f"The number of predictions ({predictions}) does not match the number of test examples ({labels})"
        )
    for (prediction, answer) in zip(predictions, labels):
        if prediction[:-1] == answer:
            correct += 1
        total += 1

    print("Accuracy = {} / {} = {}".format(correct, total, correct / total))
    return correct / total


def train_classifier(
    train_dataset, n_classes, model_size="small", epochs=5, learning_rate=0.01
):
    classifier = bolt.TextClassifier(model_size=model_size, n_classes=n_classes)

    train_start = time.perf_counter()
    classifier.train(
        train_file=train_dataset, epochs=epochs, learning_rate=learning_rate
    )
    training_time = time.perf_counter() - train_start
    mlflow.log_metric("training_time", training_time)

    return classifier


def evaluate_classifier(classifier, test_dataset, output_file="predictions.txt"):
    inference_start = time.perf_counter()
    classifier.predict(
        test_file=test_dataset,
        output_file=output_file,
    )
    inference_time = time.perf_counter() - inference_start
    accuracy = compute_accuracy(test_dataset, output_file)
    mlflow.log_metric("inference_time", inference_time)
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
        start_mlflow(args.run_name, args.train_dataset)
        log_machine_info()
        mlflow.log_params(vars(args))

    classifier = train_classifier(args.train_dataset, args.n_classes, args.model_size, args.epochs, args.learning_rate)
    evaluate_classifier(classifier, args.test_dataset, output_file=args.prediction_file_path)


if __name__ == "__main__":
    main()
