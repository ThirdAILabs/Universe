import time
import argparse
import pandas as pd
from pandas.api.types import is_numeric_dtype
from thirdai import bolt
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from bolt.python_tests.utils import compute_accuracy_with_file


def map_categories_to_integers(dataframes):
    for df in dataframes:
        for colname in df.columns:
            if df[colname].dtype == "object":
                df[colname] = pd.factorize(df[colname])[0]


def get_col_datatypes(dataset_base_filename):
    dtypes_file = dataset_base_filename + "/Dtypes.txt"
    with open(dtypes_file) as file:
        lines = file.readlines()
        dtypes = lines[0].split(",")
        # remove trailing newline character from last elem
        dtypes[-1] = dtypes[-1][:-1]
        return dtypes


def accuracy(predictions, ytest):
    val = 0
    for pred, truth in zip(predictions, list(ytest)):
        if pred == truth:
            val += 1
    return val / len(predictions)


def log_message(message, out_file):
    print(message)
    out_file.write(message)


def prep_data(data_dir, dtypes):
    train_data = pd.read_csv(data_dir + "/Train.csv", header=None)
    valid_data = pd.read_csv(data_dir + "/Valid.csv", header=None)
    test_data = pd.read_csv(data_dir + "/Test.csv", header=None)

    label_col_index = dtypes.index("label")
    train_label_col = train_data.columns[label_col_index]
    valid_label_col = valid_data.columns[label_col_index]
    test_label_col = test_data.columns[label_col_index]

    xtrain, ytrain = (
        train_data.drop(train_label_col, axis=1),
        train_data[train_label_col],
    )
    xvalid, yvalid = (
        valid_data.drop(valid_label_col, axis=1),
        valid_data[valid_label_col],
    )
    xtest, ytest = test_data.drop(test_label_col, axis=1), test_data[test_label_col]

    return xtrain, ytrain, xvalid, yvalid, xtest, ytest


def to_numpy(xdata, ydata):
    xdata = xdata.to_numpy()
    ydata = ydata.to_numpy()
    ydata = ydata.flatten()
    return xdata, ydata


def train_bolt(dtypes, ytrain, yvalid, ytest, dataset_base_filename, out_file):
    bolt_train_file = dataset_base_filename + "/Train.csv"
    bolt_valid_file = dataset_base_filename + "/Valid.csv"
    bolt_test_file = dataset_base_filename + "/Test.csv"

    start = time.time()

    prediction_file = "predictions.csv"

    # rather than saving/loading the model that performs the best on the validation set,
    # call predict(..) on the test set after every new best epoch and record that accuracy to report
    best_test_accuracy = 0
    # stop training after this many total dips in successive validation accuracy
    num_bad_epochs = 3
    max_val_acc = 0
    last_accuracy = 0
    tc = bolt.TabularClassifier("medium", ytrain.nunique())
    max_epochs = 20
    for e in range(max_epochs):
        tc.train(bolt_train_file, dtypes, epochs=1, learning_rate=0.01)
        tc.predict(bolt_valid_file, prediction_file)
        val_accuracy = compute_accuracy_with_file(
            [str(x) for x in yvalid], prediction_file
        )
        if val_accuracy < last_accuracy:
            num_bad_epochs -= 1
        elif val_accuracy > max_val_acc:
            max_val_acc = val_accuracy
            tc.predict(bolt_test_file, prediction_file)
            best_test_accuracy = compute_accuracy_with_file(
                [str(x) for x in ytest], prediction_file
            )

        last_accuracy = val_accuracy

        if num_bad_epochs == 0:
            break

    end = time.time()

    start_inference = time.time()
    tc.predict(bolt_test_file)
    end_inference = time.time()
    inference_time = (end_inference - start_inference) / len(ytest)

    log_message(
        f"BOLT Accuracy: {best_test_accuracy}, Total Training Time: {end - start}, Single Inference Time: {inference_time}\n",
        out_file,
    )


def train_xgboost(xtrain, ytrain, xvalid, yvalid, xtest, ytest, out_file):
    map_categories_to_integers([xtrain, xtest, xvalid])

    ytrain = pd.factorize(ytrain)[0]
    yvalid = pd.factorize(yvalid)[0]
    ytest = pd.factorize(ytest)[0]

    start_training = time.time()

    model = XGBClassifier(use_label_encoder=False, max_depth=6, learning_rate=0.3)
    model.fit(
        xtrain,
        ytrain,
        eval_set=[(xvalid, yvalid)],
        early_stopping_rounds=20,
        verbose=True,
    )

    end_training = time.time()

    start_inference = time.time()
    predictions = model.predict(xtest)
    end_inference = time.time()
    inference_time = (end_inference - start_inference) / len(predictions)

    log_message(
        f"XGBoost Accuracy: {accuracy(predictions, ytest)}, Total Training Time: {end_training - start_training}, Single Inference Time: {inference_time}\n",
        out_file,
    )


def train_tabnet(xtrain, ytrain, xvalid, yvalid, xtest, ytest, out_file):
    map_categories_to_integers([xtrain, xtest, xvalid])

    xtrain, ytrain = to_numpy(xtrain, ytrain)
    xvalid, yvalid = to_numpy(xvalid, yvalid)
    xtest, ytest = to_numpy(xtest, ytest)

    start_training = time.time()

    model = TabNetClassifier()
    model.fit(xtrain, ytrain, eval_set=[(xvalid, yvalid)], patience=3)

    end_training = time.time()

    start_inference = time.time()
    predictions = model.predict(xtest)
    end_inference = time.time()
    inference_time = (end_inference - start_inference) / len(predictions)

    log_message(
        f"TabNet Accuracy: {accuracy(predictions, ytest)}, Total Training Time: {end_training - start_training}, Single Inference Time: {inference_time}\n",
        out_file,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Experiment with the Tabular Auto-Classifier"
    )
    parser.add_argument(
        "--run_heavy_tests",
        action="store_true",
        default=False,
        help="Includes large datasets in the experiment (several hours long). Otherwise runs on small datasets only (a few minutes only).",
    )
    args = parser.parse_args()

    datasets = [
        "CensusIncome",
        "ChurnModeling",
        "EyeMovements",
        "PokerHandInduction",
    ]
    large_datasets = [
        "OttoGroupProductClassificationChallenge",
        "BNPParibasCardifClaimsManagement",
        "ForestCoverType",
        # "HiggsBoson", # this one takes fairly long so test at your own discretion
    ]

    if args.run_heavy_tests:
        datasets += large_datasets

    base_dir = "/share/data/tabular_benchmarks/"
    out_file = open("tabular_classifier_results.txt", "w")
    for dataset_name in datasets:
        data_dir = base_dir + dataset_name
        dtypes = get_col_datatypes(data_dir)
        xtrain, ytrain, xvalid, yvalid, xtest, ytest = prep_data(data_dir, dtypes)

        log_message(
            f"\nTraining on dataset: {dataset_name} with {xtrain.shape[0]} rows, {xtrain.shape[1]} features, and {ytrain.nunique()} categories\n",
            out_file,
        )

        train_bolt(dtypes, ytrain, yvalid, ytest, data_dir, out_file)
        train_xgboost(xtrain, ytrain, xvalid, yvalid, xtest, ytest, out_file)
        train_tabnet(xtrain, ytrain, xvalid, yvalid, xtest, ytest, out_file)

    out_file.close()


if __name__ == "__main__":
    main()
