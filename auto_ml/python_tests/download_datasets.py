import json
import os
import zipfile

import pandas as pd
import pytest
import datasets


@pytest.fixture(scope="session")
def download_census_income():
    CENSUS_INCOME_BASE_DOWNLOAD_URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    )
    TRAIN_FILE = "./census_income_train.csv"
    TEST_FILE = "./census_income_test.csv"
    COLUMN_NAMES = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "label",
    ]
    if not os.path.exists(TRAIN_FILE):
        os.system(
            f"curl {CENSUS_INCOME_BASE_DOWNLOAD_URL}adult.data --output {TRAIN_FILE}"
        )
        # reformat the train file
        with open(TRAIN_FILE, "r") as file:
            data = file.read().splitlines(True)
        with open(TRAIN_FILE, "w") as file:
            # Write header
            file.write(",".join(COLUMN_NAMES) + "\n")
            # Convert ", " delimiters to ",".
            file.writelines([line.replace(", ", ",") for line in data[1:]])

    if not os.path.exists(TEST_FILE):
        os.system(
            f"curl {CENSUS_INCOME_BASE_DOWNLOAD_URL}adult.test --output {TEST_FILE}"
        )
        # reformat the test file
        with open(TEST_FILE, "r") as file:
            data = file.read().splitlines(True)
        with open(TEST_FILE, "w") as file:
            # Write header
            file.write(",".join(COLUMN_NAMES) + "\n")
            # Convert ", " delimiters to ",".
            # Additionally, for some reason each of the labels end with a "." in the test set
            # loop through data[1:] since the first line is bogus
            file.writelines(
                [line.replace(".", "").replace(", ", ",") for line in data[1:]]
            )

    inference_samples = []
    with open(TEST_FILE, "r") as test_file:
        for line in test_file.readlines()[1:-1]:
            column_vals = {
                col_name: value
                for col_name, value in zip(COLUMN_NAMES, line.split(","))
            }
            label = column_vals["label"].strip()
            del column_vals["label"]
            inference_samples.append((column_vals, label))

    return TRAIN_FILE, TEST_FILE, inference_samples


@pytest.fixture(scope="session")
def download_clinc_dataset():
    CLINC_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00570/clinc150_uci.zip"
    CLINC_ZIP = "./clinc150_uci.zip"
    CLINC_DIR = "./clinc"
    MAIN_FILE = CLINC_DIR + "/clinc150_uci/data_full.json"
    TRAIN_FILE = "./clinc_train.csv"
    TEST_FILE = "./clinc_test.csv"

    if not os.path.exists(CLINC_ZIP):
        os.system(f"curl {CLINC_URL} --output {CLINC_ZIP}")

    if not os.path.exists(MAIN_FILE):
        with zipfile.ZipFile(CLINC_ZIP, "r") as zip_ref:
            zip_ref.extractall(CLINC_DIR)

    samples = json.load(open(MAIN_FILE))

    train_samples = samples["train"]
    test_samples = samples["test"]

    train_text, train_category = zip(*train_samples)
    test_text, test_category = zip(*test_samples)

    train_df = pd.DataFrame({"text": train_text, "category": train_category})
    test_df = pd.DataFrame({"text": test_text, "category": test_category})

    train_df["text"] = train_df["text"].apply(lambda x: x.replace(",", ""))
    train_df["category"] = pd.Categorical(train_df["category"]).codes
    test_df["text"] = test_df["text"].apply(lambda x: x.replace(",", ""))
    test_df["category"] = pd.Categorical(test_df["category"]).codes

    # The columns=["category", "text"] is just to force the order of the output
    # columns which since the model pipeline which uses this function does not
    # use the header to determine the column ordering.
    train_df.to_csv(TRAIN_FILE, index=False, columns=["category", "text"])
    test_df.to_csv(TEST_FILE, index=False, columns=["category", "text"])

    inference_samples = []
    for _, row in test_df.iterrows():
        inference_samples.append(({"text": row["text"]}, row["category"]))

    return TRAIN_FILE, TEST_FILE, inference_samples


@pytest.fixture(scope="session")
def download_brazilian_houses_dataset():
    TRAIN_FILE = "./brazilian_houses_train.csv"
    TEST_FILE = "./brazilian_houses_test.csv"

    dataset = datasets.load_dataset("inria-soda/tabular-benchmark", data_files="reg_num/Brazilian_houses.csv")

    df = pd.DataFrame(dataset["train"].shuffle())
    
    # Split in to train/test, there are about 10,000 rows in entire dataset.
    train_df = df.iloc[:8000,:]
    test_df = df.iloc[8000:,:]

    train_df = train_df.drop("Unnamed: 0", axis=1)
    test_df = test_df.drop("Unnamed: 0", axis=1)

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    for _, row in test_df.iterrows():
        print(row)
        print(type(row))
        break
    

    return TRAIN_FILE, TEST_FILE