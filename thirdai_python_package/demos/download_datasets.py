import json
import math
import os
import random
import zipfile

import numpy as np
import pandas as pd


def _download_dataset(url, zip_file, check_existence, output_dir):
    if not os.path.exists(zip_file):
        os.system(f"curl {url} --output {zip_file}")

    if any([not os.path.exists(must_exist) for must_exist in check_existence]):
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(output_dir)


def to_udt_input_batch(dataframe):
    return [
        {col_name: str(col_value) for col_name, col_value in record.items()}
        for record in dataframe.to_dict(orient="records")
    ]


def download_movielens():
    MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    MOVIELENS_ZIP = "./movielens.zip"
    MOVIELENS_DIR = "./movielens"
    RATINGS_FILE = MOVIELENS_DIR + "/ml-1m/ratings.dat"
    MOVIE_TITLES = MOVIELENS_DIR + "/ml-1m/movies.dat"
    TRAIN_FILE = "./movielens_train.csv"
    TEST_FILE = "./movielens_test.csv"
    SPLIT = 0.9
    INFERENCE_BATCH_SIZE = 5

    _download_dataset(
        url=MOVIELENS_1M_URL,
        zip_file=MOVIELENS_ZIP,
        check_existence=[RATINGS_FILE, MOVIE_TITLES],
        output_dir=MOVIELENS_DIR,
    )

    df = pd.read_csv(RATINGS_FILE, header=None, delimiter="::", engine="python")
    df.columns = ["userId", "movieId", "rating", "timestamp"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    movies_df = pd.read_csv(
        MOVIE_TITLES,
        header=None,
        delimiter="::",
        engine="python",
        encoding="ISO-8859-1",
    )
    movies_df.columns = ["movieId", "movieTitle", "genre"]
    movies_df["movieTitle"] = movies_df["movieTitle"].apply(
        lambda x: x.replace(",", "")
    )

    df = pd.merge(df, movies_df, on="movieId")
    df = df[["userId", "movieTitle", "timestamp"]]
    df = df.sort_values("timestamp")

    n_train_samples = int(SPLIT * len(df))
    train_df = df.iloc[:n_train_samples]
    test_df = df.iloc[n_train_samples:]
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    index_batch = to_udt_input_batch(df.iloc[:INFERENCE_BATCH_SIZE])
    inference_batch = to_udt_input_batch(
        df.iloc[:INFERENCE_BATCH_SIZE][["userId", "timestamp"]]
    )

    return TRAIN_FILE, TEST_FILE, inference_batch, index_batch


def download_criteo():
    CRITEO_URL = "http://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz"
    CRITEO_ZIP = "./criteo.tar.gz"
    CRITEO_DIR = "./criteo"
    MAIN_FILE = CRITEO_DIR + "/train.txt"
    CREATED_TRAIN_FILE = "./criteo/train_udt.csv"
    CREATED_TEST_FILE = "./criteo/test_udt.csv"

    os.makedirs("./criteo", exist_ok=True)

    if not os.path.exists(CRITEO_ZIP):
        print(
            f"Downloading from {CRITEO_URL}. This can take 20-40 minutes depending on"
            " the Criteo server."
        )
        os.system(f"wget -t inf -c {CRITEO_URL} -O {CRITEO_ZIP}")

    if not os.path.exists(MAIN_FILE):
        print("Extracting files. This can take up to 10 minutes.")
        os.system(f"tar -xvzf {CRITEO_ZIP} -C {CRITEO_DIR}")

    df = pd.read_csv(MAIN_FILE, delimiter="\t", header=None)
    n_train = int(0.8 * df.shape[0])
    header = (
        ["label"]
        + [f"num_{i}" for i in range(1, 14)]
        + [f"cat_{i}" for i in range(1, 27)]
    )

    print("Processing the dataset (this will take about 6-7 mins).")
    min_vals = df.iloc[:, 1:14].min()
    df.iloc[:, 1:14] = np.round(np.log(df.iloc[:, 1:14] - min_vals + 1), 2)
    min_vals = np.float32(df.iloc[:, 1:14].min())
    max_vals = np.float32(df.iloc[:, 1:14].max())
    y = np.float32(df.iloc[:, 0])
    n_unique_classes = list(df.iloc[:, 14:].nunique())

    y_train = y[:n_train]
    y_test = y[n_train:]

    if not os.path.exists(CREATED_TRAIN_FILE) or not os.path.exists(CREATED_TEST_FILE):
        print("saving the train and test datasets (this will take about 10 mins)")
        df[:n_train].to_csv(CREATED_TRAIN_FILE, header=header, index=False)
        df[n_train:].to_csv(CREATED_TEST_FILE, header=header, index=False)

    df_sample = df.iloc[n_train : n_train + 2]
    df_sample = df_sample.fillna("")
    sample_batch = [
        {header[i]: str(df_sample.iloc[0, i]) for i in range(1, 40)}
    ]  # first sample
    sample_batch.append(
        {header[i]: str(df_sample.iloc[1, i]) for i in range(1, 40)}
    )  # second sample

    return (
        CREATED_TRAIN_FILE,
        CREATED_TEST_FILE,
        y_train,
        y_test,
        min_vals,
        max_vals,
        n_unique_classes,
        sample_batch,
    )


def prep_fraud_dataset(dataset_path, seed=42):
    df = pd.read_csv(dataset_path)
    df["amount"] = (df["oldbalanceOrg"] - df["newbalanceOrig"]).abs()

    def upsample(df):
        fraud_samples = df[df["isFraud"] == 1]
        upsampling_ratio = 5
        for i in range(upsampling_ratio):
            df = pd.concat([df, fraud_samples], axis=0)
        return df

    df = upsample(df)

    df = df.sample(frac=1, random_state=seed)

    SPLIT = 0.8
    n_train_samples = int(SPLIT * len(df))
    train_df = df.iloc[:n_train_samples]
    test_df = df.iloc[n_train_samples:]

    train_filename = "fraud_detection/new_train.csv"
    test_filename = "fraud_detection/new_test.csv"

    train_df.to_csv(train_filename, index=False)
    test_df.to_csv(test_filename, index=False)

    INFERENCE_BATCH_SIZE = 5
    inference_batch = to_udt_input_batch(
        df.iloc[:INFERENCE_BATCH_SIZE][
            [
                "step",
                "type",
                "amount",
                "nameOrig",
                "oldbalanceOrg",
                "newbalanceOrig",
                "nameDest",
                "oldbalanceDest",
                "newbalanceDest",
                "isFlaggedFraud",
            ]
        ]
    )

    return train_filename, test_filename, inference_batch


def download_census_income(num_inference_samples=5, return_labels=False):
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
            # loop through data[1:] since the first line is bogus
            lines = [line.replace(", ", ",") for line in data[1:]]
            # Strip empty lines
            file.writelines([line for line in lines if len(line.strip()) > 0])

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
            lines = [line.replace(".", "").replace(", ", ",") for line in data[1:]]
            # Strip empty lines
            file.writelines([line for line in lines if len(line.strip()) > 0])

    inference_sample_range_end = (
        -1 if num_inference_samples == "all" else num_inference_samples + 1
    )

    inference_samples = []
    with open(TEST_FILE, "r") as test_file:
        for line in test_file.readlines()[1:inference_sample_range_end]:
            column_vals = {
                col_name: value
                for col_name, value in zip(COLUMN_NAMES, line.split(","))
            }
            label = column_vals["label"].strip()
            del column_vals["label"]

            if return_labels:
                inference_samples.append((column_vals, label))
            else:
                inference_samples.append(column_vals)

    return TRAIN_FILE, TEST_FILE, inference_samples


def download_query_reformulation_dataset(train_file_percentage=0.7):
    """
    The dataset is retrieved from HuggingFace:
    https://huggingface.co/datasets/snips_built_in_intents
    """
    import datasets

    dataset = datasets.load_dataset(path="embedding-data/sentence-compression")
    dataframe = pd.DataFrame(data=dataset)

    extracted_text = []

    for _, row in dataframe.iterrows():
        extracted_text.append(row.to_dict()["train"]["set"][1])

    return pd.DataFrame(data=extracted_text)


def perturb_query_reformulation_data(dataframe, noise_level, seed=42):
    random.seed(seed)

    transformation_type = ("remove-char", "permute-string")
    transformed_dataframe = []

    PER_QUERY_COPIES = 5

    for _, row in dataframe.iterrows():
        correct_query = " ".join(list(row.str.split(" ")[0]))
        query_length = len(correct_query.split(" "))
        words_to_transform = math.ceil(noise_level * query_length)

        for _ in range(PER_QUERY_COPIES):
            incorrect_query_list = correct_query.split(" ")
            transformed_words = 0
            visited_indices = set()

            while transformed_words < words_to_transform:
                random_index = random.randint(0, words_to_transform)
                if random_index in visited_indices:
                    continue
                word_to_transform = incorrect_query_list[random_index]

                if random.choices(transformation_type, k=1) == "remove-char":
                    # Remove a random character
                    char_index = random.randint(0, len(word_to_transform) - 1)
                    transformed_word = (
                        word_to_transform[0:char_index]
                        + word_to_transform[char_index + 1 :]
                    )
                    incorrect_query_list[random_index] = transformed_word

                else:
                    # Permute the characters in the string
                    transformed_word_char_list = list(word_to_transform)
                    random.shuffle(transformed_word_char_list)

                    incorrect_query_list[random_index] = "".join(
                        transformed_word_char_list
                    )

                visited_indices.add(random_index)
                transformed_words += 1

            transformed_dataframe.append(
                [correct_query, " ".join(incorrect_query_list)]
            )

    return pd.DataFrame(
        transformed_dataframe, columns=["target_queries", "source_queries"]
    )


# Supervised Training File has both the target and the source columns, whereas the
# Unsupervised Training File has only the target column.
def prepare_query_reformulation_data(seed=42):
    SUPERVISED_TRAIN_FILE_PATH = "supervised_train_file.csv"
    UNSUPERVISED_TRAIN_FILE_PATH = "unsupervised_train_file.csv"
    TEST_FILE_PATH = "test_file.csv"
    TRAIN_FILE_DATASET_PERCENTAGE = 0.7
    INFERENCE_BATCH_PERCENTAGE = 0.0001
    TRAIN_NOISE_LEVEL = 0.2
    TEST_NOISE_LEVEL = 0.4

    def get_inference_batch(dataframe):
        inference_batch = dataframe.sample(
            frac=INFERENCE_BATCH_PERCENTAGE, random_state=seed
        )
        inference_batch_as_list = []
        for _, row in inference_batch.iterrows():
            inference_batch_as_list.append(row.to_dict()[0])

        return inference_batch_as_list

    train_data = download_query_reformulation_dataset(
        train_file_percentage=TRAIN_FILE_DATASET_PERCENTAGE
    )
    inference_batch = get_inference_batch(dataframe=train_data)

    train_data_with_noise = perturb_query_reformulation_data(
        dataframe=train_data, noise_level=TRAIN_NOISE_LEVEL
    )
    sampled_train_data = train_data.sample(
        frac=1 - TRAIN_FILE_DATASET_PERCENTAGE, random_state=seed
    )

    test_data_with_noise = perturb_query_reformulation_data(
        dataframe=pd.DataFrame(sampled_train_data),
        noise_level=TEST_NOISE_LEVEL,
    )

    # TODO(Geordie): Fix this when the new CSV parser is in
    train_data_with_noise = train_data_with_noise.replace(",", "", regex=True)
    test_data_with_noise = test_data_with_noise.replace(",", "", regex=True)

    # Write dataset to CSV
    train_data_with_noise.to_csv(SUPERVISED_TRAIN_FILE_PATH, index=False)
    train_data_with_noise["target_queries"].to_csv(
        UNSUPERVISED_TRAIN_FILE_PATH, index=False
    )
    test_data_with_noise.to_csv(TEST_FILE_PATH, index=False)

    return (
        SUPERVISED_TRAIN_FILE_PATH,
        UNSUPERVISED_TRAIN_FILE_PATH,
        TEST_FILE_PATH,
        inference_batch,
    )


def download_clinc_dataset(
    num_training_files=1, clinc_small=False, file_prefix="clinc"
):
    CLINC_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00570/clinc150_uci.zip"
    CLINC_ZIP = "./clinc150_uci.zip"
    CLINC_DIR = "./clinc"
    MAIN_FILE = CLINC_DIR + "/clinc150_uci/data_full.json"
    SMALL_FILE = CLINC_DIR + "/clinc150_uci/data_small.json"
    TRAIN_FILE = f"./{file_prefix}_train.csv"
    TEST_FILE = f"./{file_prefix}_test.csv"
    TRAIN_FILES = []

    _download_dataset(
        url=CLINC_URL,
        zip_file=CLINC_ZIP,
        check_existence=[MAIN_FILE],
        output_dir=CLINC_DIR,
    )

    samples = None

    if clinc_small:
        samples = json.load(open(SMALL_FILE))
    else:
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

    test_df.to_csv(TEST_FILE, index=False, columns=["category", "text"])

    inference_samples = []
    for _, row in test_df.iterrows():
        inference_samples.append(({"text": row["text"]}, row["category"]))

    # The columns=["category", "text"] is just to force the order of the output
    # columns which since the model pipeline which uses this function does not
    # use the header to determine the column ordering.
    if num_training_files == 1:
        train_df.to_csv(TRAIN_FILE, index=False, columns=["category", "text"])

        return TRAIN_FILE, TEST_FILE, inference_samples
    else:
        training_data_per_file = len(train_df) // num_training_files

        # saving all files with TRAIN_FILE_i(0 indexed)
        for i in range(num_training_files):
            l_index, r_index = (
                i * training_data_per_file,
                (i + 1) * training_data_per_file,
            )
            filename = f"{file_prefix}_train" + f"_{i}.csv"
            train_df.iloc[l_index:r_index].to_csv(
                filename, index=False, columns=["category", "text"]
            )
            TRAIN_FILES.append(filename)
        return TRAIN_FILES, TEST_FILE, inference_samples


def download_brazilian_houses_dataset():
    TRAIN_FILE = "./brazilian_houses_train.csv"
    TEST_FILE = "./brazilian_houses_test.csv"

    import datasets

    dataset = datasets.load_dataset(
        "inria-soda/tabular-benchmark", data_files="reg_num/Brazilian_houses.csv"
    )

    df = pd.DataFrame(dataset["train"].shuffle())

    # Split in to train/test, there are about 10,000 rows in entire dataset.
    train_df = df.iloc[:8000, :]
    test_df = df.iloc[8000:, :]

    train_df = train_df.drop("Unnamed: 0", axis=1)
    test_df = test_df.drop("Unnamed: 0", axis=1)

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    inference_samples = []
    for _, row in test_df.iterrows():
        sample = dict(row)
        label = sample["totalBRL"]
        del sample["totalBRL"]
        sample = {x: str(y) for x, y in sample.items()}
        inference_samples.append((sample, label))

    return TRAIN_FILE, TEST_FILE, inference_samples


def download_internet_ads_dataset(seed=42):
    random.seed(seed)

    INTERNET_ADS_URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.data"
    )
    INTERNET_ADS_FILE = "./internet_ads.data"
    TRAIN_FILE = "./internet_ads_train.csv"
    TEST_FILE = "./internet_ads_test.csv"

    column_names = [str(i) for i in range(1558)] + ["label"]

    if not os.path.exists(INTERNET_ADS_FILE):
        os.system(f"curl {INTERNET_ADS_URL} --output {INTERNET_ADS_FILE}")

    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        header = ",".join(column_names) + "\n"

        with open(INTERNET_ADS_FILE, "r") as data_file:
            lines = data_file.readlines()
        for i, line in enumerate(lines):
            cols = line.strip().split(",")
            for j, col in enumerate(cols[:3]):
                if "?" in col:
                    cols[j] = ""
            lines[i] = ",".join(cols) + "\n"

        random.shuffle(lines)

        train_test_split = int(0.8 * len(lines))

        with open(TRAIN_FILE, "w") as train_file:
            train_file.write(header)
            train_file.writelines(lines[:train_test_split])

        with open(TEST_FILE, "w") as test_file:
            test_file.write(header)
            test_file.writelines(lines[train_test_split:])

    inference_samples = []
    with open(TEST_FILE, "r") as test_file:
        for line in test_file.readlines()[1:]:
            column_vals = {
                col_name: value
                for col_name, value in zip(column_names, line.split(","))
            }
            label = column_vals["label"].strip()
            del column_vals["label"]
            inference_samples.append((column_vals, label))

    return TRAIN_FILE, TEST_FILE, inference_samples
