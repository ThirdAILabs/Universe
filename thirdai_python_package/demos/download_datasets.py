import json
import math
import os
import random
import zipfile

import numpy as np
import pandas as pd
import requests
from thirdai._thirdai import bolt

from .beir_download_utils import (
    GenericDataLoader,
    download_and_unzip,
    remap_doc_ids,
    remap_query_answers,
    write_supervised_file,
    write_unsupervised_file,
)


def download_file(url, output_path, verify_certificate=True):
    response = requests.get(url, stream=True, verify=verify_certificate)
    response.raise_for_status()
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


def _download_dataset(
    url, zip_file, check_existence, output_dir, verify_certificate=True
):
    if not os.path.exists(zip_file):
        download_file(url, zip_file, verify_certificate)

    if any([not os.path.exists(must_exist) for must_exist in check_existence]):
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(output_dir)


def _create_inference_samples(filename, label_col):
    df = pd.read_csv(filename)
    inference_samples = []
    for _, row in df.iterrows():
        sample = dict(row)
        label = sample[label_col]
        del sample[label_col]
        sample = {x: str(y) for x, y in sample.items()}
        inference_samples.append((sample, label))
    return inference_samples


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
        verify_certificate=False,
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
    CENSUS_INCOME_BASE_DOWNLOAD_URL = "https://www.dropbox.com/scl/fi/xg5jld8rj2h3yciduts6l/census-income.zip?rlkey=xo2zs5mtvbl917kgevok4fk1q&st=ehrcbkzo&dl=1"
    CENSUS_INCOME_ZIP = "./adult.zip"
    CENSUS_INCOME_DIR = "./adult"
    TRAIN_FILE = "./census_income_train.csv"
    TEST_FILE = "./census_income_test.csv"
    _download_dataset(
        url=CENSUS_INCOME_BASE_DOWNLOAD_URL,
        zip_file=CENSUS_INCOME_ZIP,
        check_existence=["./adult/adult.data", "./adult/adult.test"],
        output_dir=CENSUS_INCOME_DIR,
    )
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
        # reformat the train file
        with open("./adult/adult.data", "r") as file:
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
        # reformat the test file
        with open("./adult/adult.test", "r") as file:
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


def prepare_query_reformulation_data(seed=42):
    TRAIN_FILE_PATH = "train_file.csv"
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
            inference_batch_as_list.append({"phrase": row.to_dict()[0]})

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
    train_data_with_noise.to_csv(TRAIN_FILE_PATH, index=False)
    test_data_with_noise.to_csv(TEST_FILE_PATH, index=False)

    return (
        TRAIN_FILE_PATH,
        TEST_FILE_PATH,
        inference_batch,
    )


def download_clinc_dataset(
    num_training_files=1, clinc_small=False, file_prefix="clinc"
):
    CLINC_URL = "https://www.dropbox.com/scl/fi/doxyeurqxvgyperfqwk0r/clinc150.zip?rlkey=s4jfwbjzfwdfro2f82vnatldp&st=u0txk4xx&dl=1"
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

    train_df["text"] = train_df["text"]
    train_df["category"] = pd.Categorical(train_df["category"]).codes
    test_df["text"] = test_df["text"]
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

    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        import datasets

        dataset = datasets.load_dataset(
            "inria-soda/tabular-benchmark", data_files="reg_num/Brazilian_houses.csv"
        )

        df = pd.DataFrame(dataset["train"].shuffle())

        # Split in to train/test, there are about 10,000 rows in entire dataset.
        train_df = df.iloc[:8000, :]
        test_df = df.iloc[8000:, :]

        train_df.to_csv(TRAIN_FILE, index=False)
        test_df.to_csv(TEST_FILE, index=False)

    inference_samples = _create_inference_samples(
        filename=TEST_FILE, label_col="totalBRL"
    )

    return TRAIN_FILE, TEST_FILE, inference_samples


def download_internet_ads_dataset(seed=42):
    random.seed(seed)

    INTERNET_ADS_DOWNLOAD_URL = "https://www.dropbox.com/scl/fi/ze6h56r9a2uy8mzpo14yf/internet-advertisements.zip?rlkey=lmgo50xhugjb4wrwblnynye1a&st=2dil84zs&dl=1"
    INTERNET_ADS_ZIP = "./internet+advertisements.zip"
    INTERNET_ADS_DIR = "./internet+advertisements"
    _download_dataset(
        url=INTERNET_ADS_DOWNLOAD_URL,
        zip_file=INTERNET_ADS_ZIP,
        check_existence=["./internet+advertisements/ad.data"],
        output_dir=INTERNET_ADS_DIR,
    )
    INTERNET_ADS_FILE = "./internet+advertisements/ad.data"
    TRAIN_FILE = "./internet_ads_train.csv"
    TEST_FILE = "./internet_ads_test.csv"

    column_names = [str(i) for i in range(1558)] + ["label"]

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

    inference_samples = _create_inference_samples(filename=TEST_FILE, label_col="label")

    return TRAIN_FILE, TEST_FILE, inference_samples


def download_mnist_dataset():
    TRAIN_FILE = "mnist"
    TEST_FILE = "mnist.t"
    if not os.path.exists(TRAIN_FILE):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 --output mnist.bz2"
        )
        os.system("bzip2 -d mnist.bz2")

    if not os.path.exists(TEST_FILE):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2 --output mnist.t.bz2"
        )
        os.system("bzip2 -d mnist.t.bz2")

    return TRAIN_FILE, TEST_FILE


def download_yelp_chi_dataset(seed=42):
    PATH = "yelp_all.csv"
    URL = "https://www.dropbox.com/s/ge2sr9iab16hc1x/yelp_all.csv"
    TRAIN_FILE = "yelp_train.csv"
    TEST_FILE = "yelp_test.csv"

    if not os.path.exists(PATH):
        # -L will follow the redirects to correctly download the file from dropbox
        os.system(f"curl -L {URL} --output {PATH}")

    all_data = pd.read_csv("yelp_all.csv")
    all_data = all_data.sample(frac=1, random_state=seed)

    numerical_col_names = ["col_" + str(i) for i in range(32)]
    numerical_col_ranges = (
        all_data[numerical_col_names].agg([min, max]).T.values.tolist()
    )

    # Create train and test splits
    train_length = all_data.shape[0] // 2
    test_length = all_data.shape[0] - train_length
    train_data, test_data = (
        all_data.head(train_length).copy(),
        all_data.tail(test_length).copy(),
    )
    train_data.to_csv(TRAIN_FILE, index=False)

    # Save the test data at first with the labels so that we can create inference samples
    test_data.to_csv(TEST_FILE, index=False)
    inference_samples = _create_inference_samples(
        filename=TEST_FILE, label_col="target"
    )

    # Zero the ground truth so the model doesn't have access to it during evaluation
    test_data["target"] = np.zeros(test_length)
    test_data.to_csv(TEST_FILE, index=False)

    udt_data_types = {
        "node_id": bolt.types.node_id(),
        **{
            col_name: bolt.types.numerical(col_range)
            for col_range, col_name in zip(numerical_col_ranges, numerical_col_names)
        },
        "target": bolt.types.categorical(n_classes=2, type="int"),
        "neighbors": bolt.types.neighbors(),
    }

    return TRAIN_FILE, TEST_FILE, inference_samples, udt_data_types


def download_amazon_kaggle_product_catalog_sampled():
    TRAIN_FILE = "amazon-kaggle-product-catalog.csv"
    if not os.path.exists(TRAIN_FILE):
        os.system(
            "curl -L https://www.dropbox.com/s/tf7e5m0cikhcb95/amazon-kaggle-product-catalog-sampled-0.05.csv?dl=0 -o amazon-kaggle-product-catalog.csv"
        )

    df = pd.read_csv(f"{os.getcwd()}/{TRAIN_FILE}")
    n_classes = df.shape[0]

    return TRAIN_FILE, n_classes


def download_agnews_dataset(corpus_file):
    from datasets import load_dataset

    corpus = load_dataset("ag_news")["train"]["text"]
    with open(corpus_file, "w") as fw:
        nothing = fw.write("id,text\n")
        count = 0
        for line in corpus:
            nothing = fw.write(str(count) + "," + line.replace(",", " ").lower() + "\n")
            count += 1

    return len(corpus)


def download_beir_dataset(dataset):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = download_and_unzip(url, ".")

    corpus, queries_test, qrels_test = GenericDataLoader(data_folder=data_path).load(
        split="test"
    )

    write_unsupervised_file(corpus, data_path)

    # we remap doc ids from 0 to N-1 so we can specify integer target in UDT
    # coldstart only works with integer target for now
    doc_ids_to_integers = remap_doc_ids(corpus)
    n_classes = len(doc_ids_to_integers)

    # Not all of the beir datasets come with a train split, some only have a test
    # split. In cases without a train split, we won't write a new supervised train file.
    if os.path.exists(data_path + "/qrels/train.tsv"):
        _, queries_train, qrels_train = GenericDataLoader(data_folder=data_path).load(
            split="train"
        )

        new_qrels_train = remap_query_answers(qrels_train, doc_ids_to_integers)

        write_supervised_file(
            queries_train, new_qrels_train, data_path, "trn_supervised.csv"
        )
    else:
        print(
            f"BEIR Dataset {dataset} doesn't come with a train split, returning None for the trn_supervised path."
        )

    new_qrels_test = remap_query_answers(qrels_test, doc_ids_to_integers)

    write_supervised_file(queries_test, new_qrels_test, data_path, "tst_supervised.csv")

    trn_supervised = (
        f"{dataset}/trn_supervised.csv"
        if os.path.exists(data_path + "/qrels/train.tsv")
        else None
    )

    return (
        f"{dataset}/unsupervised.csv",
        trn_supervised,
        f"{dataset}/tst_supervised.csv",
        n_classes,
    )
