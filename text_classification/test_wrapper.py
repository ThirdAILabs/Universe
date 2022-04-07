import pytest
from sentiment_wrapper import *
from datasets import load_dataset_builder
from datasets import load_dataset
import re
from collections import namedtuple

name = "yelp_review_full"
content = "text"
label = "label"
train_file_path = "./" + name + "_train.svm"
test_file_path = "./" + name + "_test.svm"
label_dict = {0: 0, 1: 0, 2: 0, 3: -1, 4: 1, 5: 1}
model_path = "./sentiment_pretrained_yelp_cp"
murmur_dim = 100000
seed = 42


def download_dataset(
    name, svm_path_1, svm_path_2, content_name, label_name, extra=None
):
    """
    Download and proprocess a dataset using Huggingface api
    Need to specify the dataset name, content name, label name for
    each dataset. You can find the specifics of each dataset on the
    Huggingface website. https://huggingface.co/datasets
    """
    dataset_1 = load_dataset(name, extra, split="train")
    dataset_2 = load_dataset(name, extra, split="test")

    fw_1 = open(svm_path_1, "w")
    fw_2 = open(svm_path_2, "w")

    for data_set, fw in zip([dataset_1, dataset_2], [fw_1, fw_2]):
        for data in data_set:
            # Map the star rating (0~5) to a sentiment label(0~1)
            original_label = label_dict[data[label_name]]
            if original_label == -1:
                continue
            label = str(original_label)
            fw.write(str(label) + " ")

            sentence = data[content_name]
            # Remove punctuations and convert to lowercase
            sentence = re.sub(r"[^\w\s]", "", sentence)
            sentence = sentence.lower()

            # Tokenize the sentence and featurized it
            tup = dataset.bolt_tokenizer(sentence, seed=seed, dimension=murmur_dim)
            for idx, val in zip(tup[0], tup[1]):
                fw.write(str(idx) + ":" + str(val) + " ")

            fw.write("\n")

    fw_1.close()
    fw_2.close()


def train(
    args,
    train_fn,
    accuracy_threshold,
    epoch_time_threshold=600,
    total_time_threshold=2000,
):
    """
    A generic bolt training test function imported from our mnist test.
    """
    final_accuracies = []
    final_epoch_times = []
    total_times = []

    for _ in range(args.runs):

        final_accuracy, accuracies_per_epoch, time_per_epoch = train_fn(args)
        final_accuracies.append(final_accuracy)
        final_epoch_times.append(time_per_epoch[-1])
        total_times.append(sum(time_per_epoch))

        print(
            f"Result of training {args.dataset} for {args.epochs} epochs:\n\tFinal epoch accuracy: {final_accuracy}\n\tFinal epoch time: {time_per_epoch}"
        )

        assert final_accuracies[-1] > accuracy_threshold
        assert final_epoch_times[-1] < epoch_time_threshold
        assert total_times[-1] < total_time_threshold

    return final_accuracies, final_epoch_times


def train_yelp(args):
    """
    An example of the "train_fn" in train()
    """
    layers = [
        bolt.LayerConfig(
            dim=2000,
            load_factor=args.sparsity,
            activation_function=bolt.ActivationFunctions.ReLU,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=args.hashes_per_table,
                num_tables=args.num_tables,
                range_pow=args.hashes_per_table * 3,
                reservoir_size=64,
            ),
        ),
        bolt.LayerConfig(
            dim=2,
            load_factor=1.0,
            activation_function=bolt.ActivationFunctions.Softmax,
        ),
    ]

    train_data = dataset.load_bolt_svm_dataset(train_file_path, 1024)
    test_data = dataset.load_bolt_svm_dataset(test_file_path, 256)

    network = bolt.Network(layers=layers, input_dim=100000)
    epoch_times = []
    epoch_accuracies = []

    for _ in range(args.epochs):
        times = network.train(
            train_data,
            bolt.CategoricalCrossEntropyLoss(),
            args.lr,
            1,
            rehash=6400,
            rebuild=128000,
        )
        epoch_times.append(times["epoch_times"][0])
        acc, _ = network.predict(
            test_data, metrics=["categorical_accuracy"], verbose=False
        )
        epoch_accuracies.append(acc["categorical_accuracy"][0])

    network.save(model_path)

    return epoch_accuracies[-1], epoch_accuracies, epoch_times


@pytest.mark.unit
def test_train_yelp():
    """
    Benchmark for sentiment prediction on the yelp_review_full dataset
    """
    download_dataset(name, train_file_path, test_file_path, content, label)

    args = {
        "dataset": name,
        "epochs": 1,
        "lr": 0.0001,
        "sparsity": 0.1,
        "hashes_per_table": 4,
        "num_tables": 64,
        "runs": 1,
    }
    args = namedtuple("args", args.keys())(*args.values())
    train(args, train_yelp, 0.88)


@pytest.mark.unit
def test_predict_sentence_sentiment():
    """
    Test to make sure that the prediction wrapper is working properly
    """
    sentiment_analysis_network = bolt.Network.load(filename=model_path)
    assert (
        predict_sentence_sentiment(
            sentiment_analysis_network, "I love this great product very much", seed=seed
        )
        == 1
    )
    assert (
        predict_sentence_sentiment(
            sentiment_analysis_network,
            "I hate this terrible product, not worth it",
            seed=seed,
        )
        == 0
    )


@pytest.mark.unit
def test_preprocess():
    """
    Test to make sure that the preprocess function is working properly
    """
    rows = [
        ["pos", "I love this great product very much"],
        ["neg", "I hate this terrible product, not worth it"],
    ]
    with open("mock.csv", "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    location = preprocess_data("mock.csv", is_train=False)

    sentiment_analysis_network = bolt.Network.load(filename=model_path)
    test_data = dataset.load_bolt_svm_dataset(location, 256)
    acc, _ = sentiment_analysis_network.predict(
        test_data, metrics=["categorical_accuracy"], verbose=False
    )
    assert acc["categorical_accuracy"][0] > 0.5
