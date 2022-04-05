import pytest
from sentiment_wrapper import *
from datasets import load_dataset_builder
from datasets import load_dataset
from sklearn.utils import murmurhash3_32 as mmh3
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
    dataset_1 = load_dataset(name, extra, split="train")
    dataset_2 = load_dataset(name, extra, split="test")

    fw_1 = open(svm_path_1, "w")
    fw_2 = open(svm_path_2, "w")
    f_lst = (fw_1, fw_2)
    d_lst = (dataset_1, dataset_2)

    for data_ind in range(2):
        data_set = d_lst[data_ind]
        fw = f_lst[data_ind]
        for data in data_set:
            label_ori = label_dict[data[label_name]]
            if label_ori == -1:
                continue
            label = str(label_ori)
            fw.write(str(label) + " ")

            sentence = data[
                content_name
            ]  # "content" for dbpedia & amazon_polarity, "text" for ag_news
            sentence = re.sub(r"[^\w\s]", "", sentence)
            sentence = sentence.lower()

            tup = dataset.bolt_tokenizer(sentence)
            for idx, val in zip(tup[0], tup[1]):
                fw.write(str(idx) + ":" + str(val) + " ")

            fw.write("\n")

    fw_1.close()
    fw_2.close()


def train(
    args,
    train_fn,
    accuracy_threshold,
    epoch_time_threshold=150,
    total_time_threshold=10000,
):
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
            test_data, metrics=["categorical_accuracy"], verbose=True
        )
        epoch_accuracies.append(acc["categorical_accuracy"][0])

    network.save(model_path)

    return epoch_accuracies[-1], epoch_accuracies, epoch_times


@pytest.mark.unit
def test_train_yelp():
    download_dataset(name, train_file_path, test_file_path, content, label)

    args = {
        "dataset": name,
        "epochs": 2,
        "lr": 0.0001,
        "sparsity": 0.1,
        "hashes_per_table": 4,
        "num_tables": 64,
        "runs": 1,
    }
    args = namedtuple("args", args.keys())(*args.values())
    train(args, train_yelp, 0.9)


@pytest.mark.unit
def test_predict_sentence_sentiment():
    sentiment_analysis_network = bolt.Network.load(filename=model_path)
    assert (
        predict_sentence_sentiment(
            sentiment_analysis_network, "I love this great product very much"
        )
        == 1
    )
    assert (
        predict_sentence_sentiment(
            sentiment_analysis_network, "I hate this terrible product, not worth it"
        )
        == 0
    )
