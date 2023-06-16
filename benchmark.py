import random
import string
import time

import matplotlib.pyplot as plt
import numpy as np
from thirdai import bolt, bolt_v2, dataset


def random_samples_bolt(n_samples, dim, nonzeros):
    indices = np.arange(dim)

    samples = []
    for _ in range(n_samples):
        np.random.shuffle(indices)

        vec = dataset.make_sparse_vector(indices[:nonzeros], np.ones(nonzeros))

        samples.append([bolt_v2.nn.Tensor(vec, dim)])

    return samples


def avg_predict_time_bolt(model, nonzeros, dim=100000, n_samples=1000):
    samples = random_samples_bolt(n_samples=n_samples, dim=dim, nonzeros=nonzeros)

    bolt_model = model._get_model()

    start = time.perf_counter()
    for sample in samples:
        bolt_model.forward(sample, use_sparsity=True)
    end = time.perf_counter()

    return (nonzeros, 1000 * (end - start) / len(samples))


def run_experiment_bolt():
    path = "./build/scifact_model.bolt"

    model = bolt.UniversalDeepTransformer.load(path)
    model._get_model().summary()

    for nonzeros in [10, 100, 1000, 10000]:
        n_nonzeros, avg_time = avg_predict_time_bolt(model, nonzeros=nonzeros)
        print(f"nonzeros={n_nonzeros}\tavg_predict_time={avg_time}")


def random_word(length):
    return "".join(random.choices(string.ascii_letters, k=length))


def random_words(n_words):
    lengths = [4, 5, 6, 7]
    random_lengths = random.choices(lengths, k=n_words)
    return " ".join([random_word(length) for length in random_lengths])


def avg_predict_time_udt(model, n_words, n_samples=1000):
    samples = [{"QUERY": random_words(n_words)} for _ in range(n_samples)]

    start = time.perf_counter()
    for sample in samples:
        model.predict(sample, sparse_inference=False)
    end = time.perf_counter()

    return (n_words, 1000 * (end - start) / len(samples))


def plot_results(n_words, avg_times):
    plt.plot(n_words, avg_times)
    plt.xlabel("words in input")
    plt.ylabel("avg predict time (ms)")
    plt.title("words in input vs predict time")
    plt.show()


def run_experiment_udt():
    path = "./build/scifact_model.bolt"

    # model = bolt.UniversalDeepTransformer.load(path)

    model = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(contextual_encoding="local"),
            "DOC_ID": bolt.types.categorical(delimiter=":"),
        },
        target="DOC_ID",
        n_target_classes=5000,
        integer_target=True,
        options={
            "extreme_classification": True,
            "extreme_output_dim": 50_000,
        },
    )

    model._get_model().summary()

    # n_words = list(range(10, 350, 20))  + list(range(350, 1000, 50))
    n_words = [10, 100, 1000, 10000]
    avg_times = []
    for words in n_words:
        _, avg_time = avg_predict_time_udt(model, words)
        avg_times.append(avg_time)
        print(f"num_words={words}\tavg_predict_time={avg_time}")

    # plot_results(n_words, avg_times)


# run_experiment_bolt()
run_experiment_udt()
