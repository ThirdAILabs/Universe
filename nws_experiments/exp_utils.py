import numpy as np
from matplotlib import pyplot as plt
from thirdai.bolt import NWS, NWE, SKA, Hash, Kernel, Distance, RACE
import typing as tp
from tqdm import tqdm
import math

import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker
import os
from pathlib import Path


CUR_DIR = Path(os.path.dirname(__file__))



class SquareRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    """

    name = "sqrt"

    def __init__(self, axis, **kwargs):
        # note in older versions of matplotlib (<3.1), this worked fine.
        # mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        mscale.ScaleBase.__init__(self, axis)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(0.0, vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a) ** 0.5

        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a) ** 2

        def inverted(self):
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()


mscale.register_scale(SquareRootScale)


def mae(truth, approx):
    errors = np.abs(np.array(truth) - np.array(approx))
    return sorted(errors)[math.ceil(len(truth) * 0.99)]


def mape(truth, approx):
    # errors = np.nan_to_num(
    #     np.abs((np.array(truth) - np.array(approx)) / np.array(truth)), nan=1.0
    # )
    print(truth)
    print(approx)
    min_idx = np.argmin(truth)
    print("min truth", truth[min_idx], "vs min approx", approx[min_idx])
    max_idx = np.argmax(truth)
    print("max truth", truth[max_idx], "vs max approx", approx[max_idx])
    print("++++++++++++++++++++++++++")
    errors = np.abs((np.array(truth) - np.array(approx)) / np.array(truth))
    return sorted(errors)[math.ceil(len(truth) * 0.99)]


def train_and_predict_nws(
    h: Hash, train_inputs: np.array, train_outputs: np.array, test_inputs: np.array, sparse: bool
):
    nws = NWS(hash=h, sparse=sparse)
    nws.train(train_inputs, train_outputs)
    return np.array(nws.predict(test_inputs)), nws.bytes()


def train_and_predict_nwe(
    kernel: Kernel,
    train_inputs: np.array,
    train_outputs: np.array,
    test_inputs: np.array,
):
    nwe = NWE(kernel)
    nwe.train(train_inputs, train_outputs)
    return np.array(nwe.predict(test_inputs))


def random_samples(inputs, outputs, num_samples, seed):
    np.random.seed(seed)
    rows = np.random.randint(inputs.shape[0], size=num_samples)
    return inputs[rows, :], outputs[rows]


def train_and_predict_rs(
    kernel: Kernel,
    train_inputs: np.array,
    train_outputs: np.array,
    test_inputs: np.array,
    num_samples: int,
):
    train_inputs, train_outputs = random_samples(
        train_inputs, train_outputs, num_samples, seed=314
    )
    return train_and_predict_nwe(kernel, train_inputs, train_outputs, test_inputs)


def train_and_predict_ska(
    ska: SKA, kernel: Kernel, test_inputs: np.array, num_samples: int
):
    ska.use(num_samples)
    train_inputs, train_outputs = ska.used_samples()
    return train_and_predict_nwe(kernel, train_inputs, train_outputs, test_inputs)


def run(
    hash_factories: tp.List[tp.Callable[[int], Hash]],
    kernel: Kernel,
    distance: Distance,
    train_inputs: np.array,
    train_outputs: np.array,
    test_inputs: np.array,
    random_sampling=True,
    sparse=False,
):
    input_dim = train_inputs.shape[-1]
    print("Input dim:", input_dim)

    hashes = [make_hash(input_dim) for make_hash in hash_factories]
    print("Finding truth...")
    truth = train_and_predict_nwe(kernel, train_inputs, train_outputs, test_inputs)
    
    print(truth)
    print("min truth", np.min(truth))
    print("max truth", np.max(truth))

    print("Approximating truth...")
    approxes_and_mems = [
        train_and_predict_nws(h, train_inputs, train_outputs, test_inputs, sparse)
        for h in tqdm(hashes)
    ]

    approxes, mems = zip(*approxes_and_mems)
    
    mapes = [mape(truth, approx) for approx in approxes]
    plt.plot([h.rows() ** 0.5 for h in hashes], mapes, ".-", label="NWS")
    plt.plot(range(1, 50), [1 / r for r in range(1, 50)], "-.", label="1/√R")
    plt.xlabel("√R")
    plt.ylabel("Relative error")
    plt.legend()
    plt.savefig(CUR_DIR / f"assets/nws_relative_error_with_root_r.png")
    plt.clf()

    maes = [mae(truth, approx) for approx in approxes]
    plt.plot([h.rows() ** 0.5 for h in hashes], maes, ".-", label="NWS")
    plt.plot(range(1, 50), [1 / r for r in range(1, 50)], "-.", label="1/√R")
    plt.xlabel("√R")
    plt.ylabel("Absolute error")
    plt.legend()
    plt.savefig(CUR_DIR / f"assets/nws_absolute_error_with_root_r.png")
    plt.clf()

    plt.plot(mems, mapes, ".-", label="NWS")
    plt.xlabel("Memory (Bytes)")
    plt.ylabel("Relative error")
    plt.legend()
    plt.show()
    plt.savefig(CUR_DIR / f"assets/nws_relative_error_over_memory.png")
    plt.clf()

    plt.plot(mems, mapes, ".-", label="NWS")
    plt.xlabel("Memory (Bytes)")
    plt.ylabel("Relative error")
    plt.xscale("sqrt")
    plt.legend()
    plt.savefig(CUR_DIR / f"assets/nws_absolute_error_over_memory.png")
    plt.clf()

    if random_sampling:
        print("Approximating truth with random samples...")
        # +1 for output
        num_samples_for_mems = [min(int(mem / 4 / (input_dim + 1)), len(train_inputs)) for mem in mems]
        end = len(num_samples_for_mems)
        for i in range(1, len(num_samples_for_mems)):
            if num_samples_for_mems[-1 - i] == num_samples_for_mems[-1]:
                end -=1
            else:
                break
        num_samples_for_mems = num_samples_for_mems[:end]
        random_approxes = [
            train_and_predict_rs(
                kernel, train_inputs, train_outputs, test_inputs, num_samples
            )
            for num_samples in tqdm(num_samples_for_mems)
        ]
        random_mems = [
            min(num_rows, len(train_inputs)) * 4 * (input_dim + 1)
            for num_rows in num_samples_for_mems
        ]

    if distance:
        print("Approximating truth with SKA samples...")
        ska = SKA(distance, train_inputs, train_outputs)
        ska_approxes = [
            train_and_predict_ska(ska, kernel, test_inputs, num_samples)
            for num_samples in tqdm(num_samples_for_mems)
        ]
        ska_mems = [
            min(num_rows, len(train_inputs)) * 4 * (input_dim + 1)
            for num_rows in num_samples_for_mems
        ]

    print("Plotting...")
    # maes = [mae(truth, approx) for approx in approxes]
    # plt.plot(mems, maes, '.-', label="NWS")
    # if random_sampling:
    #     random_maes = [mae(truth, random_approx) for random_approx in random_approxes]
    #     plt.plot(random_mems, random_maes, '.-', label="Random Sampling")
    # if distance:
    #     ska_maes = [mae(truth, ska_approx) for ska_approx in ska_approxes]
    #     plt.plot(ska_mems, ska_maes, '.-', label="Sparse Kernel Approximation")
    # plt.xlabel("Memory (Bytes)")
    # plt.ylabel("MAE")
    # plt.xscale("sqrt")
    # plt.legend()
    # plt.show()

    # plt.plot(mems, maes, '.-', label="NWS")
    # if random_sampling:
    #     plt.plot(random_mems, random_maes, '.-', label="Random Sampling")
    # if distance:
    #     plt.plot(ska_mems, ska_maes, '.-', label="Sparse Kernel Approximation")
    # plt.xlabel("Memory (Bytes)")
    # plt.ylabel("Log MAE")
    # plt.yscale("log")
    # plt.xscale("sqrt")
    # plt.legend()
    # plt.show()

    plt.plot(mems, mapes, ".-", label="NWS")
    if random_sampling:
        random_mapes = [mape(truth, random_approx) for random_approx in random_approxes]
        plt.plot(random_mems, random_mapes, ".-", label="Random Sampling")
    if distance:
        ska_mapes = [mape(truth, ska_approx) for ska_approx in ska_approxes]
        plt.plot(ska_mems, ska_mapes, ".-", label="Sparse Kernel Approximation")
    plt.xlabel("Memory (Bytes)")
    plt.ylabel("Relative error")
    plt.xscale("sqrt")
    plt.legend()
    plt.savefig(CUR_DIR / f"assets/all_relative_error_over_memory.png")
    plt.clf()
