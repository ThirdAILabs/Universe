import numpy as np
from matplotlib import pyplot as plt
from thirdai.bolt import NWS, NWE, Hash, Kernel
import typing as tp
from tqdm import tqdm


def mae(truth, approx):
    return np.mean(np.abs(np.array(truth) - np.array(approx)))


def mape(truth, approx):
    return np.mean(np.nan_to_num(np.abs((np.array(truth) - np.array(approx)) / np.array(truth))))


def train_and_predict_nws(h: Hash, train_inputs: np.array, train_outputs: np.array, test_inputs: np.array):
    nws = NWS(hash=h, val_dim=1)
    nws.train_parallel(train_inputs, train_outputs.reshape((-1, 1)), threads=10)
    return np.array(nws.predict(test_inputs)).reshape((-1,))


def train_and_predict_nwe(kernel: Kernel, train_inputs: np.array, train_outputs: np.array, test_inputs: np.array):
    nwe = NWE(kernel)
    nwe.train(train_inputs, train_outputs)
    return np.array(nwe.predict(test_inputs))


def random_samples(inputs, outputs, num_samples, seed):
    np.random.seed(seed)
    rows = np.random.randint(inputs.shape[0], size=num_samples)
    return inputs[rows,:], outputs[rows]


def train_and_predict_rs(kernel: Kernel, train_inputs: np.array, train_outputs: np.array, test_inputs: np.array, num_samples: int):
    train_inputs, train_outputs = random_samples(train_inputs, train_outputs, num_samples, seed=314)
    return train_and_predict_nwe(kernel, train_inputs, train_outputs, test_inputs)


def run(hash_factories: tp.List[tp.Callable[[int], Hash]], kernel: Kernel, train_inputs: np.array, train_outputs: np.array, test_inputs: np.array):
    input_dim = train_inputs.shape[-1]
    print("Input dim:", input_dim)

    hashes = [make_hash(input_dim) for make_hash in hash_factories]
    print("Finding truth...")
    truth = train_and_predict_nwe(kernel, train_inputs, train_outputs, test_inputs)
    print("Approximating truth...")
    approxes = [train_and_predict_nws(h, train_inputs, train_outputs, test_inputs) for h in tqdm(hashes)]
    # 4 = 4 bytes in a float, 2 = number of race sketches in NWS
    mems = [h.rows() * h.range() * 4 * 2 for h in hashes]
    
    print("Approximating truth with random samples")
    # +1 for output
    num_samples_for_mems = [int(mem / 4 / (input_dim + 1)) for mem in mems]
    random_approxes = [train_and_predict_rs(kernel, train_inputs, train_outputs, test_inputs, num_samples) for num_samples in num_samples_for_mems]
    random_mems = [min(num_rows, len(train_inputs)) * 4 * (input_dim + 1) for num_rows in num_samples_for_mems]
    
    print("Plotting...")
    maes = [mae(truth, approx) for approx in approxes]
    random_maes = [mae(truth, random_approx) for random_approx in random_approxes]
    plt.plot(mems, maes, '.-', label="NWS")
    plt.plot(random_mems, random_maes, '.-', label="Random Sampling")
    plt.xlabel("Memory (Bytes)")
    plt.ylabel("MAE")
    plt.legend()
    plt.show()
    
    plt.plot(mems, maes, '.-', label="NWS")
    plt.plot(random_mems, random_maes, '.-', label="Random Sampling")
    plt.xlabel("Memory (Bytes)")
    plt.ylabel("Log MAE")
    plt.yscale("log")
    plt.legend()
    plt.show()
    
    mapes = [mape(truth, approx) for approx in approxes]
    random_mapes = [mape(truth, random_approx) for random_approx in random_approxes]
    plt.plot(mems, mapes, '.-', label="NWS")
    plt.plot(random_mems, random_mapes, '.-', label="Random Sampling")
    plt.xlabel("Memory (Bytes)")
    plt.ylabel("MAPE")
    plt.legend()
    plt.show()
    
    plt.plot(mems, mapes, '.-', label="NWS")
    plt.plot(random_mems, random_mapes, '.-', label="Random Sampling")
    plt.xlabel("Memory (Bytes)")
    plt.ylabel("Log MAPE")
    plt.yscale("log")
    plt.legend(loc='best', ncol=2)
    plt.show()
