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


def run(hash_factories: tp.List[tp.Callable[[int], Hash]], kernel: Kernel, train_inputs: np.array, train_outputs: np.array, test_inputs: np.array):
    hashes = [make_hash(train_inputs.shape[-1]) for make_hash in hash_factories]
    print("Finding truth...")
    truth = train_and_predict_nwe(kernel, train_inputs, train_outputs, test_inputs)
    print("Approximating truth...")
    approxes = [train_and_predict_nws(h, train_inputs, train_outputs, test_inputs) for h in tqdm(hashes)]
    print("Plotting...")
    # 4 = 4 bytes in a float, 2 = number of race sketches in NWS
    mems = [h.rows() * h.range() * 4 * 2 for h in hashes]
    
    maes = [mae(truth, approx) for approx in approxes]
    plt.plot(mems, maes)
    plt.xlabel("Memory (Bytes)")
    plt.ylabel("MAE")
    plt.show()
    
    plt.plot(mems, np.log(np.array(maes)))
    plt.xlabel("Memory (Bytes)")
    plt.ylabel("Log MAE")
    plt.show()
    
    mapes = [mape(truth, approx) for approx in approxes]
    plt.plot(mems, mapes)
    plt.xlabel("Memory (Bytes)")
    plt.ylabel("MAPE")
    plt.show()
    
    plt.plot(mems, np.log(np.array(mapes)))
    plt.xlabel("Memory (Bytes)")
    plt.ylabel("Log MAPE")
    plt.show()
