import time
import numpy as np


def run_and_time(runnable, verbose=True):
    start = time.time()
    result = runnable()
    end = time.time()
    if verbose:
        print(f"Finished in {end - start} seconds.")
    return result, end - start


def accuracy(predictions, truth):
    assert len(predictions) == len(truth)
    return sum(np.array(predictions) == np.array(truth)) / len(truth)


