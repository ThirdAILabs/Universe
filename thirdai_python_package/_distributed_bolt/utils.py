import importlib
from functools import wraps
from time import time

from thirdai import data, logging


def get_num_cpus():
    try:
        import multiprocessing

        return multiprocessing.cpu_count()
    except ImportError:
        print("Could not find num_cpus, setting num_cpus to DEFAULT=1")
        return 1


def check_torch_installed():
    try:
        importlib.import_module("torch")
    except ImportError as e:
        raise ImportError(
            "Distributed Bolt requires Torch Distributed as its communication backend. Please ensure that Torch is installed to enable distributed training with Bolt.t"
        ) from e


def timed(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start = time()
        result = f(*args, **kwds)
        elapsed = time() - start
        logging.info("func %s | time %d ms" % (f.__name__, elapsed * 1000))
        return result

    return wrapper
