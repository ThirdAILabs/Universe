import hashlib
import math
import os
import pickle
import random
import shutil
from functools import wraps
from pathlib import Path

import numpy as np

DIRECTORY_CONNECTOR_SUPPORTED_EXT = ["pdf", "docx", "pptx", "txt", "eml"]
SUPPORTED_EXT = ["csv"] + DIRECTORY_CONNECTOR_SUPPORTED_EXT


def convert_str_to_path(str_path):
    if isinstance(str_path, str):
        return Path(str_path)
    elif isinstance(str_path, Path):
        return str_path
    else:
        raise TypeError(
            "Error converting to Path. Expected the type a 'str' or 'pathlib.Path', but"
            f" received: {type(str_path)}"
        )


def pickle_to(obj: object, filepath: Path):
    with open(filepath, "wb") as pkl:
        pickle.dump(obj, pkl)


def unpickle_from(filepath: Path):
    with open(filepath, "rb") as pkl:
        obj = pickle.load(pkl)
    return obj


def assert_file_exists(path: Path):
    if not path:
        raise ValueError("Path cannot be none")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {str(path)}")


def clean_text(text):
    return text.encode("utf-8", "replace").decode("utf-8").lower()


def hash_file(path: str, metadata=None):
    """https://stackoverflow.com/questions/22058048/hashing-a-file-in-python"""
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

    sha1 = hashlib.sha1()

    with open(path, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)

    if metadata:
        sha1.update(str(metadata).encode())

    return sha1.hexdigest()


def hash_string(string: str):
    sha1 = hashlib.sha1(bytes(string, "utf-8"))
    return sha1.hexdigest()


def random_sample(sequence, k):
    if len(sequence) > k:
        return random.sample(sequence, k)
    mult_factor = math.ceil(k / len(sequence))
    return (sequence * mult_factor)[:k]


def move_between_directories(src, dest):
    import os
    import shutil

    # gather all files
    allfiles = os.listdir(src)

    # iterate on all files to move them to destination folder
    for f in allfiles:
        src_path = os.path.join(src, f)
        dst_path = os.path.join(dest, f)
        shutil.move(src_path, dst_path)


def delete_folder(path: Path, ignore_errors: bool = True):
    shutil.rmtree(path, ignore_errors=ignore_errors)


def delete_file(path: Path, ignore_errors: bool = True):
    try:
        os.remove(path)
    except:
        if not ignore_errors:
            raise


# This decorator is used to raise a NotImplemented error if the check_func returns false. This is used for scenarios when a Funciton is not implemented for a particular class depending upon a condition
def requires_condition(
    check_func, method_name: str, method_class: str, condition_unmet_string: str = None
):
    def decorator(func):
        error_message = (
            f"The property {method_name} is not implemented for the class"
            f" {method_class}{condition_unmet_string if condition_unmet_string else ''}"
        )
        if isinstance(func, property):  # If the decorator is applied to a property.

            def wrapped_fget(self):
                if check_func(self):
                    return func.fget(self)
                else:
                    raise NotImplementedError(error_message)

            return property(wrapped_fget, func.fset, func.fdel, func.__doc__)

        @wraps(func)
        def wrapper(self, *args, **kwargs):  # If the decorator is applied to a method.
            if check_func(self):
                return func(self, *args, **kwargs)
            else:
                raise NotImplementedError(error_message)

        return wrapper

    return decorator


def normalize_scores(results):
    if len(results) == 0:
        return results
    if len(results) == 1:
        return [(results[0][0], 1.0, results[0][2])]
    ids, scores, retriever = zip(*results)
    scores = np.array(scores)
    scores -= np.min(scores)
    scores /= np.max(scores)
    return list(zip(ids, scores, retriever))


def merge_results(results_a, results_b, k):
    results_a = normalize_scores(results_a)
    results_b = normalize_scores(results_b)
    results = []
    cache = set()

    min_len = min(len(results_a), len(results_b))
    for a, b in zip(results_a, results_b):
        if a[0] not in cache:
            results.append(a)
            cache.add(a[0])
        if b[0] not in cache:
            results.append(b)
            cache.add(b[0])

    if len(results) < k:
        for i in range(min_len, len(results_a)):
            if results_a[i][0] not in cache:
                results.append(results_a[i])
        for i in range(min_len, len(results_b)):
            if results_b[i][0] not in cache:
                results.append(results_b[i])

    return results[:k]


def add_retriever_tag(results, tag):
    return [[(id, score, tag) for id, score in result] for result in results]
