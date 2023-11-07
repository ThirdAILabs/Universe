import hashlib
import math
import random
from functools import wraps

DIRECTORY_CONNECTOR_SUPPORTED_EXT = ["pdf", "docx", "pptx", "txt", "eml"]
SUPPORTED_EXT = ["csv"] + DIRECTORY_CONNECTOR_SUPPORTED_EXT


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


# This decorator is used to raise a NotImplemented error if the check_func returns false. This is used for scenarios when a Funciton is not implemented for a particular class depending upon a condition
def requires_condition(
    check_func, method_name: str, method_class: str, condition_string: str = None
):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if check_func(self):
                print("We are running the func")
                return func(self, *args, **kwargs)
            else:
                exception_string = (
                    f"The function {method_name} is not implemented for the class"
                    f" {method_class}"
                )
                if condition_string:
                    exception_string += f" {condition_string}"
                print(exception_string)
                raise NotImplementedError(exception_string)

        return wrapper

    return decorator
