import hashlib
import os
from pathlib import Path

import requests
from requests.exceptions import HTTPError


def chunks(path: Path):
    def get_name(dir_entry: os.DirEntry):
        return Path(dir_entry.path).name

    if path.is_dir():
        for entry in sorted(os.scandir(path), key=get_name):
            yield bytes(Path(entry.path).name, "utf-8")
            for chunk in chunks(Path(entry.path)):
                yield chunk
    elif path.is_file():
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                yield chunk


def hash_path(path: Path):
    # Create a SHA-256 hash object
    sha256_hash = hashlib.sha256()
    if not path.exists():
        raise ValueError("Cannot hash an invalid path.")
    for chunk in chunks(path):
        sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def get_directory_size(directory: Path):
    size = 0
    for root, dirs, files in os.walk(directory):
        for name in files:
            size += os.stat(Path(root) / name).st_size
    return size


def http_get_with_error(*args, **kwargs):
    """Makes an HTTP GET request and raises an error if status code is not
    200.
    """

    try:
        response = requests.get(*args, **kwargs)
        response.raise_for_status()  # Raises HTTPError for bad status codes, e.g. 4XX or 5XX codes
        return response
    except HTTPError as http_err:
        server_message = (
            f" Server message: {http_err.response.text}"
            if http_err.response.text
            else ""
        )
        raise HTTPError(
            f"HTTP error occurred: {http_err.response.status_code} - {http_err.response.reason}.{server_message}"
        )
