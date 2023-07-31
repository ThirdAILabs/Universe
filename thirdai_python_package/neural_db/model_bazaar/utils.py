import hashlib
import os
import shutil
from pathlib import Path

import requests
from tqdm import tqdm


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
    2XX.
    """
    try:
        response = requests.get(*args, **kwargs)
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as err:
        raise requests.exceptions.HTTPError(
            f"{response.status_code} error: {err}, content: {response.content}"
        )


def http_post_with_error(*args, **kwargs):
    """Makes an HTTP POST request and raises an error if status code is not
    2XX.
    """
    try:
        response = requests.post(*args, **kwargs)
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as err:
        raise requests.exceptions.HTTPError(
            f"{response.status_code} error: {err}, content: {response.content}"
        )


def zip_folder(folder_path):
    shutil.make_archive(folder_path, "zip", folder_path)
    return str(folder_path) + ".zip"


def upload_file(upload_url, filepath):
    chunk_size = 1024 * 1024 * 4  # 4MiB chunk size by default
    path = Path(filepath)
    total_size = path.stat().st_size
    filename = path.name

    """
    Follow https://learn.microsoft.com/en-us/rest/api/storageservices/put-block-list?tabs=azure-ad,
    to understand the below implementation.
    """

    with tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
    ) as bar:
        with open(filepath, "rb") as f:
            block_ids = []
            chunk_number = 0
            while True:
                chunk_data = f.read(chunk_size)
                if not chunk_data:
                    break

                chunk_number += 1
                block_id = f"{chunk_number:08x}"
                block_url = f"{upload_url}&comp=block&blockid={block_id}"
                response = requests.put(block_url, data=chunk_data)

                if response.status_code not in [201, 202]:
                    raise ValueError(
                        f"Block upload failed with status code: {response.status_code}"
                    )

                block_ids.append(block_id)
                bar.update(len(chunk_data))

    # Commit the block list to finalize the upload
    commit_url = f"{upload_url}&comp=blocklist"
    block_list_xml = "".join([f"<Latest>{block}</Latest>" for block in block_ids])
    requests.put(
        commit_url,
        data=f"<?xml version='1.0' encoding='utf-8'?><BlockList>{block_list_xml}</BlockList>",
    )


def get_file_size(file_path, unit="B"):
    file_size = os.path.getsize(file_path)
    exponents_map = {"B": 0, "KB": 1, "MB": 2, "GB": 3}
    if unit not in exponents_map:
        raise ValueError(
            "Must select from \
        ['B', 'KB', 'MB', 'GB']"
        )

    size = file_size / 1024 ** exponents_map[unit]
    return round(size, 3)
