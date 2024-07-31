import hashlib
import json
import os
import shutil
import sys
import time
from functools import wraps
from pathlib import Path
from urllib.parse import urljoin

import requests
from IPython.display import clear_output


def print_progress_dots(duration: int):
    for _ in range(duration):
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(1)
    clear_output(wait=True)


def create_model_identifier(model_name: str, author_username: str):
    return author_username + "/" + model_name


def create_deployment_identifier(
    model_identifier: str, deployment_name: str, deployment_username: str
):
    return model_identifier + ":" + deployment_username + "/" + deployment_name


def construct_deployment_url(host, deployment_id):
    return urljoin(host, deployment_id) + "/"


def check_deployment_decorator(func):
    """
    A decorator function to check if deployment is complete before executing the decorated method.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The decorated function.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except requests.RequestException as e:
            print(f"Error during HTTP request: {str(e)}")
            print(
                "Deployment might not be complete yet. Call `list_deployments()` to check status of your deployment."
            )
            return None

    return wrapper


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


def check_response(response):
    if not (200 <= response.status_code < 300):
        print(response.content)
        raise requests.exceptions.HTTPError(
            "Failed with status code:", response.status_code
        )

    content = json.loads(response.content)
    print(content)

    status = content["status"]

    if status != "success":
        error = content["message"]
        raise requests.exceptions.HTTPError(f"error: {error}")


def http_get_with_error(*args, **kwargs):
    """Makes an HTTP GET request and raises an error if status code is not
    2XX.
    """
    response = requests.get(*args, **kwargs)
    print("Response GET:", response.json())
    check_response(response)
    return response


def http_post_with_error(*args, **kwargs):
    """Makes an HTTP POST request and raises an error if status code is not
    2XX.
    """
    response = requests.post(*args, **kwargs)
    print("Response POST:", response.json())
    check_response(response)
    return response


def http_delete_with_error(*args, **kwargs):
    """Makes an HTTP POST request and raises an error if status code is not
    2XX.
    """
    response = requests.delete(*args, **kwargs)
    check_response(response)
    return response


def zip_folder(folder_path):
    shutil.make_archive(folder_path, "zip", folder_path)
    return str(folder_path) + ".zip"


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


def restore_postgres_db_from_file(db_uri, dump_file_path):
    import subprocess

    try:
        command = f'psql "{db_uri}" -f {dump_file_path}'

        process = subprocess.run(command, shell=True, check=True)
        print("Database restored successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to restore database: {e}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def download_files_from_s3(bucket_name, local_dir):
    import boto3

    s3_client = boto3.client("s3")
    os.makedirs(local_dir, exist_ok=True)

    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        if "Contents" in response:
            for obj in response["Contents"]:
                file_name = obj["Key"]
                local_file_path = os.path.join(local_dir, file_name)
                s3_client.download_file(bucket_name, file_name, local_file_path)
                print(f"Downloaded {file_name} to {local_file_path}")

                if file_name == "db_backup.sql":
                    restore_postgres_db_from_file(
                        os.getenv("DATABASE_URI"), local_file_path
                    )
    except Exception as e:
        print(f"An error occurred during download: {str(e)}")
