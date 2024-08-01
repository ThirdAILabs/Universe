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
    import psycopg2
    from urllib.parse import urlparse

    def create_database_if_not_exists(db_uri):
        parsed_uri = urlparse(db_uri)
        db_name = parsed_uri.path[1:]
        db_user = parsed_uri.username
        db_password = parsed_uri.password
        db_host = parsed_uri.hostname
        db_port = parsed_uri.port

        conn_params = {
            "dbname": "postgres",
            "user": db_user,
            "password": db_password,
            "host": db_host,
            "port": db_port,
        }

        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        cursor = conn.cursor()

        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()

        if not exists:
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Database {db_name} created successfully.")
        else:
            print(f"Database {db_name} already exists.")

        cursor.close()
        conn.close()

    try:
        create_database_if_not_exists(db_uri)

        command = f'psql "{db_uri}" -f {dump_file_path}'

        process = subprocess.run(command, shell=True, check=True)
        print("Database restored successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to restore database: {e}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def create_s3_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config

    aws_access_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret_access_key = os.getenv("AWS_ACCESS_SECRET")

    print(f"AWS_ACCESS_KEY={aws_access_key}")
    print(f"AWS_ACCESS_SECRET={aws_secret_access_key}")

    config = Config(
        retries={"max_attempts": 10, "mode": "standard"},
        connect_timeout=5,
        read_timeout=60,
    )
    if not aws_access_key or not aws_secret_access_key:
        config.signature_version = Config(signature_version=UNSIGNED)
        s3_client = boto3.client("s3", config=config)
    else:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_access_key,
            config=config,
        )
    return s3_client


def download_files_from_s3(bucket_name, local_dir):
    import boto3

    s3_client = create_s3_client()
    os.makedirs(local_dir, exist_ok=True)

    try:
        db_backup_file_name = "db_backup.sql"
        db_backup_local_path = os.path.join(local_dir, db_backup_file_name)
        s3_client.download_file(bucket_name, db_backup_file_name, db_backup_local_path)

        restore_postgres_db_from_file(os.getenv("DATABASE_URI"), db_backup_local_path)

        response = s3_client.list_objects_v2(
            Bucket=bucket_name, Prefix="model_and_data/"
        )
        if "Contents" in response:
            print("Starting download of model_and_data folder contents...")
            for obj in response["Contents"]:
                file_name = obj["Key"]
                local_file_path = os.path.join(local_dir, file_name)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                s3_client.download_file(bucket_name, file_name, local_file_path)
                print(f"Downloaded {file_name} to {local_file_path}")
        else:
            print("No contents found in 'model_and_data/' folder.")
    except Exception as e:
        print(f"An error occurred during download: {str(e)}")
