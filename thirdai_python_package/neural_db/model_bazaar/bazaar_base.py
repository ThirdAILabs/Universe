import concurrent.futures
import json
import os
import pickle
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union
from urllib.parse import urljoin

import requests
from pydantic import BaseModel, ValidationError
from requests.auth import HTTPBasicAuth
from thirdai.neural_db.neural_db import CancelState, NeuralDB
from tqdm import tqdm

from .utils import (
    create_model_identifier,
    get_directory_size,
    get_file_size,
    hash_path,
    http_get_with_error,
    http_post_with_error,
    zip_folder,
)


class BazaarEntry(BaseModel):
    name: str
    author_username: str
    identifier: str
    trained_on: Optional[str] = None
    num_params: int
    size: int
    size_in_memory: int
    hash: str
    domain: str
    description: Optional[str] = None
    is_indexed: bool = False
    publish_date: str
    author_email: str
    access_level: str = "public"
    thirdai_version: str

    @staticmethod
    def from_dict(entry):
        return BazaarEntry(
            name=entry["model_name"],
            author_username=entry["username"],
            identifier=create_model_identifier(
                model_name=entry["model_name"], author_username=entry["username"]
            ),
            trained_on=entry["trained_on"],
            num_params=entry["num_params"],
            size=entry["size"],
            size_in_memory=entry["size_in_memory"],
            hash=entry["hash"],
            domain=entry["domain"],
            description=entry["description"],
            is_indexed=entry["is_indexed"],
            publish_date=entry["publish_date"],
            author_email=entry["user_email"],
            access_level=entry["access_level"],
            thirdai_version=entry["thirdai_version"],
        )

    @staticmethod
    def bazaar_entry_from_json(json_entry):
        try:
            loaded_entry = BazaarEntry.from_dict(json_entry)
            return loaded_entry
        except ValidationError as e:
            print(f"Validation error: {e}")
            return None


@dataclass
class Login:
    base_url: str
    username: str
    access_token: str

    @staticmethod
    def with_email(
        base_url: str,
        email: str,
        password: str,
    ):
        # We are using HTTPBasic Auth in backend. update this when we change the Authentication in Backend.
        response = http_get_with_error(
            urljoin(base_url, "user/email-login"),
            auth=HTTPBasicAuth(email, password),
        )

        content = json.loads(response.content)
        username = content["data"]["user"]["username"]
        access_token = content["data"]["access_token"]
        return Login(base_url, username, access_token)


def auth_header(access_token):
    return {
        "Authorization": f"Bearer {access_token}",
    }


def relative_path_depth(child_path: Path, parent_path: Path):
    child_path, parent_path = child_path.resolve(), parent_path.resolve()
    relpath = os.path.relpath(child_path, parent_path)
    if relpath == ".":
        return 0
    else:
        return 1 + relpath.count(os.sep)


# Use this decorator for any function to enforce users use only after login.
def login_required(func):
    def wrapper(self, *args, **kwargs):
        if not self.is_logged_in():
            raise PermissionError(
                "This method requires login, please use '.login()' first then try again."
            )
        return func(self, *args, **kwargs)

    return wrapper


class Bazaar:
    def __init__(
        self,
        base_url,
        cache_dir: Union[Path, str],
    ):
        cache_dir = Path(cache_dir)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self._cache_dir = cache_dir
        if not base_url.endswith("/api/"):
            raise ValueError("base_url must end with '/api/'.")
        self._base_url = base_url
        self._login_instance = None

    def signup(self, email, password, username):
        json_data = {
            "username": username,
            "email": email,
            "password": password,
        }

        response = http_post_with_error(
            urljoin(self._base_url, "user/email-signup-basic"),
            json=json_data,
        )

        print(
            f"Successfully signed up. Please check your email ({email}) to verify your account."
        )

    def login(self, email, password):
        self._login_instance = Login.with_email(self._base_url, email, password)

    def is_logged_in(self):
        return self._login_instance is not None

    def fetch(
        self,
        name: str = "",
        domain: Optional[str] = None,
        username: Optional[str] = None,
        access_level: Optional[List[str]] = None,
    ):
        if self.is_logged_in():
            url = urljoin(
                self._login_instance.base_url,
                "bazaar/list",
            )
            response = http_get_with_error(
                url,
                params={
                    "name": name,
                    "domain": domain,
                    "username": username,
                    "access_level": access_level,
                },
                headers=auth_header(self._login_instance.access_token),
            )
        else:
            print("Fetching public models, login to fetch all accessible models.")
            url = urljoin(
                self._base_url,
                "bazaar/public-list",
            )
            response = http_get_with_error(
                url,
                params={
                    "name": name,
                    "domain": domain,
                    "username": username,
                },
            )
        json_entries = json.loads(response.content)["data"]

        bazaar_entries = [
            BazaarEntry.bazaar_entry_from_json(json_entry)
            for json_entry in json_entries
            if json_entry
        ]
        return bazaar_entries

    def fetch_from_cache(
        self,
        name: str = "",
        domain: Optional[str] = None,
        username: Optional[str] = None,
        access_level: Optional[List[str]] = None,
        only_check_dir_exists: bool = False,
    ):
        bazaar_entries = []
        # Walk through the directories
        for dirpath, dirnames, filenames in os.walk(self._cache_dir):
            depth = relative_path_depth(
                child_path=Path(dirpath), parent_path=Path(self._cache_dir)
            )

            if depth == 2:
                # We're two levels in, which is the level of all checkpoint dirs
                split_path = dirpath.split(os.path.sep)
                model_name = split_path[-1]
                author_username = split_path[-2]

                identifier = f"{author_username}/{model_name}"
                with open(self._cached_model_metadata_path(identifier), "r") as f:
                    bazaar_entry = BazaarEntry.from_dict(json.load(f))

                if (
                    name.lower() in model_name.lower()
                    and (not username or username == author_username)
                    and (not domain or domain == bazaar_entry.domain)
                    and (not access_level or bazaar_entry.access_level in access_level)
                ):
                    try:
                        if self._model_dir_in_cache(
                            identifier=identifier,
                            fetched_bazaar_entry=bazaar_entry,
                            only_check_dir_exists=only_check_dir_exists,
                        ):
                            bazaar_entries.append(bazaar_entry)
                    except:
                        pass

                dirnames.clear()  # Don't descend any further

            elif depth > 2:
                # We're too deep, don't process this directory
                dirnames.clear()

        return bazaar_entries

    def list_model_names(self):
        return [entry.identifier for entry in self.fetch()]

    def get_neuraldb(
        self,
        model_identifier: str,
        on_progress: Callable = lambda *args, **kwargs: None,
        cancel_state: CancelState = CancelState(),
        disable_progress_bar: bool = False,
    ):
        model_dir = self.get_model_dir(
            model_identifier, on_progress, cancel_state, disable_progress_bar
        )
        return NeuralDB.from_checkpoint(checkpoint_path=model_dir)

    def get_model_dir(
        self,
        model_identifier,
        on_progress: Callable = lambda *args, **kwargs: None,
        cancel_state: CancelState = CancelState(),
        disable_progress_bar: bool = False,
    ):
        if self.is_logged_in():
            url = urljoin(
                self._login_instance.base_url,
                f"bazaar/model",
            )
            response = http_get_with_error(
                url,
                params={"model_identifier": model_identifier},
                headers=auth_header(self._login_instance.access_token),
            )
        else:
            url = urljoin(
                self._base_url,
                "bazaar/public-model",
            )
            response = http_get_with_error(
                url,
                params={"model_identifier": model_identifier},
            )

        json_entry = json.loads(response.content)["data"]
        bazaar_entry = BazaarEntry.bazaar_entry_from_json(json_entry)

        cached_model_dir = self._model_dir_in_cache(
            identifier=model_identifier, fetched_bazaar_entry=bazaar_entry
        )
        if cached_model_dir:
            return cached_model_dir

        self._download(
            model_identifier,
            on_progress=on_progress,
            cancel_state=cancel_state,
            disable_progress_bar=disable_progress_bar,
        )

        if not cancel_state.is_canceled():
            return self._unpack_and_remove_zip(model_identifier)
        else:
            try:
                shutil.rmtree(self._cached_checkpoint_dir(model_identifier))
            except:
                pass
            return None

    # The checkpoint dir is cache_dir/model_identifier/
    # This is the parent directory for the three paths defined in the following methods
    def _cached_checkpoint_dir(self, identifier: str):
        return self._cache_dir / identifier

    # The ndb path is cache_dir/model_identifier/model.ndb
    def _cached_model_dir_path(self, identifier: str):
        return self._cached_checkpoint_dir(identifier) / "model.ndb"

    # The ndb zip download path is cache_dir/model_identifier/model.ndb.zip
    def _cached_model_zip_path(self, identifier: str):
        return self._cached_checkpoint_dir(identifier) / "model.ndb.zip"

    # The BazaarEntry json metadata path is cache_dir/author_username/model_name/metadata.json
    def _cached_model_metadata_path(self, identifier: str):
        return self._cached_checkpoint_dir(identifier) / "metadata.json"

    def _model_dir_in_cache(
        self,
        identifier: str,
        fetched_bazaar_entry: str,
        only_check_dir_exists: bool = False,
    ):
        cached_model_dir = self._cached_model_dir_path(identifier)
        if cached_model_dir.is_dir():
            if not only_check_dir_exists:
                hash_match = hash_path(cached_model_dir) == fetched_bazaar_entry.hash
                size_match = (
                    get_directory_size(cached_model_dir) == fetched_bazaar_entry.size
                )
                if hash_match and size_match:
                    return cached_model_dir
            else:
                return cached_model_dir
        return None

    def _unpack_and_remove_zip(self, identifier: str):
        zip_path = self._cached_model_zip_path(identifier)
        extract_dir = self._cached_model_dir_path(identifier)
        shutil.unpack_archive(filename=zip_path, extract_dir=extract_dir)
        os.remove(zip_path)
        return extract_dir

    def _download(
        self,
        model_identifier: str,
        on_progress: Callable,
        cancel_state: CancelState,
        disable_progress_bar: bool = False,
    ):
        if self.is_logged_in():
            url = urljoin(
                self._login_instance.base_url,
                f"bazaar/download",
            )
            response = requests.get(
                url,
                params={"model_identifier": model_identifier},
                headers=auth_header(self._login_instance.access_token),
                stream=True,
            )
        else:
            url = urljoin(
                self._base_url,
                f"bazaar/public-download",
            )
            response = requests.get(
                url, params={"model_identifier": model_identifier}, stream=True
            )
        try:
            shutil.rmtree(self._cached_checkpoint_dir(model_identifier))
        except:
            pass
        os.makedirs(self._cached_checkpoint_dir(model_identifier))

        destination = self._cached_model_zip_path(model_identifier)

        # Try to get the total size from the Content-Length header
        total_size = int(response.headers.get("Content-Length", 0))

        # Set up a progress bar
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading"
        ) as bar:
            # Destination file path
            with open(destination, "wb") as f:
                for chunk in response.iter_content(8192):  # 8192 bytes or 8KB
                    f.write(chunk)
                    bar.update(len(chunk))

    def upload_chunk(self, upload_token, chunk_number, chunk_data, bar, progress_lock):
        files = {"chunk": chunk_data}
        response = requests.post(
            urljoin(
                self._login_instance.base_url,
                "bazaar/upload-chunk",
            ),
            files=files,
            params={"chunk_number": chunk_number},
            headers=auth_header(upload_token),
        )

        if response.status_code == 200:
            with progress_lock:
                # Update the progress bar
                bar.update(len(chunk_data))
        else:
            print(f"Upload failed with status code: {response.status_code}")
            print(response.text)
            return False

        return True

    @login_required
    def push(
        self,
        name: str,
        model_path: Union[Path, str],
        trained_on: str = "Own Documents",
        is_indexed: bool = False,
        access_level: str = "public",
        description: str = "",
    ):
        model_path = Path(model_path)
        zip_path = zip_folder(model_path)

        model_hash = hash_path(model_path)

        # Generate upload token
        token_response = http_get_with_error(
            urljoin(
                self._login_instance.base_url,
                f"bazaar/upload-token",
            ),
            headers=auth_header(self._login_instance.access_token),
            params={
                "model_name": name,
                "size": int(get_file_size(zip_path, "MB")),
            },
        )
        upload_token = json.loads(token_response.content)["data"]["token"]

        # Get the total file size for progress bar
        total_size = os.path.getsize(zip_path)

        # Determine the chunk size you want to upload per request
        chunk_size = 1024 * 1024  # 1 MB chunk

        # Initialize the progress bar
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=zip_path) as bar:
            # Open the file in binary mode
            with open(zip_path, "rb") as file:
                chunk_number = 0

                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    futures = []
                    progress_lock = threading.Lock()

                    while True:
                        # Read a chunk of the file
                        chunk_data = file.read(chunk_size)
                        if not chunk_data:
                            break  # End of file

                        # Increment the chunk number
                        chunk_number += 1

                        # Submit the task to the thread pool
                        future = executor.submit(
                            self.upload_chunk,
                            upload_token,
                            chunk_number,
                            chunk_data,
                            bar,
                            progress_lock,
                        )
                        futures.append(future)

                    # Collect the return status of all threads
                    threads_status = [
                        future.result()
                        for future in concurrent.futures.as_completed(futures)
                    ]

                # Check if all uploads were successful
                if all(threads_status):
                    print("File upload completed successfully.")
                else:
                    print("File upload failed.")

        db = NeuralDB.from_checkpoint(checkpoint_path=model_path)
        model = db._savable_state.model.model._get_model()
        num_params = model.num_params()
        thirdai_version = model.thirdai_version()

        size = get_directory_size(model_path)

        # TODO: Get actual size in memory when db is loaded
        # This is a temporary approximation of how much RAM a model will take.
        # Approximation comes from 4x explosion of weights in ADAM optimizer.
        udt_pickle = model_path / "model.pkl"
        documents_pickle = model_path / "documents.pkl"
        logger_pickle = model_path / "logger.pkl"
        size_in_memory = (
            os.path.getsize(udt_pickle) * 4
            + os.path.getsize(documents_pickle)
            + os.path.getsize(logger_pickle)
        )

        json_data = {
            "trained_on": trained_on,
            "num_params": num_params,
            "is_indexed": is_indexed,
            "size": size,
            "size_in_memory": size_in_memory,
            "hash": model_hash,
            "access_level": access_level,
            "description": description,
            "thirdai_version": thirdai_version,
        }

        response = http_post_with_error(
            urljoin(
                self._login_instance.base_url,
                "bazaar/upload-commit",
            ),
            params={"total_chunks": chunk_number},
            json=json_data,
            headers=auth_header(upload_token),
        )

        os.remove(zip_path)

    @login_required
    def delete(
        self,
        model_identifier: str,
    ):
        delete_response = http_post_with_error(
            urljoin(
                self._login_instance.base_url,
                f"bazaar/delete",
            ),
            headers=auth_header(self._login_instance.access_token),
            json={
                "model_identifier": model_identifier,
            },
        )

        print("Successfully requested admin to delete the model.")
