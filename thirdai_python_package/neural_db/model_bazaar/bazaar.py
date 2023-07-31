import json
import os
import shutil
from pathlib import Path
from typing import Callable, List, Optional
from urllib.parse import urljoin

import requests
from neural_db.neural_db import NeuralDB as ndb
from pydantic import BaseModel
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

from .utils import (
    get_directory_size,
    get_file_size,
    hash_path,
    http_get_with_error,
    http_post_with_error,
    upload_file,
    zip_folder,
)


class BazaarEntry(BaseModel):
    model_name: str
    trained_on: str
    num_params: str
    size: int
    hash: str
    identifier: str
    domain: str
    description: str
    is_indexed: str
    publish_date: str
    author_email: str
    author_username: str
    access_level: str

    @staticmethod
    def from_dict(entry):
        return BazaarEntry(
            model_name=entry["model_name"],
            trained_on=entry["trained_on"],
            num_params=entry["num_params"],
            size=entry["size"],
            hash=entry["hash"],
            identifier=entry["saved_path"],
            domain=entry["domain"],
            description=entry["description"],
            is_indexed=entry["is_indexed"],
            publish_date=entry["publish_date"],
            author_email=entry["author_email"],
            author_username=entry["author_username"],
            access_level=entry["access_level"],
        )


class Login:
    def __init__(
        self,
        email: str,
        password: str,
        base_url: str = "https://staging-modelzoo.azurewebsites.net/",
    ):
        self._base_url = base_url
        # We are using HTTPBasic Auth in backend. update this when we change the Authentication in Backend.
        response = http_get_with_error(
            urljoin(self._base_url, "user/email-login"),
            auth=HTTPBasicAuth(email, password),
        )

        content = json.loads(response.content)
        self._access_token = content["data"]["access_token"]

        self._user_id = content["data"]["user"]["user_id"]

    @property
    def access_token(self):
        return self._access_token

    @property
    def user_id(self):
        return self._user_id

    @property
    def base_url(self):
        return self._base_url


def auth_header(access_token):
    return {
        "Authorization": f"Bearer {access_token}",
    }


# Use this decorator for any function to enforce users use only after login.
def login_required(func):
    def wrapper(self, *args, **kwargs):
        if not self.is_logged_in():
            raise PermissionError(
                "You have to login to use this functionality. try '.login()' method."
            )
        return func(self, *args, **kwargs)

    return wrapper


class Bazaar:
    def __init__(
        self,
        cache_dir: Path = Path("./bazaar_cache"),
        base_url="https://staging-modelzoo.azurewebsites.net/",
    ):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self._cache_dir = cache_dir
        # registry stores all the Modelbazaar objects fetched so far.
        self._registry = {}
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
        if not self._login_instance:
            self._login_instance = Login(
                email=email, password=password, base_url=self._base_url
            )
        else:
            print(f"Already logged in")

    def is_logged_in(self):
        return self._login_instance != None

    def fetch(
        self,
        name: str = "",
        domain: Optional[str] = None,
        username: Optional[str] = None,
        access_level: Optional[List[str]] = None,
    ):
        if self.is_logged_in():
            url = urljoin(
                self._login_instance._base_url,
                f"bazaar/{self._login_instance._user_id}/list",
            )
            response = http_get_with_error(
                url,
                params={
                    "name": name,
                    "domain": domain,
                    "username": username,
                    "access_level": access_level,
                },
                headers=auth_header(self._login_instance._access_token),
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
        entries = [BazaarEntry.from_dict(entry) for entry in json_entries]
        for entry in entries:
            self._registry[entry.identifier] = entry

        return entries

    def list_model_names(self):
        self.fetch()
        return list(self._registry.keys())

    def neuraldb_object(self, checkpoint_path: str):
        db = ndb(user_id="user")
        db.from_checkpoint(checkpoint_path)
        return db

    def get_neuraldb(
        self,
        model_name: str,
        author_username: str,
        on_progress: Callable = lambda *args, **kwargs: None,
    ):
        results = self.fetch(name=model_name, username=author_username)
        if not results:
            raise ValueError(
                f"There is no model with given model name {model_name} and user {author_username}, try .fetch() to see the available models."
            )
        identifier = results[0].identifier
        cached_model_dir = self._model_dir_in_cache(identifier)
        if cached_model_dir:
            return self.neuraldb_object(cached_model_dir)

        if self._cached_model_zip_path(identifier).is_file():
            self._unpack_and_remove_zip(identifier)
            cached_model_dir = self._model_dir_in_cache(identifier)
            if cached_model_dir:
                return self.neuraldb_object(cached_model_dir)

        self._download(identifier, on_progress=on_progress)
        folder_path = self._unpack_and_remove_zip(identifier)
        return self.neuraldb_object(folder_path)

    def _cached_model_dir_path(self, identifier: str):
        return self._cache_dir / identifier

    def _cached_model_zip_path(self, identifier: str):
        return self._cache_dir / f"{identifier}.zip"

    def _model_dir_in_cache(self, identifier: str):
        cached_model_dir = self._cached_model_dir_path(identifier)
        if cached_model_dir.is_dir():
            bazaar_entry = self._registry[identifier]
            hash_match = hash_path(cached_model_dir) == bazaar_entry.hash
            size_match = get_directory_size(cached_model_dir) == bazaar_entry.size
            if hash_match and size_match:
                return cached_model_dir
        return None

    def _unpack_and_remove_zip(self, identifier: str):
        zip_path = self._cached_model_zip_path(identifier)
        extract_dir = self._cached_model_dir_path(identifier)
        shutil.unpack_archive(filename=zip_path, extract_dir=extract_dir)
        os.remove(zip_path)
        return extract_dir

    def _download(self, identifier: str, on_progress: Callable):
        if self.is_logged_in():
            signing_url = urljoin(
                self._login_instance._base_url,
                f"bazaar/{self._login_instance._user_id}/download",
            )
            signing_response = http_get_with_error(
                signing_url,
                params={"saved_path": identifier},
                headers=auth_header(self._login_instance._access_token),
            )
        else:
            signing_url = urljoin(
                self._base_url,
                f"bazaar/public-download",
            )
            signing_response = http_get_with_error(
                signing_url,
                params={"saved_path": identifier},
            )
        download_url = json.loads(signing_response.content)["data"]["url"]
        destination = self._cached_model_zip_path(identifier)

        if not os.path.exists(os.path.dirname(destination)):
            os.makedirs(os.path.dirname(destination))

        # Streaming, so we can iterate over the response.
        response = requests.get(download_url, allow_redirects=True, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024 * 4  # 4MiB
        size_so_far = 0
        with open(destination, "wb") as file, tqdm(
            total=total_size_in_bytes, unit="B", unit_scale=True, desc="Downloading"
        ) as progress_bar:
            for data in response.iter_content(block_size):
                size_so_far += len(data)
                on_progress(size_so_far / total_size_in_bytes)
                file.write(data)
                progress_bar.update(len(data))

        if size_so_far != total_size_in_bytes:
            raise ValueError("Failed to download.")

    @login_required
    def push(
        self,
        name: str,
        model_path: Path,
        trained_on: str,
        num_params: str,
        is_indexed: bool = False,
        access_level: str = "public",
        description: str = None,
    ):
        zip_path = zip_folder(model_path)

        model_hash = hash_path(model_path)

        model_response = http_get_with_error(
            urljoin(
                self._login_instance._base_url,
                f"bazaar/{self._login_instance._user_id}/model-check",
            ),
            headers=auth_header(self._login_instance._access_token),
            params={"hash": str(model_hash)},
        )

        model_content = json.loads(model_response.content)

        if model_content["data"]["model_present"]:
            raise ValueError("This model is already uploaded.")

        url_response = http_get_with_error(
            urljoin(
                self._login_instance._base_url,
                f"bazaar/{self._login_instance._user_id}/upload-url",
            ),
            headers=auth_header(self._login_instance._access_token),
            params={
                "name": name,
                "size": int(get_file_size(zip_path, "MB")),
            },
        )

        upload_url = json.loads(url_response.content)["data"]["url"]

        upload_file(upload_url, zip_path)

        size = get_directory_size(model_path)

        response = http_post_with_error(
            urljoin(
                self._login_instance._base_url,
                f"bazaar/{self._login_instance._user_id}/upload-info",
            ),
            headers=auth_header(self._login_instance._access_token),
            json={
                "name": name,
                "trained_on": trained_on,
                "num_params": num_params,
                "is_indexed": is_indexed,
                "size": size,
                "hash": model_hash,
                "access_level": access_level,
                "description": description,
            },
        )

        os.remove(zip_path)
