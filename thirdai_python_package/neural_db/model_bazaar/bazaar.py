import json
import os
import shutil
from pathlib import Path
from typing import Callable, List, Optional, Union
from urllib.parse import urljoin

import requests
from pydantic import BaseModel, ValidationError
from thirdai.neural_db.models import CancelState

from thirdai import neural_db

from .utils import get_directory_size, hash_path, http_get_with_error


class BazaarEntry(BaseModel):
    model_name: str
    trained_on: str
    num_params: int
    size: int
    size_in_memory: int
    hash: str
    identifier: str
    domain: str
    description: str = None
    is_indexed: bool = False
    publish_date: str
    author_email: str
    author_username: str
    access_level: str = "public"
    thirdai_version: str

    @staticmethod
    def from_dict(entry):
        return BazaarEntry(
            model_name=entry["model_name"],
            trained_on=entry["trained_on"],
            num_params=entry["num_params"],
            size=entry["size"],
            size_in_memory=entry["size_in_memory"],
            hash=entry["hash"],
            identifier=entry["saved_path"],
            domain=entry["domain"],
            description=entry["description"],
            is_indexed=entry["is_indexed"],
            publish_date=entry["publish_date"],
            author_email=entry["author_email"],
            author_username=entry["author_username"],
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


def relative_path_depth(child_path: Path, parent_path: Path):
    child_path, parent_path = child_path.resolve(), parent_path.resolve()
    relpath = os.path.relpath(child_path, parent_path)
    if relpath == ".":
        return 0
    else:
        return 1 + relpath.count(os.sep)


class Bazaar:
    def __init__(
        self,
        cache_dir: Union[Path, str] = Path("./bazaar_cache"),
        base_url="https://staging-modelzoo.azurewebsites.net/api/",
    ):
        cache_dir = Path(cache_dir)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self._cache_dir = cache_dir
        self._base_url = base_url
        self._login_instance = None

    def fetch(
        self,
        name: str = "",
        domain: Optional[str] = None,
        username: Optional[str] = None,
    ):
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

    def fetch_meta_cache_model(self):
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

                bazaar_entries.append(bazaar_entry)

                dirnames.clear()  # Don't descend any further

            elif depth > 2:
                # We're too deep, don't process this directory
                dirnames.clear()

        return bazaar_entries

    def list_model_names(self):
        return [entry.identifier for entry in self.fetch()]

    def get_neuraldb(
        self,
        model_name: str,
        author_username: str,
        on_progress: Callable = lambda *args, **kwargs: None,
        cancel_state: CancelState = CancelState(),
    ):
        model_dir = self.get_model_dir(
            model_name, author_username, on_progress, cancel_state
        )
        return neural_db.NeuralDB.from_checkpoin(checkpoint_path=model_dir)

    def get_model_dir(
        self,
        model_name: str,
        author_username: str,
        on_progress: Callable = lambda *args, **kwargs: None,
        cancel_state: CancelState = CancelState(),
    ):
        identifier = f"{author_username}/{model_name}"

        url = urljoin(
            self._base_url,
            "bazaar/model",
        )
        response = http_get_with_error(
            url,
            params={"saved_path": identifier},
        )

        json_entry = json.loads(response.content)["data"]
        bazaar_entry = BazaarEntry.bazaar_entry_from_json(json_entry)

        cached_model_dir = self._model_dir_in_cache(
            identifier=identifier, fetched_bazaar_entry=bazaar_entry
        )
        if cached_model_dir:
            return cached_model_dir

        self._download(
            identifier,
            on_progress=on_progress,
            cancel_state=cancel_state,
        )
        if not cancel_state.is_canceled():
            return self._unpack_and_remove_zip(identifier)
        else:
            try:
                shutil.rmtree(self._cached_checkpoint_dir(identifier))
            except:
                pass
            return None

    # The checkpoint dir is cache_dir/author_username/model_name/
    # This is the parent directory for the three paths defined in the following methods
    def _cached_checkpoint_dir(self, identifier: str):
        return self._cache_dir / identifier

    # The ndb path is cache_dir/author_username/model_name/model.ndb
    def _cached_model_dir_path(self, identifier: str):
        return self._cached_checkpoint_dir(identifier) / "model.ndb"

    # The ndb zip download path is cache_dir/author_username/model_name/model.zip
    def _cached_model_zip_path(self, identifier: str):
        return self._cached_checkpoint_dir(identifier) / "model.zip"

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
        identifier: str,
        on_progress: Callable,
        cancel_state: CancelState,
    ):
        signing_url = urljoin(
            self._base_url,
            f"bazaar/public-download",
        )
        signing_response = http_get_with_error(
            signing_url,
            params={"saved_path": identifier},
        )
        try:
            shutil.rmtree(self._cached_checkpoint_dir(identifier))
        except:
            pass
        os.makedirs(self._cached_checkpoint_dir(identifier))

        model_metadata = json.loads(signing_response.content)["data"]
        download_url = model_metadata["url"]
        model_metadata.pop("url", None)
        with open(self._cached_model_metadata_path(identifier), "w") as f:
            json.dump(model_metadata, f)

        destination = self._cached_model_zip_path(identifier)

        # Streaming, so we can iterate over the response.
        response = requests.get(download_url, allow_redirects=True, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024 * 4  # 4MiB
        size_so_far = 0

        with open(destination, "wb") as file:
            for data in response.iter_content(block_size):
                if cancel_state.is_canceled():
                    break
                size_so_far += len(data)
                on_progress(size_so_far / total_size_in_bytes)
                file.write(data)

        if cancel_state.is_canceled():
            try:
                shutil.rmtree(self._cached_checkpoint_dir(identifier))
            except:
                pass
        else:
            if size_so_far != total_size_in_bytes:
                raise ValueError("Failed to download.")
