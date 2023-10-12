import hashlib
import math
import random
import urllib.parse
from dataclasses import dataclass
from typing import Optional

SUPPORTED_EXT = [".pdf", ".docx", ".csv"]


class Credentials:
    def __init__(self, username: str, password: str) -> None:
        """
        Creates the Base object for taking credentials as input
        """
        self._username = username
        self._password = password

    @property
    def username(self):
        return self._username

    def with_configs(self, **kwargs):
        raise NotImplementedError()


class SqlCredentials(Credentials):
    def __init__(self, username: str, password: str) -> None:
        """
        Creates the cred object with SQL credentials as input
        """
        super().__init__(username, password)

    def with_configs(self, host: str, port: int, database_name: str, table_name: str):
        """
        Configs to initialize a SQL Database connection
        Args:
            host: str, the hostname or IP address of the database server
            port: int, the port number for the database connection
            database_name: str, the name of the specific database to connect to
        """
        self._host = host
        self._port = port
        self._database_name = database_name
        self._table_name = table_name

    def get_db_url(self):
        db_url = f"postgresql://{self._username}:{self._password}@{self._host}:{self._port}/{self._database_name}"
        return db_url


class SharePointCredentials(Credentials):
    def __init__(self, username: str, password: str) -> None:
        """
        Creates the cred object with sharepoint credentials as input
        """
        super().__init__(username, password)

    def with_configs(self, site_url: str, library_path="Shared Documents"):
        """
        Configs to fetch the documents from sharepoint
        Args:
            site_url: URL or web address that points to a specific SharePoint site or site collection. E.g: https://<organization>.sharepoint.com/sites/yourSite
            library_path: location of the sharepoint folder. Default is 'Shared Documents'
        """
        self._site_url = urllib.parse.quote(site_url)
        self._library_path = library_path


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
