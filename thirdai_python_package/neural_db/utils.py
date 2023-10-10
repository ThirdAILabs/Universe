import hashlib
import math
import urllib.parse
import random

SUPPORTED_EXT = [".pdf", ".docx", ".csv"]

class ClientCredentials:
    def __init__(
        self, username: str, password: str, host: str, port: int, database_name: str
    ):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.database_name = database_name
        
    def __init__(
        self, username: str, password: str, site_url: str, library_path: str = "Shared Documents"
    ):
        """
            Creates Object of ClientCredential to fetch the documents from sharepoint
            Args: 
                username: Registered username or email ID
                password: password for the registered account
                site_url: URL or web address that points to a specific SharePoint site or site collection. E.g: https://<organization>.sharepoint.com/sites/yourSite
                library_path: location of the sharepoint folder. Defaults to 'Shared Documents'
        """
        self._username = username
        self._password = password
        self._site_url = urllib.parse.quote(site_url)
        self._library_path = library_path
        
    def get_db_url(self):
        db_url = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}"
        return db_url
    

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
