import os
import pathlib

CACHE_DIR = pathlib.Path.home() / ".cache" / "thirdai"


def _unzip_targz(targz_path, destination_parent_folder):
    os.system(f"tar -xzf {targz_path} -C {destination_parent_folder}")


def _download_file(url, download_path):
    os.system(f"curl -L {url} -o {download_path}")


def ensure_targz_installed(download_url, unzipped_dir_name):
    global CACHE_DIR
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    unzipped_dir_path = CACHE_DIR / unzipped_dir_name
    targz_download_path = CACHE_DIR / "temp.tar.gz"

    cached = False
    if not unzipped_dir_path.exists():
        print(
            f"Downloading {download_url}, which will be unzipped and saved in directory {unzipped_dir_path}"
        )

        _download_file(download_url, download_path=targz_download_path)
        _unzip_targz(targz_download_path, destination_parent_folder=CACHE_DIR)
        targz_download_path.unlink()

        if not unzipped_dir_path.exists():
            raise ValueError(
                "Unzipped directory name was different than anticipated, caching will not work as expected."
            )

    else:
        cached = True
        print(f"{unzipped_dir_path} already exists, skipping download.")

    return unzipped_dir_path, cached
