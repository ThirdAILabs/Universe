import json
from distutils.version import StrictVersion

import requests

OLD_MODEL_PATH = "./old_models"


def get_package_versions(package_name):
    url = "https://pypi.org/pypi/%s/json" % (package_name,)
    data = json.loads(requests.get(url).content)
    versions = list(data["releases"].keys())
    versions.sort(key=StrictVersion, reverse=True)
    return versions


def get_filtered_versions():
    with open("thirdai.version") as version_file:
        full_version = version_file.read().strip()
        minor_version = ".".join(full_version.split(".")[:-1]) + "."

    versions = get_package_versions("thirdai")

    filtered_versions = [
        version
        for version in versions
        if version[: len(minor_version)] == minor_version
    ]

    return filtered_versions
