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
