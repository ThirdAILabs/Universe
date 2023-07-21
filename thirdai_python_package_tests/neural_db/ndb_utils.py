import os

import pytest


@pytest.fixture
def create_simple_dataset():
    filename = "simple.csv"
    with open(filename, "w") as file:
        file.writelines(["text,id\n", "apples are red,0\n", "spinach is green,1\n"])

    yield filename

    os.remove(filename)
