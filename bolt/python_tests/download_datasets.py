import pytest
import os


@pytest.fixture(scope="session")
def download_mnist():
    TRAIN_FILE = "mnist"
    TEST_FILE = "mnist.t"
    if not os.path.exists(TRAIN_FILE):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 --output mnist.bz2"
        )
        os.system("bzip2 -d mnist.bz2")

    if not os.path.exists(TEST_FILE):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2 --output mnist.t.bz2"
        )
        os.system("bzip2 -d mnist.t.bz2")

    return TRAIN_FILE, TEST_FILE
