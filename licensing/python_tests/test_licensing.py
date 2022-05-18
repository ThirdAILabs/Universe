# Add a release test marker for all tests in this file (the tests will only
# work when built in release mode)
import pytest

pytestmark = [pytest.mark.release]


def this_should_require_a_license():

    from thirdai import bolt, search

    bolt.Network(
        layers=[
            bolt.FullyConnected(
                dim=256, activation_function=bolt.ActivationFunctions.ReLU
            )
        ],
        input_dim=10,
    )

    from thirdai import search

    search.DocRetrieval(
        centroids=[[0.0]], hashes_per_table=1, num_tables=1, dense_input_dimension=1
    )


import os
from pathlib import Path

dir_path = Path(__file__).resolve().parent
valid_license_path = dir_path / "license.serialized"
nonexisting_license_path = dir_path / "nonexisting_license.serialized"
expired_license_path = dir_path / "expired_license.serialized"
invalid_license_path = dir_path / "invalid_license.serialized"


def test_with_valid_license():
    os.environ["THIRDAI_LICENSE_PATH"] = str(valid_license_path)
    this_should_require_a_license()


def test_with_no_license():
    os.environ["THIRDAI_LICENSE_PATH"] = str(nonexisting_license_path)
    with pytest.raises(Exception, match=r".*no license file found.*"):
        this_should_require_a_license()


def test_with_expired_license():
    os.environ["THIRDAI_LICENSE_PATH"] = str(expired_license_path)
    with pytest.raises(Exception, match=r".*license file is expired.*"):
        this_should_require_a_license()


def test_with_invalid_license():
    os.environ["THIRDAI_LICENSE_PATH"] = str(invalid_license_path)
    with pytest.raises(Exception, match=r".*license verification failure.*"):
        this_should_require_a_license()
