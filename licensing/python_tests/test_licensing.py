# Add a release test marker for all tests in this file (the tests will only
# work when built in release mode)
import pytest

pytestmark = [pytest.mark.release]


def this_should_require_a_license_search(license_path):

    from thirdai import search

    search.DocRetrieval(
        centroids=[[0.0]],
        hashes_per_table=1,
        num_tables=1,
        dense_input_dimension=1,
        license_path=str(license_path),
    )


def this_should_require_a_license_bolt(license_path):

    from thirdai import bolt

    bolt.Network(
        layers=[
            bolt.FullyConnected(
                dim=256, activation_function=bolt.ActivationFunctions.ReLU
            )
        ],
        input_dim=10,
        license_path=str(license_path),
    )

    from thirdai import search

    search.DocRetrieval(
        centroids=[[0.0]],
        hashes_per_table=1,
        num_tables=1,
        dense_input_dimension=1,
        license_path=str(license_path),
    )


from pathlib import Path

dir_path = Path(__file__).resolve().parent
valid_license_path = dir_path / "license.serialized"
nonexisting_license_path = dir_path / "nonexisting_license.serialized"
expired_license_path = dir_path / "expired_license.serialized"
invalid_license_path = dir_path / "invalid_license.serialized"


def test_with_valid_license():
    this_should_require_a_license_search(valid_license_path)
    this_should_require_a_license_bolt(valid_license_path)


def test_with_expired_license():
    with pytest.raises(Exception, match=r".*license file is expired.*"):
        this_should_require_a_license_search(expired_license_path)
    with pytest.raises(Exception, match=r".*license file is expired.*"):
        this_should_require_a_license_bolt(expired_license_path)


def test_with_invalid_license():
    with pytest.raises(Exception, match=r".*license verification failure.*"):
        this_should_require_a_license_search(invalid_license_path)
    with pytest.raises(Exception, match=r".*license verification failure.*"):
        this_should_require_a_license_bolt(invalid_license_path)
