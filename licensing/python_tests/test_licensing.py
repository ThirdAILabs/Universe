# Add a release test marker for all tests in this file (the tests will only
# work when built in release mode)
import pytest

pytestmark = [pytest.mark.release]


def this_should_require_a_license_search():

    from thirdai import search

    search.DocRetrieval(
        centroids=[[0.0]], hashes_per_table=1, num_tables=1, dense_input_dimension=1
    )


def this_should_require_a_license_bolt():

    from thirdai import bolt

    bolt.Network(
        layers=[
            bolt.FullyConnected(
                dim=256, activation_function=bolt.ActivationFunctions.ReLU
            )
        ],
        input_dim=10,
    )


from pathlib import Path

dir_path = Path(__file__).resolve().parent
valid_license_path = dir_path / "license.serialized"
nonexisting_license_path = dir_path / "nonexisting_license.serialized"
expired_license_path = dir_path / "expired_license.serialized"
invalid_license_path = dir_path / "invalid_license.serialized"


def test_with_valid_license():
    import thirdai

    thirdai.set_thirdai_license_path(str(valid_license_path))
    this_should_require_a_license_search()
    this_should_require_a_license_bolt()


def test_with_expired_license():
    import thirdai

    thirdai.set_thirdai_license_path(str(expired_license_path))
    with pytest.raises(Exception, match=r".*license file is expired.*"):
        this_should_require_a_license_search()
    with pytest.raises(Exception, match=r".*license file is expired.*"):
        this_should_require_a_license_bolt()


def test_with_invalid_license():
    import thirdai

    thirdai.set_thirdai_license_path(str(invalid_license_path))
    with pytest.raises(Exception, match=r".*license verification failure.*"):
        this_should_require_a_license_search()
    with pytest.raises(Exception, match=r".*license verification failure.*"):
        this_should_require_a_license_bolt()
