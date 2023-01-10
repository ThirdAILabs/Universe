# Add a release test marker for all tests in this file (the tests will only
# work when built in release mode)
import platform

import pytest
from licensing_utils import this_should_require_a_license_bolt

pytestmark = [pytest.mark.release]


def this_should_require_a_license_search():

    from thirdai import search

    search.DocRetrieval(
        centroids=[[0.0]], hashes_per_table=1, num_tables=1, dense_input_dimension=1
    )


from pathlib import Path

dir_path = Path(__file__).resolve().parent
valid_license_path = dir_path / "license.serialized"
nonexisting_license_path = dir_path / "nonexisting_license.serialized"
expired_license_path = dir_path / "expired_license.serialized"
invalid_license_path = dir_path / "invalid_license.serialized"


def test_with_valid_license():
    import thirdai

    thirdai.licensing.set_path(str(valid_license_path))
    this_should_require_a_license_search()
    this_should_require_a_license_bolt()


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="TSK-568: Expired license currently does an access violation on Windows",
)
def test_with_expired_license():
    import thirdai

    thirdai.licensing.set_path(str(expired_license_path))
    with pytest.raises(Exception, match=r".*license file is expired.*"):
        this_should_require_a_license_search()
    with pytest.raises(Exception, match=r".*license file is expired.*"):
        this_should_require_a_license_bolt()


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="TSK-568: Invalid license currently does an access violation on Windows",
)
def test_with_invalid_license():
    import thirdai

    thirdai.licensing.set_path(str(invalid_license_path))
    with pytest.raises(Exception, match=r".*license verification failure.*"):
        this_should_require_a_license_search()
    with pytest.raises(Exception, match=r".*license verification failure.*"):
        this_should_require_a_license_bolt()


# See e.g. https://stackoverflow.com/questions/34931263/how-to-run-specific-code-after-all-tests-are-executed
# This sets the license back to a valid value after each tests run
@pytest.fixture(autouse=True)
def set_license_back_to_valid():
    import thirdai

    # The yield means that it will run AFTER each test, not before.
    yield
    thirdai.licensing.set_path(str(valid_license_path))
