import pytest
import thirdai
from licensing_utils import this_should_require_a_license_bolt

pytestmark = [pytest.mark.release]

# Note that neither of these tests check what happens when we don't activate.
# at all. This is because these tests are running in CI with a valid license
# file accessible and when we fall back to the license file method it will still
# work. However, the fact that test_file_licensing.py passes, which includes bad
# license file checks, shows that when we don't activate Keygen we do indeed
# fall back to license files.
# TODO(Josh): Consider how to refactor CI to test this more clearly.

# I created this key on Keygen, and it is good for a while
GOOD_KEY = "94TN-9LUT-KXWK-K4VE-CPEW-3U9K-3R7H-HREL"

# I created this key on Keygen and let it expire
EXPIRED_KEY = "78BF4E-1EACCA-3432A5-D633E2-7B182B-V3"

# I created this key on Keygen and revoked it
SUSPENDED_KEY = "9R3F-KLNJ-M3M4-KWLW-9E9E-7TNT-4FXH-V7R9"

# This key is not on Keygen
NONEXISTENT_KEY = "THIS-IS-A-VERY-NONEXISTENT-KEY"


def test_keygen_good_key():
    thirdai.licensing.activate(GOOD_KEY)
    this_should_require_a_license_bolt()


def test_expired_key():
    thirdai.licensing.activate(EXPIRED_KEY)
    with pytest.raises(
        RuntimeError,
        match=r".*returned the following message: is expired",
    ):
        this_should_require_a_license_bolt()


def test_suspended_key():
    thirdai.licensing.activate(SUSPENDED_KEY)
    with pytest.raises(
        RuntimeError,
        match=r".*returned the following message: is suspended",
    ):
        this_should_require_a_license_bolt()


def test_nonexistent_key():
    thirdai.licensing.activate(NONEXISTENT_KEY)
    with pytest.raises(
        RuntimeError,
        match=r".*returned the following message: does not exist",
    ):
        this_should_require_a_license_bolt()


# This fixture removes the stored access key after each test finishes, ensuring
# that other tests that run in this pytest environment will get a clean
# licensing slate
@pytest.fixture(autouse=True)
def set_license_back_to_valid():
    # The yield means that pytest will wait until the test finishes to run
    # the code below it
    yield
    thirdai.licensing.deactivate()
