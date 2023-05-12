import pytest
import thirdai
from licensing_utils import (
    deactivate_license_at_start_of_demo_test,
    run_udt_training_routine,
)

pytestmark = [pytest.mark.release]

# Note that neither of these tests check what happens when we don't activate.
# at all. This is because these tests are running in CI with a valid license
# file accessible and when we fall back to the license file method it will still
# work. However, the fact that test_file_licensing.py passes, which includes bad
# license file checks, shows that when we don't activate Keygen we do indeed
# fall back to license files.

# I created this key on Keygen, and it is good for a while
GOOD_KEY = "94TN-9LUT-KXWK-K4VE-CPEW-3U9K-3R7H-HREL"

# I created this key on Keygen and let it expire
EXPIRED_KEY = "78BF4E-1EACCA-3432A5-D633E2-7B182B-V3"

# I created this key on Keygen and revoked it
SUSPENDED_KEY = "9R3F-KLNJ-M3M4-KWLW-9E9E-7TNT-4FXH-V7R9"

# This key is not on Keygen
NONEXISTENT_KEY = "THIS-IS-A-VERY-NONEXISTENT-KEY"

# This is a key on Keygen good for a while that does not allow model save
# loading, but that is otherwise full access
NO_SAVE_LOAD_KEY = "EU7E-HLRJ-LXW4-TMHX-7AUC-AFFR-WJAK-9CTW"


def test_keygen_good_key():
    thirdai.licensing.activate(GOOD_KEY)
    run_udt_training_routine()


def test_expired_key():
    with pytest.raises(
        RuntimeError,
        match=r".*returned the following message: is expired",
    ):
        thirdai.licensing.activate(EXPIRED_KEY)


def test_suspended_key():
    with pytest.raises(
        RuntimeError,
        match=r".*returned the following message: is suspended",
    ):
        thirdai.licensing.activate(SUSPENDED_KEY)


def test_nonexistent_key():
    with pytest.raises(
        RuntimeError,
        match=r".*returned the following message: does not exist",
    ):
        thirdai.licensing.activate(NONEXISTENT_KEY)


def test_no_save_load_key():
    thirdai.licensing.activate(NO_SAVE_LOAD_KEY)
    run_udt_training_routine(do_save_load=False)

    with pytest.raises(
        Exception,
        match=r"Saving and loading of models is not authorized under this license",
    ):
        run_udt_training_routine()
