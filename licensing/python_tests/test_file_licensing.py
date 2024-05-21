# Add a release test marker for all tests in this file (the tests will only
# work when built in release mode)
import platform

import pytest
from licensing_utils import (
    deactivate_license_at_start_of_demo_test,
    run_udt_training_routine,
)

pytestmark = [pytest.mark.release]


def this_should_require_a_license_search():
    from thirdai import search

    search.DocRetrieval(
        centroids=[[0.0]], hashes_per_table=1, num_tables=1, dense_input_dimension=1
    )


def this_should_require_a_license_query_reformulation():
    from thirdai import bolt

    bolt.UniversalDeepTransformer(
        source_column="source_queries",
        target_column="target_queries",
        dataset_size="medium",
    )


from pathlib import Path

dir_path = Path(__file__).resolve().parent.parent / "licenses"
valid_license_path = dir_path / "full_license_expires_mar_2025"
nonexisting_license_path = dir_path / "nonexisting_license"
expired_license_path = dir_path / "full_expired_license"
invalid_license_1_path = dir_path / "invalid_license_1"
invalid_license_2_path = dir_path / "invalid_license_2"
no_save_load_license_path = dir_path / "no_save_load_license"
max_output_dim_100_license_path = dir_path / "max_output_dim_100_license"
max_train_samples_100_license_path = dir_path / "max_train_samples_100_license"


def test_with_valid_license():
    import thirdai

    thirdai.licensing.set_path(str(valid_license_path))
    this_should_require_a_license_search()
    run_udt_training_routine()
    this_should_require_a_license_query_reformulation()


def test_license_print(capfd):
    import thirdai

    thirdai.licensing.set_path(str(max_train_samples_100_license_path), verbose=True)
    out, err = capfd.readouterr()
    print(out)
    assert (
        out.strip()
        == "This is a license with an expiry time of 1797555168817 epoch ms. "
        "Entitlements are as follows: FULL_DATASET_ACCESS, LOAD_SAVE, "
        "MAX_OUTPUT_DIM 4294967295, MAX_TRAIN_SAMPLES 100,"
    )
    assert err == ""


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="TSK-568: Expired license currently does an access violation on Windows",
)
def test_with_expired_license():
    import thirdai

    with pytest.raises(Exception, match=r"license file is expired"):
        thirdai.licensing.set_path(str(expired_license_path))
    with pytest.raises(
        Exception,
        match=r"call either licensing.set_path, licensing.start_heartbeat, or licensing.activate with a valid license.",
    ):
        this_should_require_a_license_search()
    with pytest.raises(
        Exception,
        match=r"call either licensing.set_path, licensing.start_heartbeat, or licensing.activate with a valid license.",
    ):
        run_udt_training_routine()
    with pytest.raises(
        Exception,
        match=r"call either licensing.set_path, licensing.start_heartbeat, or licensing.activate with a valid license.",
    ):
        this_should_require_a_license_query_reformulation()


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="TSK-568: Invalid license currently does an access violation on Windows",
)
def test_with_invalid_license():
    import thirdai

    for invalid_license_path in invalid_license_1_path, invalid_license_2_path:
        with pytest.raises(Exception, match=r"license verification failure"):
            thirdai.licensing.set_path(str(invalid_license_path))
        with pytest.raises(
            Exception,
            match=r"call either licensing.set_path, licensing.start_heartbeat, or licensing.activate with a valid license.",
        ):
            this_should_require_a_license_search()
        with pytest.raises(
            Exception,
            match=r"call either licensing.set_path, licensing.start_heartbeat, or licensing.activate with a valid license.",
        ):
            run_udt_training_routine()
        with pytest.raises(
            Exception,
            match=r"call either licensing.set_path, licensing.start_heartbeat, or licensing.activate with a valid license.",
        ):
            this_should_require_a_license_query_reformulation()


def test_no_save_load_license():
    import thirdai

    thirdai.licensing.set_path(str(no_save_load_license_path))
    run_udt_training_routine(do_save_load=False)

    with pytest.raises(
        Exception,
        match=r"Saving and loading of models is not authorized under this license",
    ):
        run_udt_training_routine()


def test_restricted_output_dim_license():
    import thirdai

    thirdai.licensing.set_path(str(max_output_dim_100_license_path))
    run_udt_training_routine(n_classes=2)

    with pytest.raises(
        Exception,
        match=r"This model's output dim is too large to be allowed under this license",
    ):
        run_udt_training_routine(n_classes=102)


def test_max_train_samples_license():
    import thirdai

    thirdai.licensing.set_path(str(max_train_samples_100_license_path))
    run_udt_training_routine(num_data_points=2)

    with pytest.raises(
        Exception,
        match=r"This model has exceeded the number of training examples allowed for this license",
    ):
        run_udt_training_routine(num_data_points=102)
