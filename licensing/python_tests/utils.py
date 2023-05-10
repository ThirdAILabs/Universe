import pytest


@pytest.fixture(autouse=True)
def deactivate_license_at_start_of_demo_test():
    import thirdai

    thirdai.licensing.deactivate()
