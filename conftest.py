from pathlib import Path

import pytest
import thirdai

universe_dir = Path(__file__).parent


def set_working_license():
    thirdai.licensing.set_path(
        str(
            (
                universe_dir
                / "licensing"
                / "licenses"
                / "full_license_expires_mar_2024"
            ).resolve()
        )
    )


# Automatically sets a working license file before we run any tests
@pytest.fixture(scope="session", autouse=True)
def enable_full_access_licensing(request):
    set_working_license()
