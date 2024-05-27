from pathlib import Path

import pytest
import thirdai

from thirdai_python_package.demos import bert_base_uncased

universe_dir = Path(__file__).parent


def set_working_license():
    try:
        thirdai.licensing.deactivate()
        thirdai.licensing.set_path(
            str(
                (
                    universe_dir
                    / "licensing"
                    / "licenses"
                    / "full_license_expires_mar_2025"
                ).resolve()
            )
        )
    except AttributeError as e:
        # Ignore this, since it just means our package was not built with licensing
        pass


# Automatically sets a working license file before we run any module.
# We need this to handle tests that leave the license in an unwanted state for future tests
@pytest.fixture(scope="module", autouse=True)
def enable_full_access_licensing_module(request):
    set_working_license()


# Automatically sets a working license file before we run a test session.
# We need this because some test fixtures that load models have session scopes,
# so this needs to be called before those fixtures
@pytest.fixture(scope="session", autouse=True)
def enable_full_access_licensing_session(request):
    set_working_license()


@pytest.fixture(scope="session")
def download_bert_base_uncased():
    return bert_base_uncased()
