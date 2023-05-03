from pathlib import Path

import pytest
import thirdai

from thirdai_python_package.demos import bert_base_uncased as bert_base_uncased_wrapped

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
    try:
        set_working_license()
    except AttributeError as e:
        # Ignore this, since it just means our package was not built with licensing
        pass


@pytest.fixture(scope="session")
def download_bert_tokenizer():
    return bert_base_uncased_wrapped()
