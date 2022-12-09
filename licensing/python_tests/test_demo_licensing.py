import pytest
import thirdai
from model_test_utils import get_udt_census_income_model
from test_udt_simple import make_simple_trained_model
from thirdai.demos import download_census_income

pytestmark = [pytest.mark.release]

# I created this key on Keygen, it should be good only for the census data
CENSUS_KEY = "RRR9-XT7L-F7NH-TJYN-NCAM-9YTR-HUKL-PMAT"


def test_census_key_works_on_census():
    thirdai.activate(CENSUS_KEY)
    train_filename, _, _ = download_census_income()
    model = get_udt_census_income_model()
    model.train(train_filename, epochs=5, learning_rate=0.01)


def test_census_key_fails_on_others():
    thirdai.activate(CENSUS_KEY)
    with pytest.raises(
        RuntimeError,
        match="This dataset is not authorized under this license.",
    ):
        make_simple_trained_model()


# This fixture removes the stored access key after each test finishes, ensuring
# that other tests that run in this pytest environment will get a clean
# licensing slate
@pytest.fixture(autouse=True)
def set_license_back_to_valid():
    # The yield means that pytest will wait until the test finishes to run
    # the code below it
    yield
    thirdai.deactivate()
