import pytest
import shutil

# MockModel.tar.gz is a zipped directory containing just an empty pytorch.bin
# file and an empty centroids.npy file. We expect an error to be thrown
# when we try to load the checkpoint, but the error's message will tell us
# that the download itself was successful.
MOCK_URL = "https://www.dropbox.com/s/mfijqowbhe3uy3y/MockModel.tar.gz?dl=0"
MOCK_DIR_NAME = "MockModel"


@pytest.mark.unit
def test_dropbox_model_download():

    global MOCK_URL, MOCK_DIR_NAME

    from thirdai import embeddings

    try:
        test = embeddings.DocSearchModel(
            local_path=None, download_metadata=(MOCK_URL, MOCK_DIR_NAME)
        )
    except Exception as e:
        assert "is not a valid JSON file" in str(e)
        return

    assert False


@pytest.mark.unit
def test_model_caching():

    global MOCK_URL, MOCK_DIR_NAME

    from thirdai._download import ensure_targz_installed

    download_path, _ = ensure_targz_installed(
        download_url=MOCK_URL, unzipped_dir_name=MOCK_DIR_NAME
    )
    shutil.rmtree(download_path)

    _, cached_before = ensure_targz_installed(
        download_url=MOCK_URL, unzipped_dir_name=MOCK_DIR_NAME
    )
    _, cached_after = ensure_targz_installed(
        download_url=MOCK_URL, unzipped_dir_name=MOCK_DIR_NAME
    )

    assert cached_before == False
    assert cached_after == True
