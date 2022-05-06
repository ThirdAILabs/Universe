import pytest
import shutil

# MockModel.tar.gz is a zipped directory containing just an empty pytorch.bin
# file, an empty centroids.npy file, and an empty confiy.json file. We expect an
# error to be thrown when we try to load the checkpoint, but the error's message
# will tell us that the download itself was successful.
MOCK_URL = "https://www.dropbox.com/s/mfijqowbhe3uy3y/MockModel.tar.gz?dl=0"
MOCK_DIR_NAME = "MockModel"


@pytest.mark.unit
def test_dropbox_model_download():

    global MOCK_URL, MOCK_DIR_NAME

    from thirdai import embeddings

    with pytest.raises(Exception, match=r".*not a valid JSON file.*") as e_info:
        embeddings.DocSearchModel(
            local_path=None, download_metadata=(MOCK_URL, MOCK_DIR_NAME)
        )


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

    assert cached_before is False
    assert cached_after is True
