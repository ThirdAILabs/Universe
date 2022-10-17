import os
import pytest
from thirdai import bolt, deployment


pytestmark = [pytest.mark.integration, pytest.mark.release]

QUERIES_FILES = "./correct_queries.csv"
CONFIG_FILE = "./flash_index_config"


if __name__ == "__main__":
    pass
