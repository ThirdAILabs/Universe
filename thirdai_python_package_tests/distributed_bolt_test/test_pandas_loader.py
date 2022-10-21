import os

import pytest

pytestmark = [pytest.mark.distributed]
from thirdai.distributed_bolt import PandasColumnMapGenerator


# TODO(Josh): Make test a bit more rigorous
def test_pandas_loader():
    test_file = "pandas_test_file.csv"
    num_rows = 10000
    with open(test_file, "w") as f:
        f.write("col1,col2\n")
        for row in range(num_rows):
            f.write(f"{row},{row}\n")

    num_nodes = 6
    loaders = [
        PandasColumnMapGenerator(
            path=test_file, num_nodes=num_nodes, node_index=i, lines_per_load=500
        )
        for i in range(num_nodes)
    ]

    total_num_rows = 0
    for loader in loaders:
        while not ((next_load := loader.next()) is None):
            total_num_rows += next_load.num_rows()
    assert total_num_rows == num_rows

    os.remove(test_file)
