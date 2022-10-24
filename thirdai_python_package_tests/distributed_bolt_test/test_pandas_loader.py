import os

import pytest

pytestmark = [pytest.mark.distributed]

def test_pandas_loader():
    from thirdai.distributed_bolt import PandasColumnMapGenerator

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

    ordered_chunks = []
    while any(next_loads := [loader.next() for loader in loaders]):
        ordered_chunks += [
            next_load for next_load in next_loads if next_load is not None
        ]

    assert sum([chunk.num_rows() for chunk in ordered_chunks]) == num_rows

    current_vec_id = 0
    for chunk in ordered_chunks:
        col1, col2 = chunk["col1"], chunk["col2"]
        for vec_id_in_chunk in range(0, chunk.num_rows()):
            assert col1[vec_id_in_chunk] == current_vec_id
            assert col2[vec_id_in_chunk] == current_vec_id
            current_vec_id += 1

    os.remove(test_file)
