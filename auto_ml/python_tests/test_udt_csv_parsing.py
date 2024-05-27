import os

import pytest
from thirdai import bolt


@pytest.fixture
def write_data():
    filename = "csv_parsing_temp.csv"
    n_lines = 10

    with open(filename, "w") as out:
        out.write("text,class\n")
        for i in range(n_lines):
            out.write(f'"This is a paragraph.\nAnd another one.",{i}')

    yield filename, n_lines

    os.remove(filename)


@pytest.mark.unit
def test_udt_classifier_csv_parsing(write_data):
    class CountCallback(bolt.train.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.batch_end_count = 0

        def on_batch_end(self):
            self.batch_end_count += 1

        def num_batches(self):
            return self.batch_end_count

    filename, n_lines = write_data

    model = bolt.UniversalDeepTransformer(
        data_types={
            "text": bolt.types.text(),
            "class": bolt.types.categorical(),
        },
        target="class",
        n_target_classes=n_lines,
    )

    batch_counter = CountCallback()

    model.train(filename, epochs=1, batch_size=1, callbacks=[batch_counter])

    assert batch_counter.num_batches() == n_lines
