from thirdai.bolt import SRPKernel, SRP
from exp_utils import run
import pandas as pd

power = 1
kernel = SRPKernel(power=power)

def make_hash_factory(rows):
    def factory(input_dim):
        return SRP(
            input_dim=input_dim,
            hashes_per_row=power,
            rows=rows,
            seed=314,
        )
    return factory

hash_factories = [
    make_hash_factory(rows)
    for rows in (
        list(range(10, 100, 10)) +
        list(range(100, 1000, 50)) +
        list(range(1000, 10_000, 500))
    )
]

TRAIN_SIZE = 10_000
TEST_SIZE = 10_000

df = pd.read_csv("/Users/benitogeordie/Grad School Prep/NWS Paper/experiments/physics/supersymmetry_dataset_shuffled.csv", nrows=TRAIN_SIZE + TEST_SIZE)

input_columns = list(df.columns)
input_columns.remove("SUSY")
inputs = df[input_columns].to_numpy()

outputs = df["SUSY"].to_numpy()

train_inputs = inputs[:TRAIN_SIZE]
train_outputs = outputs[:TRAIN_SIZE]

test_inputs = inputs[TRAIN_SIZE:]

run(hash_factories, kernel, train_inputs, train_outputs, test_inputs)