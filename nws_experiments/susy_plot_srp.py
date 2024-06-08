from thirdai.bolt import SRPKernel, SRP, Theta
from exp_utils import run
import pandas as pd
import numpy as np

power = 1
kernel = SRPKernel(power=power)
distance = Theta()


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
        [r * r for r in range(1, 50, 1)]
        # + list(range(100, 1000, 100))
        # + list(range(1000, 10_000, 1000))
        # + list(range(10_000, 100_000, 10000))
    )
]

TRAIN_SIZE = 10_000
TEST_SIZE = 10_000

df = pd.read_csv(
    "/Users/benitogeordie/Grad School Prep/NWS Paper/experiments/physics/supersymmetry_dataset_shuffled.csv",
    nrows=TRAIN_SIZE + TEST_SIZE,
)

input_columns = list(df.columns)
input_columns.remove("SUSY")
inputs = df[input_columns].to_numpy()

inputs -= np.mean(inputs, axis=0)
inputs /= np.linalg.norm(inputs, axis=1).reshape((-1, 1))

outputs = df["SUSY"].to_numpy()

train_inputs = inputs[:TRAIN_SIZE]
train_outputs = outputs[:TRAIN_SIZE]

test_inputs = inputs[TRAIN_SIZE:]

run(
    hash_factories,
    kernel,
    distance,
    train_inputs,
    train_outputs,
    test_inputs,
    random_sampling=True,
)
