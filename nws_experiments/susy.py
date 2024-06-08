import time
from thirdai.bolt import NWS, SRP
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

df = pd.read_csv(
    "/Users/benitogeordie/Grad School Prep/NWS Paper/experiments/physics/supersymmetry_dataset.csv"
)
df = df.sample(frac=1)

input_columns = list(df.columns)
input_columns.remove("SUSY")
inputs = df[input_columns].to_numpy()

outputs = [[out] for out in list(df["SUSY"])]

eighty_percent = len(df) * 8 // 10
train_size = eighty_percent

train_inputs = inputs[:train_size]
train_outputs = outputs[:train_size]
test_inputs = inputs[train_size:]
test_outputs = outputs[train_size:]

train_input_mean = np.mean(train_inputs, axis=0)
train_inputs -= train_input_mean
test_inputs -= train_input_mean
train_inputs = normalize(train_inputs, axis=1, norm="l2").tolist()
test_inputs = normalize(test_inputs, axis=1, norm="l2").tolist()

with open("susy.out", "a", buffering=1) as w:

    def log(string):
        w.write(string + "\n")
        print(string, flush=True)

    for rows in [100]:
        for hashes_per_row in [19]:
            log(f"{rows=} {hashes_per_row=}")

            start = time.time()
            srp = SRP(
                input_dim=len(input_columns),
                rows=rows,
                hashes_per_row=hashes_per_row,
            )
            nws = NWS(hash=srp, val_dim=1)
            end = time.time()
            log(f"Construction time (s): {end - start}")

            start = time.time()
            nws.train_parallel(
                inputs=train_inputs,
                outputs=train_outputs,
                threads=10,
            )
            end = time.time()
            log(f"Training time (s): {end - start}")

            start = time.time()
            predictions = nws.predict(inputs=test_inputs)
            end = time.time()
            log(f"Prediction time (s): {end - start}")

            correct = sum(
                [
                    round(predicted) == int(expected)
                    for [predicted], [expected] in zip(predictions, test_outputs)
                ]
            )

            log(f"Accuracy: {correct / len(test_outputs)}\n")
