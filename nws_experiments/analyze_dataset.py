import pandas as pd
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("/Users/benitogeordie/Grad School Prep/NWS Paper/experiments/physics/supersymmetry_dataset_shuffled.csv", nrows=5000)
input_columns = list(df.columns)
input_columns.remove("SUSY")
inputs = df[input_columns].to_numpy()
# inputs -= inputs.min(axis=0)
# inputs /= inputs.max(axis=0)

distances = []

for i in tqdm(range(len(inputs))):
    for j in range(i + 1, len(inputs)):
        distances.append(np.linalg.norm(inputs[i] - inputs[j]))


def hist(stuff):
    plt.hist(stuff)
    plt.show()
    plt.clf()


sorted_distances = sorted(distances)
hist(sorted_distances)


min_distances = []

for i in tqdm(range(len(inputs))):
    min_dist = float("inf")
    for j in range(len(inputs)):
        if i == j:
            continue
        min_dist = min(min_dist, np.linalg.norm(inputs[i] - inputs[j]))
    min_distances.append(min_dist)

hist(min_distances)