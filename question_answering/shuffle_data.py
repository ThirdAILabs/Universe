import numpy as np
from tqdm import tqdm

positives = np.load("tokenized_positives.npy")
negatives = np.load("tokenized_negatives.npy")
labels = np.random.randint(low=0, high=2, size=len(positives))

for i, label in tqdm(list(enumerate(labels))):
    if label:
        temp = np.array(positives[i])
        positives[i][:] = negatives[i]
        negatives[i][:] = temp

np.save("tokenized_passages_1.npy", positives)
np.save("tokenized_passages_2.npy", negatives)
np.save("labels.npy", labels)
