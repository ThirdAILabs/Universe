from typing import Tuple

import numpy as np
from thirdai import bolt_v2 as bolt


def create_dataset(
    shape: Tuple[int], n_batches: int, noise_std=0.1, return_np_labels=False
):
    n_classes = shape[-1]

    possible_one_hot_encodings = np.eye(n_classes)

    labels = np.random.choice(n_classes, size=(n_batches, *(shape[:-1]))).astype(
        "uint32"
    )

    examples = possible_one_hot_encodings[labels]

    reshaped_labels = labels.reshape((*labels.shape, 1))

    noise = np.random.normal(0, noise_std, examples.shape)

    examples = (examples + noise).astype("float32")

    data_batches = []
    label_batches = []
    for i in range(len(examples)):
        data = bolt.nn.Tensor(examples[i])
        label = bolt.nn.Tensor(
            reshaped_labels[i],
            np.ones_like(reshaped_labels[i], dtype=np.float32),
            dense_dim=n_classes,
        )

        data_batches.append([data])
        label_batches.append([label])

    if return_np_labels:
        return data_batches, label_batches, labels

    return data_batches, label_batches
