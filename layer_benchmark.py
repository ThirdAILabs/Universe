import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from thirdai import bolt


def make_sparse_tensor(X, sparsity):
    nonzeros = int(X.shape[1] * sparsity)
    values, indices = torch.sort(X, descending=True)
    return bolt.nn.Tensor(
        indices=indices[:, :nonzeros].numpy(),
        values=values[:, :nonzeros].numpy(),
        dense_dim=X.shape[1],
    )


def train_bolt(X, Y, n_classes, input_sparsity, output_sparsity):
    input_dim = X[0].shape[1]

    input_ = bolt.nn.Input(dim=input_dim)
    output = bolt.nn.FullyConnected(
        dim=n_classes,
        input_dim=input_dim,
        sparsity=output_sparsity,
        activation="softmax",
    )(input_)
    loss = bolt.nn.losses.CategoricalCrossEntropy(
        output, labels=bolt.nn.Input(n_classes)
    )
    model = bolt.nn.Model(inputs=[input_], outputs=[output], losses=[loss])

    if input_sparsity < 1.0:
        X = [[make_sparse_tensor(x, sparsity=input_sparsity)] for x in X]
    else:
        X = [[bolt.nn.Tensor(x.numpy())] for x in X]
    Y = [y.unsqueeze(-1) for y in Y]
    Y = [[bolt.nn.Tensor(y.numpy(), torch.ones_like(y).numpy(), n_classes)] for y in Y]

    train_times = []
    update_times = []
    total_times = []
    for x, y in zip(X, Y):
        ts = time.perf_counter()
        model.train_on_batch(x, y)
        te = time.perf_counter()

        us = time.perf_counter()
        model.update_parameters(0.001)
        ue = time.perf_counter()

        train_times.append(te - ts)
        update_times.append(ue - us)
        total_times.append(ue - ts)

    return {
        "name": "bolt",
        "input_sparsity": input_sparsity,
        "output_sparsity": output_sparsity,
        "train_time": np.mean(train_times),
        "update_time": np.mean(update_times),
        "total_time": np.mean(total_times),
    }


def train_torch(X, Y, n_classes):
    input_dim = X[0].shape[1]
    layer = torch.nn.Linear(in_features=input_dim, out_features=n_classes)

    opt = torch.optim.Adam(layer.parameters(), lr=0.001)

    train_times = []
    update_times = []
    total_times = []
    for x, y in zip(X, Y):
        ts = time.perf_counter()
        loss = F.cross_entropy(layer(x), target=y)
        loss.backward()
        te = time.perf_counter()

        us = time.perf_counter()
        opt.step()
        opt.zero_grad()
        ue = time.perf_counter()

        train_times.append(te - ts)
        update_times.append(ue - us)
        total_times.append(ue - ts)

    return {
        "name": "torch",
        "input_sparsity": 1.0,
        "output_sparsity": 1.0,
        "train_time": np.mean(train_times),
        "update_time": np.mean(update_times),
        "total_time": np.mean(total_times),
    }


def run_experiment(input_dim, n_classes, batch_size, n_batches, show=True):
    X = torch.rand(size=(n_batches, batch_size, input_dim))
    Y = torch.randint(low=0, high=n_classes, size=(n_batches, batch_size))

    results = [train_torch(X, Y, n_classes)]
    print(results[-1])
    for input_sparsity in [0.01, 0.05, 0.1, 0.2, 1.0]:
        for output_sparsity in [0.01, 0.05, 0.1, 0.2]:
            results.append(
                train_bolt(
                    X,
                    Y,
                    n_classes,
                    input_sparsity=input_sparsity,
                    output_sparsity=output_sparsity,
                )
            )
            print(results[-1])

    df = pd.DataFrame.from_records(results)
    df.sort_values(by="total_time", axis=0, inplace=True)

    if show:
        print(f"Input dim = {input_dim}, Output dim = {n_classes}")
        print(df.to_markdown(index=False))
        print()
    return df


run_experiment(input_dim=1_000, n_classes=50_000, batch_size=10_000, n_batches=4)
run_experiment(input_dim=10_000, n_classes=50_000, batch_size=10_000, n_batches=4)
run_experiment(input_dim=20_000, n_classes=50_000, batch_size=10_000, n_batches=4)
