import os
import time

import numpy as np
import pytest
import torch
import torchvision
from thirdai import smx

pytestmark = [pytest.mark.unit]


def load_datasets(download_dir):
    transform = torchvision.transforms.ToTensor()

    train_set = torchvision.datasets.mnist.MNIST(
        root=download_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.mnist.MNIST(
        root=download_dir, train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=250)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=250)

    return train_loader, test_loader


def flatten(x):
    x = smx.from_numpy(x)
    x = smx.reshape(x, smx.Shape(x.shape[0], 784))
    return smx.Variable(x)


def test_smx_mlp_mnist():
    download_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../build/mnist_dataset"
    )
    train_loader, test_loader = load_datasets(download_dir=download_dir)

    model = smx.Sequential(
        [smx.Linear(256, 784), smx.Activation("relu"), smx.Linear(10, 256)]
    )

    optimizer = smx.optimizers.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        for x, y in train_loader:
            out = model(flatten(x))
            y = smx.Variable(
                smx.from_numpy(y.numpy().astype(np.uint32)), requires_grad=False
            )

            loss = smx.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        correct, total = 0, 0
        for x, y in test_loader:
            out = model(flatten(x)).tensor.numpy()

            correct += np.sum(np.argmax(out, axis=1) == y.numpy())
            total += len(x)

        accuracy = correct / total
        print(f"Epoch {epoch} accuracy={accuracy}")

    assert accuracy >= 0.95


class SparseModel(smx.Module):
    def __init__(self):
        super().__init__()

        self.hidden = smx.SparseLinear(dim=20000, input_dim=784, sparsity=0.01)
        self.output = smx.Linear(dim=10, input_dim=20000)

    def forward(self, x):
        out = self.hidden(x)
        out = smx.relu(out)
        out = self.output(out)
        return out


def test_smx_sparse_mlp_mnist():
    download_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../build/mnist_dataset"
    )
    train_loader, test_loader = load_datasets(download_dir=download_dir)

    model = SparseModel()

    optimizer = smx.optimizers.Adam(model.parameters(), lr=0.001)
    optimizer.register_on_update_callback(model.hidden.on_update_callback())

    for epoch in range(5):
        s = time.perf_counter()
        for x, y in train_loader:
            out = model(flatten(x))
            y = smx.Variable(
                smx.from_numpy(y.numpy().astype(np.uint32)), requires_grad=False
            )

            loss = smx.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        e = time.perf_counter()
        print(f"epoch {e-s}")

        correct, total = 0, 0
        for x, y in test_loader:
            out = model(flatten(x)).tensor.numpy()

            correct += np.sum(np.argmax(out, axis=1) == y.numpy())
            total += len(x)

        accuracy = correct / total
        print(f"Epoch {epoch} accuracy={accuracy}")
