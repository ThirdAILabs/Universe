import os

import numpy as np
import torch
import torchvision
from thirdai import smx


def load_datasets(download_dir):
    transform = torchvision.transforms.ToTensor()

    train_set = torchvision.datasets.mnist.MNIST(
        root=download_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.mnist.MNIST(
        root=download_dir, train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=128)

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

            loss = smx.cross_entropy(out, smx.from_numpy(y.numpy().astype(np.uint32)))
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
