import torchvision
import torch
from thirdai import smx
import os
import numpy as np


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


def apply_model(x, hidden, output):
    x = smx.from_numpy(x)
    x = smx.reshape(x, smx.Shape(x.shape[0], 784))
    x = smx.Variable(x)

    out = hidden(x)
    out = smx.relu(out)
    out = output(out)
    return out


def test_smx_mlp_mnist():
    download_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../build/mnist_dataset"
    )
    train_loader, test_loader = load_datasets(download_dir=download_dir)

    hidden = smx.Linear(256, 784)
    output = smx.Linear(10, 256)

    optimizer = smx.optimizers.Adam(hidden.parameters() + output.parameters(), lr=0.001)

    for epoch in range(5):
        for x, y in train_loader:
            out = apply_model(x.numpy(), hidden, output)

            loss = smx.cross_entropy(out, smx.from_numpy(y.numpy().astype(np.uint32)))
            loss.backward()
            optimizer.apply()
            optimizer.zero_grad()

        correct, total = 0, 0
        for x, y in test_loader:
            out = apply_model(x.numpy(), hidden, output).tensor.numpy()

            correct += np.sum(np.argmax(out, axis=1) == y.numpy())
            total += len(x)

        accuracy = correct / total
        print(f"Epoch {epoch} accuracy={accuracy}")

    assert accuracy >= 0.95
