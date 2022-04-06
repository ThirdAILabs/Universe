import torch
import torch.nn as nn
from mlflow_logger import ExperimentLogger
from torchvision import datasets, transforms

torch.manual_seed(101)

# TODO(vihan) Move this parameters to a config and track them
input_size = 784  # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


def compute_validation_accuracy(model, test_loader):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28)
            labels = labels
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    return 100.0 * n_correct / n_samples


def train(train_loader, test_loader, mlflow_logger):
    model = NeuralNet(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mlflow_logger.log_start_training()
    for _ in range(num_epochs):
        for images, labels in train_loader:
            images = images.reshape(-1, 28 * 28)
            labels = labels
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = compute_validation_accuracy(model, test_loader)
        mlflow_logger.log_epoch(acc)

    acc = compute_validation_accuracy(model, test_loader)
    print(f"Accuracy of the network on the 10000 test images: {acc} %")

    mlflow_logger.log_final_accuracy(acc)


def main():

    train_dataset = datasets.MNIST(
        root="/data/mnist/pytorch_data",
        train=True,
        transform=transforms.ToTensor(),
        download=False,
    )

    test_dataset = datasets.MNIST(
        root="/data/mnist/pytorch_data",
        train=False,
        transform=transforms.ToTensor(),
        download=False,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )

    with ExperimentLogger(
        experiment_name="MNIST Benchmark",
        dataset="mnist",
        algorithm="feedforward",
        framework="PyTorch",

    ) as mlflow_logger:
        train(train_loader, test_loader, mlflow_logger)


if __name__ == "__main__":
    main()
