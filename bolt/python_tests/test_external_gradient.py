import pytest
import torch
from thirdai import bolt

N_CLASSES = 50
HIDDEN_DIM = 100
BATCH_SIZE = 100


class WrappedBoltModel:
    def __init__(self):
        input_ = bolt.nn.Input(dim=N_CLASSES)
        out = bolt.nn.FullyConnected(
            dim=HIDDEN_DIM, input_dim=input_.dim(), activation="relu"
        )(input_)
        loss = bolt.nn.losses.ExternalLoss(
            output=out, external_gradients=bolt.nn.Input(dim=out.dim())
        )

        self.model = bolt.nn.Model(inputs=[input_], outputs=[out], losses=[loss])

    def forward(self, x):
        x = bolt.nn.Tensor(x.numpy())
        out = self.model.forward([x], use_sparsity=True)[0]
        out = torch.from_numpy(out.activations)
        out.requires_grad = True
        return out

    def backpropagate(self, grad):
        # Bolt expects that the gradient is to minimize the loss
        grad = bolt.nn.Tensor(-grad.numpy())
        self.model.backpropagate([grad])

    def update_parameters(self, lr):
        self.model.update_parameters(lr)


def get_dataset(n_batches, batch_size):
    labels = torch.randint(0, N_CLASSES, size=(n_batches * batch_size,))
    inputs = torch.nn.functional.one_hot(labels, num_classes=N_CLASSES).type(
        torch.float32
    )
    inputs += torch.normal(torch.zeros_like(inputs), torch.full_like(inputs, 0.1))

    inputs = torch.split(inputs, batch_size, dim=0)
    labels = torch.split(labels, batch_size, dim=0)

    return inputs, labels


@pytest.mark.unit
def test_bolt_with_torch_output():
    bolt_model = WrappedBoltModel()

    output = torch.nn.Linear(HIDDEN_DIM, N_CLASSES)
    opt = torch.optim.Adam(output.parameters(), lr=0.001)

    train_x, train_y = get_dataset(n_batches=20, batch_size=BATCH_SIZE)
    (test_x,), (test_y,) = get_dataset(n_batches=1, batch_size=200)

    for _ in range(10):
        output.train()

        for x, y in zip(train_x, train_y):
            opt.zero_grad()
            hidden = bolt_model.forward(x)
            out = output(hidden)

            loss = torch.nn.functional.cross_entropy(out, y)
            loss.backward()
            bolt_model.backpropagate(hidden.grad)

            opt.step()
            bolt_model.update_parameters(0.001)

        output.eval()

        out = output(bolt_model.forward(test_x))
        preds = torch.argmax(out, dim=1)

        acc = preds.eq(test_y).sum().item() / len(preds)

    assert acc >= 0.95  # Accuracy should be ~0.99-1.0


@pytest.mark.unit
def test_invalid_grad_dim():
    model = WrappedBoltModel()

    model.forward(torch.rand(10, N_CLASSES))

    with pytest.raises(ValueError):
        model.backpropagate(torch.rand(10, N_CLASSES + 1))


@pytest.mark.unit
def test_invalid_grad_batch_size():
    model = WrappedBoltModel()

    model.forward(torch.rand(10, N_CLASSES))

    with pytest.raises(ValueError):
        model.backpropagate(torch.rand(11, N_CLASSES))
