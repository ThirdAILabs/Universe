from thirdai import smx, hashing
import numpy as np
import random
import time
import tqdm


def load_dataset(path, batch_size, input_dim, n_classes, training):
    lines = open(path).readlines()
    if training:
        random.shuffle(lines)

    batches = []
    for i in range(0, len(lines), batch_size):
        labels = []

        label_offsets, label_indices, label_values = [0], [], []
        offsets, indices, values = [0], [], []
        for line in lines[i : i + batch_size]:
            items = line.split(" ")
            sample_labels = list(map(int, items[0].split(",")))
            labels.append(sample_labels)
            label_indices.extend(sample_labels)
            label_values.extend([1.0] * len(sample_labels))
            label_offsets.append(len(label_indices))

            for kv in items[1:]:
                key, value = kv.split(":")
                indices.append(int(key))
                values.append(float(value))
            offsets.append(len(indices))

        inputs = smx.CsrTensor(
            row_offsets=offsets,
            col_indices=indices,
            col_values=values,
            dense_shape=smx.Shape(len(offsets) - 1, input_dim),
        )
        inputs = smx.Variable(inputs, requires_grad=False)
        if training:
            labels = smx.CsrTensor(
                row_offsets=label_offsets,
                col_indices=label_indices,
                col_values=label_values,
                dense_shape=smx.Shape(len(label_offsets) - 1, n_classes),
            )
            labels = smx.Variable(labels, requires_grad=False)

        batches.append((inputs, labels))

    return batches


class SparseModel(smx.Module):
    train_sparsity = 0.005

    def __init__(self, input_dim, n_classes):
        super().__init__()

        self.hidden = smx.Linear(dim=256, input_dim=input_dim)
        self.output = smx.SparseLinear(
            dim=n_classes, input_dim=256, sparsity=self.train_sparsity
        )

    def forward(self, x, y=None):
        out = smx.relu(self.hidden(x))
        out = self.output(out, y)
        return out

    def training(self):
        self.output.sparsity = self.train_sparsity

    def eval(self):
        self.output.sparsity = 1.0


input_dim = 135909
n_classes = 670091
model = SparseModel(input_dim=input_dim, n_classes=n_classes)

optimizer = smx.optimizers.Adam(model.parameters(), lr=1e-4)
optimizer.register_on_update_callback(model.output.on_update_callback())

train_batches = load_dataset(
    path="/share/data/amazon-670k/train_shuffled_noHeader.txt",
    batch_size=1024,
    input_dim=input_dim,
    n_classes=n_classes,
    training=True,
)
test_batches = load_dataset(
    path="/share/data/amazon-670k/test_shuffled_noHeader_sampled.txt",
    batch_size=2048,
    input_dim=input_dim,
    n_classes=n_classes,
    training=False,
)

for epoch in range(5):
    model.training()
    ts = time.perf_counter()

    total_forward_backward, total_update = 0, 0
    for x, y in tqdm.tqdm(train_batches):
        optimizer.zero_grad()

        fs = time.perf_counter()
        out = model(x, y)
        loss = smx.cross_entropy(out, y.tensor)
        loss.backward()
        fe = time.perf_counter()
        total_forward_backward += fe - fs

        us = time.perf_counter()
        optimizer.step()
        ue = time.perf_counter()
        total_update += ue - us

    te = time.perf_counter()

    model.eval()
    correct, total = 0, 0

    es = time.perf_counter()
    for x, y in test_batches:
        out = model(x).tensor.numpy()

        for pred, labels in zip(np.argmax(out, axis=1), y):
            if pred in labels:
                correct += 1
        total += len(y)
    ee = time.perf_counter()

    print(
        f"epoch {epoch} train_time={te-ts:.3f}s accuracy={correct / total} eval_time={ee - es}s"
    )

    print(
        f"total_forward_backward={total_forward_backward} total_update={total_update}"
    )
