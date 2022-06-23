from sklearn.datasets import load_svmlight_file
import numpy as np
from thirdai import match


def list_of_lists_to_csr(lists, use_softmax):
    offsets = np.zeros(shape=(len(lists) + 1,), dtype="uint32")
    for i in range(1, len(offsets)):
        offsets[i] = offsets[i - 1] + len(lists[i - 1])
    values = np.ones(shape=(offsets[-1],)).astype("float32")
    if use_softmax:
        for i in range(1, len(offsets)):
            start = offsets[i - 1]
            end = offsets[i]
            length = end - start
            for j in range(start, end):
                values[j] /= length
    indices = np.concatenate(lists).astype("uint32")
    return (indices, values, offsets)


def get_data(path, use_softmax, limit):
    data = load_svmlight_file(path, multilabel=True)
    data_x = (
        data[0].indices.astype("uint32"),
        data[0].data.astype("float32"),
        data[0].indptr.astype("uint32")[:limit],
    )
    data_y = list_of_lists_to_csr(data[1], use_softmax)
    return data_x, data_y, data[1]


use_softmax = True

train_x, train_y, train_gt = get_data(
    "/Users/josh/amazon-670k/train_shuffled_noHeader.txt", use_softmax
)
test_x, test_y, test_gt = get_data(
    "/Users/josh/amazon-670k/test_shuffled_noHeader_sampled.txt", use_softmax, limit=1000
)

test_match = match.Match(
    max_label=670091,
    num_classifiers=4,
    input_dim=135909,
    hidden_layer_dim=512,
    hidden_layer_sparsity=1,
    last_layer_dim=20000,
    last_layer_sparsity=0.05,
    use_softmax=use_softmax,
)

test_match.index(
    train_x=train_x,
    train_y=train_y,
    test_x=test_x,
    test_y=test_y,
    num_iterations=1,
    num_epochs_per_iteration=3,
)

query_results = test_match.query(test_x)
recalled = [q in gt for q, gt in zip(query_results, test_gt)]
print(query_results)
print(test_gt)
print(f"R1@1 = {sum(recalled) / len(recalled)}")
