import numpy as np
from thirdai import bolt

model = bolt.UniversalDeepTransformer(
    data_types={
        "QUERY": bolt.types.text(),
        "DOC_ID": bolt.types.categorical(delimiter=":"),
    },
    target="DOC_ID",
    n_target_classes=10000,
    integer_target=True,
    options={
        "extreme_output_dim": 1000,
        "fhr": 100,
        "extreme_num_hashes": 4,
        "embedding_dimension": 2048,
        "extreme_classification": True,
    },
)
index = model.get_index()


def make_input_dataset(tensors: list):
    tensorlist = tensors
    dataset = [tensorlist]
    return dataset


def make_output_dataset(
    labels: list, output_dim: int, mach_index, label_dim=4294967295
):
    def make_sparse_tensor(labels, output_dim):
        return bolt.nn.Tensor(
            indices=labels, values=[[1] * len(x) for x in labels], dense_dim=output_dim
        )

    output_tensor = make_sparse_tensor(labels, output_dim)

    def transform_label_to_mach_label(mach_index, labels):
        mach_labels = []
        for label in labels:
            new_labels = []
            for l in label:
                new_labels += mach_index.get_entity_hashes(l)
            mach_labels.append(new_labels)
        return mach_labels

    mach_labels = transform_label_to_mach_label(mach_index, labels)

    mach_label_tensor = make_sparse_tensor(labels=mach_labels, output_dim=label_dim)

    return make_input_dataset([output_tensor, mach_label_tensor])


input1 = np.arange(1000).reshape(10, 100)
input2 = np.arange(2000).reshape(10, 200)
tensor1 = bolt.nn.Tensor(values=input1, with_grad=False)
tensor2 = bolt.nn.Tensor(values=input2, with_grad=False)

labels = [[i] for i in range(tensor1.activations.shape[0])]

input_dataset = make_input_dataset([tensor1])
output_dataset = make_output_dataset(labels=labels, output_dim=1000, mach_index=index)


print(output_dataset[0][0].active_neurons[0])
print(output_dataset[0][1].active_neurons[0])
print(input_dataset[0][0].activations[0])


model.train_on_tensors(input=input_dataset, output=output_dataset)
