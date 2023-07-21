import torch
import onnx
from thirdai import bolt
import pickle
import numpy as np


def string_to_tensor(input_string):
    """
    Convert a string to a PyTorch tensor using one-hot encoding.

    Args:
        input_string (str): The input string to be converted to a tensor.

    Returns:
        torch.Tensor: The one-hot encoded tensor representing the input string.
    """
    # Define the character set to use (ASCII characters from 0 to 127)
    charset = [chr(i) for i in range(128)]

    # Get the length of the dictionary (number of unique characters)
    num_chars = len(charset)

    # Initialize an empty tensor with appropriate size
    tensor = torch.zeros(len(input_string), num_chars)

    # Create a dictionary mapping each character to its index
    char_to_index = {char: index for index, char in enumerate(charset)}

    # Convert each character in the input string to its corresponding index and set the corresponding position to 1
    for i, char in enumerate(input_string):
        index = char_to_index[char]
        tensor[i][index] = 1

    return tensor


def tensor_to_string(tensor):
    """
    Convert a one-hot encoded PyTorch tensor back to the original string.

    Args:
        tensor (torch.Tensor): The one-hot encoded tensor.

    Returns:
        str: The original string represented by the tensor.
    """
    # Get the indices of the maximum values along the second dimension (axis=1)
    max_indices = torch.argmax(tensor, dim=1)

    # Create a dictionary mapping indices to their corresponding characters
    charset = [chr(i) for i in range(128)]
    index_to_char = {index: char for index, char in enumerate(charset)}

    # Convert indices to characters using the provided mapping
    chars = [index_to_char[index.item()] for index in max_indices]

    # Join the characters to form the original string
    return "".join(chars)


class BoltOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, model):
        # out = model.predict({"query": "hello"})
        out = x.numpy()
        return torch.tensor(out, requires_grad=True)

    @staticmethod
    def backward(ctx, dy):
        return None, None


class BoltModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = bolt.UniversalDeepTransformer(
            data_types={
                "query": bolt.types.text(),
                "label": bolt.types.categorical(),
            },
            target="label",
            integer_target=True,
            n_target_classes=4,
            options={"fhr": 10000},
        )

    def forward(self, x):
        out = BoltOp.apply(x, self.model)
        return out


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.Tensor(x.numpy())


def export_custom_op():
    # dummy_input = string_to_tensor("mo salah is the best player in the world")
    dummy_input = torch.randn(1, 100)
    f = "./model.onnx"
    torch_model = SimpleModel()
    torch_model.eval()

    torch.onnx.export(
        torch_model,
        dummy_input,
        f,
        input_names=["input"],
        output_names=["output"],
    )


export_custom_op()

# Load the ONNX model
model = onnx.load("model.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

import onnxruntime as ort

ort_session = ort.InferenceSession("model.onnx")
print(ort_session.get_inputs())
# output1 = ort_session.run(
#     None,
#     {"input": torch.randn(1, 100).numpy()},
# )
# output2 = ort_session.run(
#     None,
#     {"input": torch.randn(1, 100).numpy()},
# )
outputs = ort_session.run(
    None,
    # {}
    {"input": np.random.rand(1, 100).astype(np.float32)},
)
print(outputs)
