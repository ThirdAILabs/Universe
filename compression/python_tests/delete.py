import numpy as np
from thirdai import bolt, dataset
import sys

INPUT_DIM = 10
HIDDEN_DIM = 10
OUTPUT_DIM = 10
LEARNING_RATE = 0.002
ACCURACY_THRESHOLD = 0.8


def build_simple_hidden_layer_model(
    input_dim=10,
    hidden_dim=50,
    output_dim=10,
):
    input_layer = bolt.graph.Input(dim=input_dim)

    hidden_layer = bolt.graph.FullyConnected(
        dim=hidden_dim,
        activation="relu",
    )(input_layer)

    output_layer = bolt.graph.FullyConnected(dim=output_dim, activation="softmax")(
        hidden_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    return model


model = build_simple_hidden_layer_model()
model.compile(loss=bolt.CategoricalCrossEntropyLoss())
layer1 = model.get_layer("fc_1")
par_ref = layer1.weights


wc0 = par_ref.compress(
    compression_scheme="count_sketch",
    compression_density=float(sys.argv[1]),
    seed_for_hashing=1,
    sample_population_size=1,
)
# print(wc0)

# print(par_ref.get())
x1 = par_ref.get()
par_ref.set(wc0)
# print(par_ref.get())
x2 = par_ref.get()
# print(np.linalg.norm(x2 - x1))
# print(np.linalg.norm(x1))
# print(np.linalg.norm(x2))
print(f"relative loss: {np.linalg.norm(x2 - x1)/np.linalg.norm(x1)}")
