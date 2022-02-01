from thirdai import bolt
import tensorflow.keras as keras

layers_ = [
    bolt.LayerConfig(dim=256, activation_function="ReLU"),
    bolt.LayerConfig(dim=10, load_factor=0.4, activation_function="Softmax"),
]

network = bolt.Network(layers=layers_, input_dim=784)

network.train(
    batch_size=250,
    train_data="/home/ubuntu/mnist",
    test_data="/home/ubuntu/mnist.t",
    learning_rate=0.0001,
    epochs=1,
)


inp_dim = network.get_input_dim()
num_layers = network.get_num_layers()
layer_sizes = network.get_layer_sizes()
act_funcs = network.get_activation_functions()

############  To Keras #############

keras_layers = [keras.layers.InputLayer((inp_dim,))]
keras_layers += [
    keras.layers.Dense(
        layer_sizes[i], activation=act_funcs[i].lower(), name="layer" + str(i)
    )
    for i in range(num_layers)
]

model = keras.models.Sequential(keras_layers)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

model.save("./bolt_to_keras_model")

model2 = keras.models.load_model("./bolt_to_keras_model")
