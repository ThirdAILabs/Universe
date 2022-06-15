from thirdai import bolt

input1 = bolt.Input(dim=100)
input2 = bolt.Input(dim=20)

# __call__ operator calls the addPredessors() method on the node.
fc1 = bolt.FullyConnected(dim=40, sparsity=0.2, activation="relu")(input1)
fc2 = bolt.FullyConnected(dim=10, sparsity=1.0, activation="relu")(input2)

concat = bolt.Concatenate()([fc1, fc2])

output = bolt.FullyConnected(dim=4, activation="softmax")(concat)

model = bolt.Model(inputs=[input1, input2], output=output)

model.compile(loss=bolt.CategoricalCrossEntropy())

# model.train(
#   data, labels, learning_rate, epochs, etc.
# )

# model.predict(
#   data, labels, metrics, etc.
# )
