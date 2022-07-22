from thirdai import bolt, dataset
import numpy as np
from sklearn.metrics import roc_auc_score

def get_labels(dataset: str):
    labels = []
    with open(dataset) as file:
        for line in file.readlines():
            items = line.strip().split()
            label = int(items[0])
            labels.append(label)
    return np.array(labels)


input_layer = bolt.graph.Input(dim=15)
hidden_layer = bolt.graph.FullyConnected(
    dim=1000, sparsity=0.2, activation="relu")(input_layer)

token_input = bolt.graph.TokenInput()
embedding_layer = bolt.graph.Embedding(
    num_embedding_lookups=8, lookup_size=16, log_embedding_block_size=10)(token_input)

concat = bolt.graph.Concatenate()([hidden_layer, embedding_layer])

hidden_layer2 = bolt.graph.FullyConnected(
    dim=1000, sparsity=0.2, activation="relu")(concat)
output_layer = bolt.graph.FullyConnected(
    dim=2, activation="softmax")(hidden_layer2)


model = bolt.graph.Model(inputs=[input_layer], token_inputs=[
                         token_input], output=output_layer)
model.compile(bolt.CategoricalCrossEntropyLoss())

train_data, train_tokens, train_labels = dataset.load_click_through_dataset(
    "/Users/nmeisburger/ThirdAI/data/criteo-small/train_shuf.txt",
    batch_size=256,
    num_numerical_features=15,
    num_categorical_features=24
)

test_data, test_tokens, test_labels = dataset.load_click_through_dataset(
    "/Users/nmeisburger/ThirdAI/data/criteo-small/test_shuf.txt",
    batch_size=256,
    num_numerical_features=15,
    num_categorical_features=24
)

train_cfg = bolt.graph.TrainConfig.make(learning_rate=0.0001, epochs=1)

predict_cfg = bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).return_activations()

true_labels = get_labels("/Users/nmeisburger/ThirdAI/data/criteo-small/test_shuf.txt")

for e in range(4):
    model.train([train_data], [train_tokens], train_labels, train_cfg)
    _, outputs = model.predict([test_data], [test_tokens], test_labels, predict_cfg)
    print("ROC AUC = ", roc_auc_score(true_labels, outputs[:,1]))