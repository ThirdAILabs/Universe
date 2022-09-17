import pytest
from thirdai import bolt, dataset
import numpy as np
from sklearn.metrics import roc_auc_score

pytestmark = [pytest.mark.unit]

def generate_dataset(n_classes, n_samples, batch_size):
    possible_one_hot_encodings = np.eye(n_classes)

    lhs_tokens = np.random.choice(n_classes, size=n_samples).astype("uint32")
    rhs_tokens = np.random.choice(n_classes, size=n_samples).astype("uint32")
    labels_np = np.random.choice(2, size=n_samples)

    # Make the tokens the same where the label is 1
    rhs_tokens = np.where(labels_np, lhs_tokens, rhs_tokens)

    lhs_inputs = possible_one_hot_encodings[lhs_tokens]
    rhs_inputs = possible_one_hot_encodings[rhs_tokens]

    lhs_inputs += np.random.normal(0, 0.1, lhs_inputs.shape)
    rhs_inputs += np.random.normal(0, 0.1, rhs_inputs.shape)

    lhs_dataset = dataset.from_numpy(lhs_inputs.astype("float32"), batch_size)
    rhs_dataset = dataset.from_numpy(rhs_inputs.astype("float32"), batch_size)

    labels_dataset = dataset.from_numpy(labels_np.astype("float32"), batch_size)

    return lhs_dataset, rhs_dataset, labels_dataset, labels_np


def create_model(input_dim):
    lhs_input = bolt.graph.Input(input_dim)
    rhs_input = bolt.graph.Input(input_dim)

    lhs_hidden = bolt.graph.FullyConnected(dim=100, activation="relu")(lhs_input)
    rhs_hidden = bolt.graph.FullyConnected(dim=100, activation="relu")(rhs_input)

    dot = bolt.graph.DotProduct()(lhs_hidden, rhs_hidden)

    model = bolt.graph.Model(inputs=[lhs_input, rhs_input], output=dot)

    model.compile(bolt.BinaryCrossEntropyLoss())

    return model

def test_dot_product():
    n_classes = 100
    n_samples = 2000
    batch_size = 100
    
    train_rhs_data, train_lhs_data, train_labels, _ = generate_dataset(n_classes, n_samples, batch_size)
    test_rhs_data, test_lhs_data, test_labels, test_labels_np = generate_dataset(n_classes, n_samples, batch_size)

    model = create_model(n_classes)

    train_cfg = bolt.graph.TrainConfig.make(learning_rate=0.01, epochs=1)
    predict_cfg = bolt.graph.PredictConfig.make().return_activations()

    for _ in range(20):
      model.train([train_lhs_data, train_rhs_data], train_labels, train_cfg)
      _, activations = model.predict([test_lhs_data, test_rhs_data], test_labels, predict_cfg)

      print(roc_auc_score(test_labels_np, activations[:,0]))