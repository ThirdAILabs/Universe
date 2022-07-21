from thirdai import bolt, dataset
import sys

if len(sys.argv) != 2:
    print("Invalid args: Usage python3 <script>.py <train_data>")
    sys.exit(1)

TRAIN_DATA = sys.argv[1]
TEST_DATA = "/share/data/BERT/sentences_tokenized_shuffled_trimmed_test_100k.txt"
INPUT_DIM = 100000
BATCH_SIZE = 2048
VOCAB_SIZE = 30224

input_layer = bolt.graph.Input(dim=INPUT_DIM)

hidden_layer = bolt.graph.FullyConnected(dim=200, activation="relu")(input_layer)

output_layer = bolt.graph.FullyConnected(
    dim=VOCAB_SIZE, sparsity=0.01, activation="softmax"
)(hidden_layer)

model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

model.compile(bolt.CategoricalCrossEntropyLoss())

mlm_loader = dataset.MLMDatasetLoader(pairgram_range=INPUT_DIM)

train_data, train_tokens, train_labels = mlm_loader.load(
    filename=TRAIN_DATA, batch_size=BATCH_SIZE
)

test_data, test_tokens, test_labels = mlm_loader.load(
    filename=TEST_DATA, batch_size=BATCH_SIZE
)

train_config = (
    bolt.graph.TrainConfig.make(learning_rate=0.0001, epochs=1)
    .with_metrics(["mean_squared_error"])
    .with_rebuild_hash_tables(10000)
    .with_reconstruct_hash_functions(100000)
)

predict_config = bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"])

for e in range(20):
    model.train(train_data, test_tokens, train_labels, train_config)
    model.predict(test_data, test_tokens, test_labels, predict_config)
