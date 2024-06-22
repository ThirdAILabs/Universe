print("Loading embeddings...")
import numpy as np
train_embeddings = np.load("train_embeddings.np", allow_pickle=True)
val_embeddings = np.load("val_embeddings.np", allow_pickle=True)
test_embeddings = np.load("test_embeddings.np", allow_pickle=True)

print("Loading labels...")
train_labels = np.load("train_labels.np", allow_pickle=True)
val_labels = np.load("val_labels.np", allow_pickle=True)
test_labels = np.load("test_labels.np", allow_pickle=True)

models = {}

def run_model_at_scale(scale, models):
    print("Scale:", scale)
    print("Initializing NWS")
    from thirdai.bolt import NWS, L2Hash
    srp = L2Hash(
        input_dim=train_embeddings.shape[1],
        hashes_per_row=1,
        rows=1000,
        scale=scale,
        seed=314,
    )
    model = NWS(srp, sparse=True)
    models[scale] = model
    print("Training...")
    from utils import run_and_time
    run_and_time(lambda: model.train(train_embeddings, train_labels))
    print("Evaluating...")
    from utils import accuracy
    raw_preds = model.predict(test_embeddings)
    predictions = np.round(np.array(model.predict(test_embeddings))).astype(np.int32)
    acc = accuracy(predictions, test_labels)
    print(f"At {scale=}, accuracy:", acc)
    models[scale] = {
        'model': model,
        'raw_preds': raw_preds,
        'preds': predictions,
        'acc': acc,
    }

for scale in [1.0, 0.5, 2.0, 0.25, 0.1]:
    run_model_at_scale(scale, models)