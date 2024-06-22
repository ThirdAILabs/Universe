print("Loading embeddings...")
import numpy as np
train_embeddings = np.load("train_embeddings.np", allow_pickle=True)
val_embeddings = np.load("val_embeddings.np", allow_pickle=True)
test_embeddings = np.load("test_embeddings.np", allow_pickle=True)

print("Loading labels...")
train_labels = np.load("train_labels.np", allow_pickle=True)
val_labels = np.load("val_labels.np", allow_pickle=True)
test_labels = np.load("test_labels.np", allow_pickle=True)

print("Initializing model...")
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

print("Fitting...")
from utils import run_and_time
run_and_time(lambda: model.fit(train_embeddings, train_labels))

print("Evaluating...")
from utils import accuracy
predictions = model.predict(test_embeddings)
print("Accuracy:", accuracy(predictions, test_labels))