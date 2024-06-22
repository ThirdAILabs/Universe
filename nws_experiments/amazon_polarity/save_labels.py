print("Loading labels...")
from datasets import load_dataset
ds = load_dataset("mteb/amazon_polarity", "default")
train_labels = ds["train"]["label"]
val_labels = ds["validation"]["label"]
test_labels = ds["test"]["label"]

print("Saving labels...")
import numpy as np
np.array(train_labels).dump("train_labels.np")
np.array(val_labels).dump("val_labels.np")
np.array(test_labels).dump("test_labels.np")