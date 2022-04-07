from thirdai import bolt, dataset
import numpy as np
import csv
import mmh3
import re

# Wrapper for predicting the sentiment of one sentence. User specifies the network
# and the sentence to be predicted. Returns 1 for positive, 0 for negative.
def predict_sentence_sentiment(network: bolt.Network, text, seed=42):
    feat_hash_dim = 100000

    # Remove punctuations and convert to lowercase
    sentence = re.sub(r"[^\w\s]", "", text)
    sentence = sentence.lower()

    tup = dataset.bolt_tokenizer(sentence, seed=seed, dimension=feat_hash_dim)
    x_idxs = tup[0]
    x_idxs = np.array(x_idxs)
    x_offsets = np.int32([0, len(x_idxs)])
    x_vals = tup[1]
    y_idxs = np.array([np.uint32(0)])
    y_vals = np.array([np.float32(1)])
    y_offsets = np.array([np.uint32(10), np.uint32(10)])

    temp = network.predict(
        x_idxs,
        x_vals,
        x_offsets,
        y_idxs,
        y_vals,
        y_offsets,
        batch_size=1,
        metrics=["categorical_accuracy"],
        verbose=False,
    )
    pred = np.argmax(temp[1][0])

    return 1 if pred > 0 else 0


# Wrapper for preprocessing text file into svm format. The function accepts .csv file
# with two columns, where the first column is a single word "pos" or "neg", and the
# second column is the sentence.
# The function returns the path to the svm file.
def preprocess_data(file_name, is_train, target_location=None, seed=42):
    dimension = 100000
    if file_name.find(".csv") == -1:
        raise ValueError("Only .csv files are supported")

    post = "_train" if is_train else "_test"
    if target_location is None:
        target_location = "preprocessed_data"
    target_location = target_location + post + ".svm"

    fw = open(target_location, "w")
    csvreader = csv.reader(open(file_name, "r"))

    for line in csvreader:
        label = 1 if line[0] == "pos" else 0
        fw.write(str(label) + " ")

        # Remove punctuations and convert to lowercase
        sentence = re.sub(r"[^\w\s]", "", line[1])
        sentence = sentence.lower()

        # Tokenize the sentence and featurized it
        tup = dataset.bolt_tokenizer(sentence, seed=seed, dimension=dimension)
        for idx, val in zip(tup[0], tup[1]):
            fw.write(str(idx) + ":" + str(val) + " ")

        fw.write("\n")
    fw.close()

    return target_location
