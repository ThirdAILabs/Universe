from thirdai import bolt, dataset
import numpy as np
import csv
from sklearn.utils import murmurhash3_32 as mmh3
import re

def predict_sentence_sentiment(network: bolt.Network, text, seed=42):
    feat_hash_dim = 100000
    
    sentence = re.sub(r'[^\w\s]','', text)
    sentence = sentence.lower()
    
    x_idxs = [mmh3(itm, seed=seed) % feat_hash_dim for itm in sentence.split()]
    x_idxs = np.array(x_idxs)
    x_offsets = np.int32([0,len(x_idxs)])
    x_vals = np.ones(len(x_idxs))
    y_idxs = np.array([np.uint32(0)])
    y_vals = np.array([np.float32(1)])
    y_offsets = np.array([np.uint32(10), np.uint32(10)])

    temp = network.predict(
        x_idxs, x_vals, x_offsets, 
        y_idxs, y_vals, y_offsets, 
        batch_size=1, 
        metrics=["categorical_accuracy"],
        verbose=False
    )
    pred = np.argmax(temp[1][0])

    # if pred > 0:
    #     print('positive!', flush=True)
    # else:
    #     print('negative!', flush=True)
    
    return 1 if pred > 0 else 0


def preprocess_data(file_name, batch_size, target_location=None, train=True, seed=42):
    if file_name.find(".csv") == -1:
        raise ValueError("Only .csv files are supported")

    post = "_train" if train else "_test"
    if target_location is None:
        target_location = "preprocessed_data"
    target_location = target_location + post + ".svm"

    fw = open(target_location, 'w')
    csvreader = csv.reader(open(file_name, 'r'))

    for line in csvreader:
        label = 1 if line[0] == "pos" else 0
        fw.write(str(label) + ' ')

        sentence = re.sub(r'[^\w\s]','', line[1])
        sentence = sentence.lower()
        ### BOLT TOKENIZER START
        tup = dataset.bolt_tokenizer(sentence)
        for idx, val in zip(tup[0], tup[1]):
            fw.write(str(idx) + ':' + str(val) + ' ')
        ### BOLT TOKENIZER END

        fw.write('\n')
    fw.close()

    return target_location

# Just tests
if __name__ == "__main__":
    rows = [['pos', 'I love this movie'], ['neg', 'I hate this movie']]
    with open('countries.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    preprocess_data('countries.csv', 1, train=True)