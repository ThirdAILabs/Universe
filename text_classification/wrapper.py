from thirdai import bolt, dataset
import numpy as np
from sklearn.utils import murmurhash3_32 as mmh3
import re
from collections import defaultdict

def predict_sentence_sentiment(network: bolt.Network, text, seed=42):
    feat_hash_dim = 100000
    
    sentence = re.sub(r'[^\w\s]','', text)
    sentence = sentence.lower()
    
    x_idxs = [mmh3.hash(itm, seed) % feat_hash_dim for itm in sentence.split()]
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
        metrics=["categorical_accuracy"]
    )
    pred = np.argmax(temp[1][0])

    if pred > 0:
        print('positive!', flush=True)
    else:
        print('negative!', flush=True)


def preprocess_data(file_name, batch_size, target_location=None, train=True, seed=42):
    if file_name.find(".csv") == -1:
        raise ValueError("Only .csv files are supported")

    post = "_train" if train else "_test"
    if target_location is None:
        target_location = "preprocessed_data"
    target_location = target_location + post + ".svm"

    fw = open(target_location, 'w')
    with open(file_name) as f:
        for line in f:
            sentence = re.sub(r'[^\w\s]','', line)
            sentence = sentence.lower()
            itms = sentence.split()
            label = 1 if itms[0] == "pos" else 0
            fw.write(str(label) + ' ')

            for single in itms[1:]:
                temp = mmh3(single, seed=seed, positive=True) % murmur_dim
                raw[temp] += 1

            for k,v in raw.items():
                fw.write(str(k) + ':' + str(v) + ' ')
    
            fw.write('\n')
    fw.close()