import pytest

pytestmark = [pytest.mark.unit, pytest.mark.integration]
from thirdai import dataset, bolt
from thirdai.dataset import blocks,text_encodings
import numpy as np
import mmh3
import os

def combine_hashes(a,b):
    """
    This is the standard implementation of combineHashes in hashing::HashUtils
    """
    c = b + np.uint32(0x9e3779b9) + (a << 6) + (a >> 2)
    return a^c

def get_expected_pairgrams(sentence):
    words = sentence.split()
    svm_line = "0 "
    hash_values = []
    for i in range(len(words)):
        hash_value = np.uint32(mmh3.hash(words[i],341)) # we need uint32 to be consistent with our cpp murmurhash return type
        hash_values.append(hash_value)
    for i in range(len(hash_values)):
        for j in range(i,len(hash_values)):
            hash_value = (combine_hashes(hash_values[i],hash_values[j]))%100
            svm_line = svm_line + str(hash_value)+":1 "
    with open('pairgrams.svm', 'w') as file:
        file.write(svm_line)
    x,y = dataset.load_bolt_svm_dataset('pairgrams.svm',1)
    os.remove('pairgrams.svm')
    return x,y

def get_expected_unigrams(sentence):
    words = sentence.split()
    svm_line = "0 "
    for i in range(len(words)):
        hash_value = (np.uint32(mmh3.hash(words[i], 341)) % 100)
        svm_line = svm_line + str(hash_value)+":1 "
    with open('unigrams.svm', 'w') as file:
        file.write(svm_line)
    x,y = dataset.load_bolt_svm_dataset('unigrams.svm',1)
    os.remove('unigrams.svm')
    return x,y

def get_actual_unigrams(sentence):
    return dataset.string_to_bolt_dataset(sentence,100)

def get_actual_pairgrams(sentence):
    return dataset.string_to_bolt_dataset(sentence,100,"pairgrams")
    

def build_network():
    layers = [
        bolt.FullyConnected(
            dim=100,
            activation_function=bolt.ActivationFunctions.ReLU,
        ),
        bolt.FullyConnected(
            dim=10,
            activation_function=bolt.ActivationFunctions.ReLU,
        ),
        bolt.FullyConnected(
            dim=3,
            activation_function=bolt.ActivationFunctions.Softmax,
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=100)
    return network


def test_sentence_to_boltdataset_unigrams():
    """
    checks the activations are same when we get the bolt datsets 
    from two different ways, this is to ensure we are getting same
    boltdatsets from two methods.
    """
    x,y = get_expected_unigrams("hello world how are you")
    z = get_actual_unigrams("hello world how are you")
    network = build_network()
    metrics1,act1 = network.predict(x,y)
    metrics2,act2 = network.predict(z,None)
    assert act1.all() == act2.all()

def test_sentence_to_boltdataset_pairgrams():
    x,y = get_expected_pairgrams("hello world how are you")
    z = get_actual_pairgrams("hello world how are you")
    network = build_network()
    metrics1,act1 = network.predict(x,y)
    metrics2,act2 = network.predict(z,None)
    assert act1.all() == act2.all()
