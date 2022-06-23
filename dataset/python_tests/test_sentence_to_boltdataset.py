import pytest

pytestmark = [pytest.mark.unit, pytest.mark.integration]
from thirdai import dataset, bolt
from thirdai.dataset import blocks,text_encodings
import numpy as np
import mmh3
import os

def combine_hashes(a,b):
    """
    we are combining hashes this way because , this is the 
    way geordie had done.
    """
    c = b + np.uint32(0x9e3779b9) + (a << 6) + (a >> 2)
    return a^c
def get_bolt_dataset_from_python_pairgrams(sentence):
    words = sentence.split()
    line = "0 "
    temp = []
    for i in range(len(words)):
        hash_value = (np.uint32(mmh3.hash(words[i],341)))%100000
        temp.append(hash_value)
    for i in range(len(temp)):
        for j in range(i,len(temp)):
            hash_value = (combine_hashes(temp[i],temp[j]))%100000
            line = line + str(hash_value)+":1 "
    with open('hello.svm', 'w') as file:
        file.write(line)
    x,y = dataset.load_bolt_svm_dataset('hello.svm',256)
    os.remove('hello.svm')
    return x,y

def get_bolt_dataset_from_python_unigrams(sentence):
    words = sentence.split()
    line = "0 "
    for i in range(len(words)):
        hash_value = (np.uint32(mmh3.hash(words[i],341)))%100000
        line = line + str(hash_value)+":1 "
    with open('unigrams.svm', 'w') as file:
        file.write(line)
    x,y = dataset.load_bolt_svm_dataset('unigrams.svm',256)
    os.remove('unigrams.svm')
    return x,y
def get_bolt_dataset_from_dataset_unigrams(sentence):
    temp = dataset.sentence_to_boltdataset(sentence,blocks.Text(col=0, encoding=text_encodings.UniGram(dim=100000)))
    return temp

def get_bolt_dataset_from_dataset_pairgrams(sentence):
    temp = dataset.sentence_to_boltdataset(sentence,blocks.Text(col=0, encoding=text_encodings.PairGram(dim=100000)))
    return temp

def build_network():
    layers = [
        bolt.FullyConnected(
            dim=100,
            sparsity=0.2,
            activation_function=bolt.ActivationFunctions.ReLU,
        ),
        bolt.FullyConnected(
            dim=10,
            sparsity=0.3,
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
    x,y = get_bolt_dataset_from_python_unigrams("hello world how are you")
    z = get_bolt_dataset_from_dataset_unigrams("hello world how are you")
    network = build_network()
    temp1,act1 = network.predict(x,y)
    temp2,act2 = network.predict(z,None)
    assert act1.all() == act2.all()

def test_sentence_to_boltdataset_pairgrams():
    x,y = get_bolt_dataset_from_python_pairgrams("hello world how are you")
    z = get_bolt_dataset_from_dataset_pairgrams("hello world how are you")
    network = build_network()
    temp1,act1 = network.predict(x,y)
    temp2,act2 = network.predict(z,None)
    assert act1.all() == act2.all()
