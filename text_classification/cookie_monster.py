from thirdai import bolt, dataset
import numpy as np

class CookieMonster():
    def __init__(self, size="2Gb", output_dimension=2):
        self.model_size = size
        self.bolt_classifier = bolt.TextClassifier(model_size=size, n_classes=output_dimension)
        self.out_dim = output_dimension
    
    def download_hidden_weights(self, path_to_weights):
        # download weights from mlflow and load them into bolt
        pass

    def upload_hidden_weights(self, path_to_weights):
        pass

    def set_output_dimension(self, dimension):
        weights = self.bolt_classifier.get_hidden_weights()
        biases = self.bolt_classifier.get_hidden_biases()
        del(self.bolt_classifier) # delete old model

        self.bolt_classifier = bolt.TextClassifier(model_size=self.model_size, n_classes=dimension)

        self.bolt_classifier.set_hidden_weights(weights)
        self.bolt_classifier.set_hidden_biases(biases)

    
    