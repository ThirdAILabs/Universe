from thirdai import bolt, dataset
import numpy as np

class CookieMonster():
    def __init__(self, size="small"):
        self.model_size = size
        self.bolt_classifier = bolt.TextClassifier(model_size=size, n_classes=2)
    
    def load_hidden_weights(self, path_to_weights):
        #TODO(Henry): Add wrapper in text classifier to set weights&biases
        # download weights from mlflow and load them into bolt
        pass

    def set_output_dimension(self, dimension):
        #TODO(Henry): Add wrapper in text classifier to get weights&biases
        weights = self.bolt_classifier.get_weights()
        biases = self.bolt_classifier.get_biases()
        del(self.bolt_classifier) # delete old model

        self.bolt_classifier = bolt.TextClassifier(model_size=self.model_size, n_classes=dimension)

        self.bolt_classifier.set_weights(weights)
        self.bolt_classifier.set_biases(biases)

    
    