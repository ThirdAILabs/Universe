import numpy as np
import ray





@ray.remote(num_cpus=2, max_restarts=2)
class Supervisor:
    def __init__(self, layers, workers, config):
        self.layers = layers
        self.workers = workers
        self.num_of_batches = ray.get(self.workers[0].num_of_batches.remote())
        self.weights_biases = ray.get(self.workers[0].returnParams.remote())
        
    

    def batch_to_train(self, id, batch_no):
        return int(self.num_of_batches/len(self.workers))*id + batch_no

    def subworkCircularCommunication(self, batch_no):
        updates = ray.get([self.workers[id].calculateGradientsCircular.remote(self.batch_to_train(id, batch_no)) for id in range(len(self.workers))])

        # First Run
        update_id = 0
        for i in range(self.no_of_workers-1):
            if i == self.no_of_workers - 2:
                blocking_run = ray.get([w.processRing.remote(update_id, avg_gradients=True) for w in self.workers])
            else:
                blocking_run = ray.get([w.processRing.remote(update_id) for w in self.workers])
            update_id -= 1

        # Second Run 
        update_id = 1
        for i in range(self.no_of_workers-1):
            blocking_run = ray.get([w.processRing.remote(update_id, reduce=False) for w in self.workers])
            update_id -= 1
        return True

    def subworkLinearCommunication(self, batch_no):
        gradients_list = ray.get([self.workers[id].calculateGradientsLinear.remote(self.batch_to_train(id, batch_no)) for id in range(len(self.workers))])
        gradients_list = ray.get(gradients_list)

        w_gradients_avg = np.array([np.zeros((self.layers[layer_no+1], self.layers[layer_no])) for layer_no in range(len(self.layers)-1)])
        b_gradients_avg = np.array([np.zeros((self.layers[layer_no+1])) for layer_no in range(len(self.layers)-1)])


        for w_gradients,b_gradients in gradients_list:
            w_gradients_avg += w_gradients
            b_gradients_avg += b_gradients
        
        self.w_gradients_avg = w_gradients_avg/len(self.workers)
        self.b_gradients_avg = b_gradients_avg/len(self.workers)
        return True

    
    def gradients_avg(self):
        return self.w_gradients_avg, self.b_gradients_avg

    
    def subworkUpdateParameters(self, learning_rate):
        check_update_parameter = ray.get([w.updateParameters.remote(learning_rate) for w in self.workers])
        return True

    
    def check_weights(self):
        weights_0, biases_0 = ray.get(self.workers[0].returnParams.remote())
        weights_1, biases_1 = ray.get(self.workers[1].returnParams.remote())
        print('weights 0: ', weights_0)
        print('weights 1: ', weights_1)
        print('biases 0: ', biases_0)
        print('biases 1: ', biases_1)


    def weights_biases(self):
        return self.weights_biases

