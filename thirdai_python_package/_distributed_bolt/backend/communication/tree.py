import ray
import numpy as np

class Tree:
    def __init__(self, id, model, primary_worker, num_workers):
        self.id = id
        self.model = model
        self.primary_worker = primary_worker
        self.num_workers = num_workers
        self.w_gradients = []
        self.b_gradients = []

    def calculate_gradients(self, batch_id):
        self.model.calculate_gradients(batch_id)
        self.w_gradients, self.b_gradients = self.model.get_calculated_gradients()
        self.w_gradients = np.array(self.w_gradients)
        self.b_gradients = np.array(self.b_gradients)
        return True
    

    def receive_gradients(self):
        if self.id is 0:
            self.w_gradients, self.b_gradients = self.primary_worker.get_calculated_gradients()
        else:
            self.w_gradients, self.b_gradients = ray.get(
                self.primary_worker.get_calculated_gradients.remote()
            )
        self.model.set_gradients(self.w_gradients, self.b_gradients)
        return True

    def add_child_gradients(self, child_worker, avg_gradients):
        w_gradients_child, b_gradients_child = ray.get(child_worker.get_calculated_gradients.remote())
        self.w_gradients += w_gradients_child
        self.b_gradients += b_gradients_child
        if avg_gradients:
            self.w_gradients = np.divide(self.w_gradients, self.num_workers)
            self.b_gradients = np.divide(self.b_gradients, self.num_workers)

    def get_calculated_gradients(self):
        return self.w_gradients, self.b_gradients

