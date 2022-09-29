import ray
import pygloo
from ...utils import get_gradients, set_gradients
import numpy as np



class GlooBackend:
    def __init__(self, model, id, num_workers):
        self.model = model
        self.id = id
        self.gradients = []
        self.averaged_gradients = []
        self.context = pygloo.rendezvous.Context(id, num_workers)
        # Prepare device and store for rendezvous
        attr = pygloo.transport.tcp.attr("localhost")
        dev = pygloo.transport.tcp.CreateDevice(attr)
        file_store_path = f"{ray.worker._global_node.get_session_dir_path()}" + "/collective/gloo/rendezvous"
        fileStore = pygloo.rendezvous.FileStore(file_store_path)
        store = pygloo.rendezvous.PrefixStore(str(num_workers), fileStore)

        self.context.connectFullMesh(store, dev)
    
    def compute_and_store_batch_gradients(self, batch_no):
        self.model.compute_and_store_batch_gradients(batch_no)



    def all_reduce_gradients(self, num_workers):
        self.gradients = np.array(get_gradients(self.model))
        self.averaged_gradients = np.zeros_like(self.gradients, dtype=np.float32)
        sendptr = self.gradients.ctypes.data
        recvptr = self.averaged_gradients.ctypes.data
        pygloo.allreduce(self.context, sendptr, recvptr,
                        self.gradients.size, pygloo.glooDataType_t.glooFloat32,
                        pygloo.ReduceOp.SUM, pygloo.allreduceAlgorithm.RING)
        
       
        for gradient_id in range(len(self.averaged_gradients)):
            self.averaged_gradients[gradient_id] /= num_workers


    def receive_gradients(self):
        set_gradients(self.model, self.averaged_gradients)