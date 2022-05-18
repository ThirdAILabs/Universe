from thirdai import bolt
import numpy as np

network = bolt.Network(layers=[], input_dim=0)

# network.example(data="hello")

x = np.array([1, 2, 3], dtype=np.uint32)
network.example(data=x)

x = np.array([1.1, 2.2, 37955.94023], dtype=np.float32)
network.example(data=x)
