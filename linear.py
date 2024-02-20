import time

import numpy as np
from thirdai import smx

N, M, K = 2000, 50000, 1000

x = np.random.rand(N, K).astype(np.float32)
w = np.random.rand(M, K).astype(np.float32)
b = np.random.rand(M).astype(np.float32)


linear = smx.Linear(M, K)
linear.weight = smx.Variable(smx.from_numpy(w), requires_grad=True)
linear.bias = smx.Variable(smx.from_numpy(b), requires_grad=True)

X = smx.Variable(smx.from_numpy(x), requires_grad=True)

s = time.perf_counter()
Y = linear(X)
e = time.perf_counter()

print(f"smx: {(e-s):.3f}")


s = time.perf_counter()
y = np.matmul(x, w.T) + b
e = time.perf_counter()

print(f"np : {(e-s):.3f}")

assert np.allclose(Y.tensor.numpy(), y)
