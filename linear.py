import time

import numpy as np
import torch
from thirdai import smx

N, M, K = 2000, 50000, 1000

x = np.random.rand(N, K).astype(np.float32)
w = np.random.rand(M, K).astype(np.float32)
b = np.random.rand(M).astype(np.float32)

y = np.matmul(x, w.T) + b

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

tx = torch.from_numpy(x)
tw = torch.from_numpy(w)
tw.requires_grad = True
tb = torch.from_numpy(b)
tb.requires_grad = True

torch_linear = torch.nn.Linear(M, K)
torch_linear.weight = torch.nn.Parameter(tw)
torch_linear.bias = torch.nn.Parameter(tb)

s = time.perf_counter()
# ty = torch.matmul(tx, torch.transpose(tw, 0, 1)) + tb
ty = torch_linear(tx)
e = time.perf_counter()

print(f"torch: {(e-s):.3f}")

assert np.allclose(Y.tensor.numpy(), ty.detach().numpy())
