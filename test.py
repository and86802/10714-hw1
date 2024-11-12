import sys
sys.path.append("./python")
sys.path.append("./apps")
sys.path.append("./tests/hw1")
import needle as ndl
from simple_ml import *
from test_autograd_hw import *
import numpy as np

# print(gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(3, 1)), shape=(3, 3)))
x = ndl.Tensor(np.random.randn(3, 1))
print(x.backward())