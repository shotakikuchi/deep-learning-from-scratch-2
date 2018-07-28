import os
import sys

sys.path.append(os.pardir)

import numpy as np
from common.layers import Matmul

# sample context data
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# initialize weights
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# create layers
in_layer0 = Matmul(W_in)
in_layer1 = Matmul(W_in)
out_layer = Matmul(W_out)

# forward
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)

print(s)

