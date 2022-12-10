import numpy as np
sizes = [2, 2, 2]
x = [np.random.randn(y, 1) for y in sizes[1:]]
y = [np.random.sample((sizes[0], sizes[1]))]
z = np.random.rand(4,1)
print(z)
