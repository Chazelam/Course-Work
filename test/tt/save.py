import numpy as np

sizes = [30, 10, 2]

biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

b = np.arange(12).reshape(3, 4)
print(b)

np.save("example", b)
new = np.load("example.npy")
#ff
print(new)