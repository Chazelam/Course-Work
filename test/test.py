import numpy as np
weights = [[1, 2, 3],
           [4, 4, 2]]

biases = [[2], [1]]
a = [4, 2, 1]

for b, w in zip(biases, weights):
    x = np.dot(w, a)
    #print(x)


sizes = [30, 5, 2]
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
print(weights) 
biases = [np.random.randn(y, 1) for y in sizes[1:]]
#print(biases) 