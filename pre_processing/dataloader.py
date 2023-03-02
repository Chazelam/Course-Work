import numpy as np
import cv2 as cv

def vectorized_result(j, n):
    e = np.zeros((n, 1))
    e[j - 1] = 1.0
    return e

def load_data():
    training_data = []
    n = 2 #23
    m = 2 #24
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Открытие файла
            file_name = "/root/VScode/NeuralNetworks/DATA/new_data/{0}/original_{1}_{2}.png".format(i, i, j)
            img = cv.imread(file_name, cv.IMREAD_GRAYSCALE)
            img = np.reshape(img, (-1, 1))
            img = img.astype("float32")
            res = list([img, vectorized_result(i, n)])
            training_data.append(res)
            return training_data