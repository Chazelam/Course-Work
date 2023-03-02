# Задача: 
#  Пройтись по всем изображениям
#   посчитать
#    1. макс. х , макс у
#     2. ср. значения

import cv2 as cv

def avrg(x):
    return int(sum(x)/len(x))

Ys = []
Xs = []
n = 23
for i in range(1, n + 1):
    for j in range(1, 25):
        # Открытие файла
        file_name = "/root/VScode/NeuralNetworks/new_data/{0}/original_{1}_{2}.png".format(i, i, j)
        img = cv.imread(file_name)
        Y, X, Channels = img.shape
        if X != 500 or Y != 500:
            print(file_name, "\n", X, Y)
        Ys.append(Y)
        Xs.append(X)

print("max x is ", max(Xs))
print("max y is ", max(Ys))
print("avrg is {0}x{1}".format(avrg(Xs), avrg(Ys)))