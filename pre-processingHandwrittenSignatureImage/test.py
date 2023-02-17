import cv2 as cv
import numpy as np

nX = nY = 500
right_ratio = nX/nY

file_name = "/root/VScode/NeuralNetworks/new_data/19/original_19_1.png"
img = cv.imread(file_name)
Y, X, Channels = np.shape(img)
print(X, "x", Y)
current_ratio = X / Y
if current_ratio == right_ratio:
    final = cv.resize(img, (nX, nY))
elif current_ratio > right_ratio:
    final = cv.resize(img, (500, int(Y * (nX/X))))
    missing = nY - int(Y * (nX/X))
    if missing % 2 == 0:
        top = bottom = missing // 2
    else:
        top = missing // 2
        bottom = top + 1
    final = cv.copyMakeBorder(final, top, bottom, 0, 0, cv.BORDER_CONSTANT, None, value = 0)
    cv.imwrite("fin.png", final)
else:
    final = cv.resize(img, (int(X * (nY/Y)), 500))
    print(int(X * (nY/Y)), "x500")
    missing = nX - int(X * (nY/Y))
    print("miss - ", missing)
    if missing % 2 == 0:
        left = right = missing // 2
    else:
        left = missing // 2
        right = top + 1
    
    final = cv.copyMakeBorder(final, 0, 0, left, right, cv.BORDER_CONSTANT, None, value = 0)
    cv.imwrite("fin.png", final) 