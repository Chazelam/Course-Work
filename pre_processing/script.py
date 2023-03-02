import numpy as np
import cv2 as cv

# Расчет порога бинаризации методом Оцу 
def otsy(img):
    bins_num = 256
    hist, bins_edges = np.histogram(img, bins_num)    # построение гистограммы значений пикселей
    bin_mids = (bins_edges[:-1] + bins_edges[1:]) / 2 # Нахождение центра каждого отрезка
    # Итерация по массиву и нахождение вероятностей каждого значения пикселя
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_mids) / weight1
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    index_of_max_val = np.argmax(inter_class_variance)
    threshold = bin_mids[:-1][index_of_max_val]
    return threshold

nX = 300
nY = 150
right_ratio = nX/nY
n = 1 #23
m = 1 #24

for i in range(1, n + 1):
    for j in range(1, m + 1):
        # Открытие файла
        file_name = "/root/VScode/NeuralNetworks/Data[CEDAR]/{0}/original_{1}_{2}.png".format(i, i, j)
        original_img = cv.imread(file_name)
        # Преобразование в полу-тоновое
        gray_img = cv.cvtColor(original_img, cv.COLOR_RGB2GRAY)
        T = otsy(gray_img) # Нахождение порога бтнаризации методом Оцу
        ret, bin_img = cv.threshold(gray_img, T, 255, 0) # Бинаризация
        median = cv.medianBlur(bin_img, 5) # Медианная фильтрация масокй 5х5
        non_median = cv.bitwise_not(median) # Инвертирование
        non_zero = np.nonzero(non_median) # Нахождение всех не нулевых пикселей
        # Обрезание нулей
        img = non_median[min(non_zero[0]):max(non_zero[0]), min(non_zero[1]):max(non_zero[1])]
        # Преобразование к формату 500х500
        Y, X = np.shape(img)
        current_ratio = X / Y
        if current_ratio == right_ratio:
            final = cv.resize(img, (nX, nY))
        elif current_ratio > right_ratio:
            final = cv.resize(img, (nX, int(Y * (nX/X))))
            missing = nY - int(Y * (nX/X))
            if missing % 2 == 0:
                top = bottom = missing // 2
            else:
                top = missing // 2
                bottom = top + 1
            final = cv.copyMakeBorder(final, top, bottom, 0, 0, cv.BORDER_CONSTANT, None, value = 0)
        else:
            final = cv.resize(img, (int(X * (nY/Y)), nY))
            missing = nX - int(X * (nY/Y))
            if missing % 2 == 0:
                left = right = missing // 2
            else:
                left = missing // 2
                right = left + 1
            final = cv.copyMakeBorder(final, 0, 0, left, right, cv.BORDER_CONSTANT, None, value = 0)
        print(type(final))
        cv.imwrite("/root/VScode/NeuralNetworks/new_data/{0}/original_{1}_{2}.png".format(i, i, j), final)        