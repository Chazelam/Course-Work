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

# Открытие файла
file_name = "/root/VScode/NeuralNetworks/pre-processingHandwrittenSignatureImage/signatureExample(CEDAR 21)/original_21_2.png"
original_img = cv.imread(file_name)
# Преобразование в полу-тоновое
gray_img = cv.cvtColor(original_img, cv.COLOR_RGB2GRAY)
T = otsy(gray_img) # Нахождение порога бтнаризации методом Оцу
ret, bin_img = cv.threshold(gray_img, T, 255, 0) # Бинаризация
median = cv.medianBlur(bin_img, 5) # Медианная фильтрация масокй 5х5
non_median = cv.bitwise_not(median) # Инвертирование
non_zero = np.nonzero(non_median) # Нахождение всех не нулевых пикселей
# Обрезание нулей
croped = non_median[min(non_zero[0]):max(non_zero[0]), min(non_zero[1]):max(non_zero[1])]
cv.imwrite("croped.png", croped)
# Преобразование к формату 300х150
final = cv.resize(croped, (300, 150), 1, 1)
cv.imwrite("fin.png", final)