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

file_name = "/root/VScode/NeuralNetworks/pre-processingHandwrittenSignatureImage/signatureExample(CEDAR 21)/original_21_2.png"
original_img = cv.imread(file_name)
gray_img = cv.cvtColor(original_img, cv.COLOR_RGB2GRAY)
T = otsy(gray_img)
ret, bin_img = cv.threshold(gray_img, T, 255, 0)
cv.imwrite("bin.png", bin_img)
median = cv.medianBlur(bin_img, 5)
cv.imwrite("median.png", median)