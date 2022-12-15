import pickle as cPickle
import gzip
import numpy as np


"""
Ф-я позращает данные MNIST в виде кортежа, 
содержащего обучающие данные, данные проверки и данные испытаний.

"training_data" возвращается как кортеж с двумя записями.
  Первая запись:
    Фактические тренировочные изображения. 
    Это numpy.ndarray из 50 000 записей. 
    Каждая запись, в свою очередь, numpy.ndarray с 784(28*28) значениями пикселей изображения.
  Вторая запись: 
    это numpy.ndarray содержащий 50 000 записей. 
    Эти записи - просто цифры от 0 до 9 для соответствующих изображений (Ответы)

"validation_data" и "test_data" тоже самое что и "training_data", но содержат по 10 000 изображений.
"""
def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    u = cPickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)


"""
Возвращает кортеж, (training_data, validation_data, test_data). 
На основе ``load_data``, но в формате более удобном для использования 
в нашей реализации нейронных сетей.

В частности, "training_data" представляет собой список, 
  содержащий 50 000 кортежий "(x, y)". 
   "x" - это 784-мерный numpy.ndarray содержащий входное изображение. 
   "y" является 10-мерным numpy.ndarray, представляющий единичный вектор,  
    соответствующий правильной цифре для "x".

"validation_data" и "test_data" — это списки, содержащие 10 000 кортежей "(x, y)". 
    "x" - это 784-мерный numpy.ndarray содержащий входное изображение.
    "y" - это соответствующая классификация, т. е. цифровые значение соответствующего "x".
"""
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
