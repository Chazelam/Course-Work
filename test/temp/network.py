import random
import numpy as np

class Network(object):
    # Выполняется автоматически
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Создание вектора смещений (Nx1), где N - кол-во слоев - 1(входной слой)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Создание вектора весов (1 x N), где N - кол-во слоев - 1(входной слой)
        # Каждый элемент которого это numpy.array(l x l-1), где l - номер слоя
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # Расчет выхода сети по заданным весам и смещениям
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a # Выход - вектор (10х1)??????

    # Стахастический градиентный спуск
    def SGD(self, training_data, epochs, mini_batch_size, learningRate, test_data = None):
        # "training_data" - список из кортежей (inputData, outputData), 
        # где inputData - обучающие входные данные(784 пикселя) и outputData - желаемый выход(число от 1 до 10).
        if test_data:               # Если указали необязательный пар-м test_data(эталон)
            n_test = len(test_data) # Задаём значение n_test
        n = len(training_data)

        for epoch in range(epochs):
            # Перемешивание тренировачных данных
            random.shuffle(training_data)
            # Создание массива мини выборок
            # Каждый элемент - отдельный мини пакет данных
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:             # Для каждого пакета
                self.update_mini_batch(mini_batch, learningRate) # Изменяем веса и смещения в соответствии с эталоном
            
            # Если указан "test_data", то сеть будет оцениваться по тестовым данным после каждой эпохи. 
            if test_data:
                print("Epoch {0}: {1} / {2}:".format(epoch, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(epoch))


    # Изменяет веса и смещения с использованием градиентного спуска
    def update_mini_batch(self, mini_batch, learningRate):
        # Создание нулевого массивов той же формы что и biases, weights 
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for inputData, outputData in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(inputData, outputData)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Изменяем веса и смещения
        self.weights = [w-(learningRate/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learningRate/len(mini_batch))*nb 
                        for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, inputData, outputData):
        # Создание нулевого массивов той же формы что и biases, weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # feedforward
        activation = inputData    # Значения нейронов слоя
        activations = [inputData] # Массив значений нейронов, слой за слоем
        zs = []                   # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights): # Для каждого слоя:
            z = np.dot(w, activation)+b    # Подсчет Z ф-и
            zs.append(z)                   # добавление ее в массив
            activation = sigmoid(z)        # Подсчет активаций слоя
            activations.append(activation) # добавление их в массив
        
        # Обратное распространение ошибки (см. алгоритм в конспекте)
        delta = self.cost_derivative(activations[-1], outputData) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    # Значение ф-и ошибки
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(inputData)), outputData) for (inputData, outputData) in test_data]
        return sum(int(inputData == outputData) for (inputData, outputData) in test_results)

    def cost_derivative(self, output_activations, outputData):
        return (output_activations-outputData)

# Ф-я активации - сигмоида
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Производная от ф-и активации
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))