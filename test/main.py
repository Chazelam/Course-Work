import numpy as np

def sigmoid(x):
    # Наша функция активации: f(x) = 1 / (1 + e^(-x))
    return 1/(1 + np.exp(-x))

class NeuralNetwork:
    #создание самой сети
    def __init__(self, x, y):
        self.input      = x                                     #входний слой
        self.weights1   = np.random.rand(self.input.shape[1],4) #Веса
        self.weights2   = np.random.rand(4,1)                   #Веса
        self.y          = y                                     #Скрытый слой
        self.output     = np.zeros(y.shape)                     #Выходной слой

    #Ф-я прямого распрастранения(активация нейронов)
    def feedforward(self):
        #<слой> = <ф-я активации>(<матричное умножение>(<преведущий слой>, <вектор весов соединяющих эти слои>) + <смещение>)
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))  #Вычисление 1 слоя
        self.output = sigmoid(np.dot(self.layer1, self.weights2)) #Вычисление выходного слоя

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

x = np.array([1, 1, 1])
print(NeuralNetwork(x, 1).weights1)
