import dataloader
import network

#Загрузка данных MNIST
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# print(training_data[0])
training_data, test_data = dataloader.load_data()
# print(len(training_data))

#Настройка сети из 30 скрытых нейронов.
inputSize = 45000      #Размер входного слоя (28*28)
hiddenNeurons = 50   #Кол-во скрытых нейронов
outputSize = 3      #Размер выходного слоя
net = network.Network([inputSize, hiddenNeurons, 50, 50, outputSize])

#Наконец, используем стохастический градиентный спуск для обучения:
epoch = 15          #Кол-во эпох
miniPackage = 72    #Размер мини-пакета
LeaningSpeed = 1.0  #Скорость обучения (η)
net.SGD(training_data, epoch, miniPackage, LeaningSpeed, test_data=test_data)