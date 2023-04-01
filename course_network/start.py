import network
import dataloader

#Загрузка данных
training_data, test_data = dataloader.load_data()

#Настройка сети
inputSize = 52500      #Размер входного слоя (28*28)
outputSize = 10      #Размер выходного слоя
net = network.Network([inputSize, 20, outputSize])

epoch = 10          #Кол-во эпох
miniPackage = 24    #Размер мини-пакета
LeaningSpeed = 1    #Скорость обучения (η)
net.SGD(training_data, epoch, miniPackage, LeaningSpeed, test_data=test_data)