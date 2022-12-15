import mnist_loader
import network

#Загрузка данных MNIST
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

print(training_data[0])



# #Настройка сети из 30 скрытых нейронов.
# inputSize = 784      #Размер входного слоя (28*28)
# hiddenNeurons = 30   #Кол-во скрытых нейронов
# outputSize = 10      #Размер выходного слоя
# net = network.Network([inputSize, hiddenNeurons, outputSize])

# #Наконец, используем стохастический градиентный спуск для обучения:
# epoch = 1           #Кол-во эпох
# miniPackage = 10    #Размер мини-пакета
# LeaningSpeed = 3.0  #Скорость обучения (η)
# net.SGD(training_data, epoch, miniPackage, LeaningSpeed, test_data=test_data)  