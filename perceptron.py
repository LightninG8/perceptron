import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptyc_weights = 2 * np.random.random((3,1)) - 1

print('Случайные инициализирующие веса:')
print(synaptyc_weights)

#Метод обратного распространения
for i in range(20000):
    input_layer = training_inputs
    outputs = sigmoid( np.dot(input_layer, synaptyc_weights) )

    err = training_outputs - outputs
    adjustments = np.dot( input_layer.T, err * (outputs * (1 - outputs)) )

    synaptyc_weights += adjustments

print('Веса после обучения:')
print(synaptyc_weights)

print('Результат после обучения:')
print(outputs)

#Тест
new_inputs = np.array([1,1,0]) #новая ситуация
output = sigmoid( np.dot(new_inputs, synaptyc_weights) )

print('Новая ситуация:')
print(output)

