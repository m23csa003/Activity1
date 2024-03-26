import matplotlib.pyplot as plt
import numpy as np

# plot Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# plot ReLU function
def relu(x):
    return np.maximum(0, x)

# plot Leaky ReLU function
def leaky_relu(x):
    return np.maximum(0.1*x, x)

# plot tanh function
def tanh(x):
    return np.tanh(x)


x = np.linspace(-10, 10, 100)

y = sigmoid(x)
plt.plot(x, y)
plt.title('Sigmoid Function')
plt.show()

y = relu(x)
plt.plot(x, y)
plt.title('ReLU Function')
plt.show()

y = leaky_relu(x)
plt.plot(x, y)
plt.title('Leaky ReLU Function')
plt.show()

y = tanh(x)
plt.plot(x, y)
plt.title('tanh Function')
plt.show()

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

for i in random_values:
    print(f'sigmoid of {i} = {sigmoid(i)}')
    print(f'relu of {i} = {relu(i)}')
    print(f'leaky_relu of {i} = {leaky_relu(i)}')
    print(f'tanh of {i} = {tanh(i)}')
    print('-----------------')

# This is a bug-fix

