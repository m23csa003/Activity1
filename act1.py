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

# This is a bug-fix
