# header files needed
import numpy as np

### loss functions
# function for calculating mean square error
def mean_square_loss(y_original, y_pred):
    return np.mean((y_pred - y_original) ** 2)

# function to calculate accuracy
def accuracy(y_original, y_pred):
    return np.mean(y_original == y_pred)


### activation functions and their derivatives
# sigmoid
def sigmoid(input):
    return (1 / (1 + np.exp(-input)))

# derivative of sigmoid
def sigmoid_derivative(input):
    return (sigmoid(input) * (1.0 - sigmoid(input)))

# relu
def relu(input):
    return np.maximum(0, input)

# derivative of relu
def relu_derivative(input):
    input[input <= 0] = 0
    input[input > 0] = 1
    return input

# leaky relu
def leaky_relu(input):
    return np.maximum(0.03 * input, input)

# derivative of leaky relu
def leaky_relu_derivative(input):
    input[input <= 0] = 0.03
    input[input > 0] = 1
    return input

# tanh function
def tanh(input):
    return ((np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input)))

# derivative of tanh
def tanh_derivative(input):
    return (1.0 - (tanh(input) * tanh(input)))
