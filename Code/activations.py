# header files needed
import numpy as np

# sigmoid function
def sigmoid(input):
    return (1 / (1 + np.exp(-input)))

# relu function
def relu(input):
    return np.maximum(0, input)
