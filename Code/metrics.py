# header files
import numpy as np

# function for calculating mean square error
def mean_square_loss(y_original, y_pred):
    return np.mean((y_pred - y_original) ** 2)
