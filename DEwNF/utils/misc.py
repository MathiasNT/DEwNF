import numpy as np


def circle_transform(z, max_val):
    x_val = np.cos(2*np.pi*z/max_val)
    y_val = np.sin(2*np.pi*z/max_val)
    return x_val, y_val