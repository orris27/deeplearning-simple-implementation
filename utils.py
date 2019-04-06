import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x)) # Same result as np.exp(x) while avoiding overflow
    return e_x / e_x.sum(axis=0)

