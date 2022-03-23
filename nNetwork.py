import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



# Activation function
def relu(Z):
    return np.maximum(Z,0)

def softmax(Z):
    soft = np.exp(Z)/np.sum(np.exp(Z))

    return soft


def init_parmter():
    w1 = np.random(10,784) - 0.5
    b1 = np.random(10,1) - 0.5
    w2 = np.random(10,10) - 0.5
    b2 = np.random(10,1) - 0.5

    return w1,b1,w2,b2 

def forwardBropagation(w1,b1,w2,b2,X_input):
    Z1 = w1.dot(X_input)+b1
    A1 = relu(Z1)
    Z2 = w2.dot(A1)+b2
    A2 = softmax(Z2)

    return Z1 ,A1, Z2, A2


