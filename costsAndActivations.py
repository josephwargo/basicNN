import numpy as np

# functions to compute costs and gradients of costs
def MSE(y, y_pred):
    return np.mean(.5*((y-y_pred)**2))
def MSEgradient(y, y_pred):
    return -(y-y_pred)

# no gradient function because this will only be paired with softmax, and they have a joint gradient function
def crossEntropyLoss(y, y_pred):
    y_pred = np.clip(y_pred, 10e-8, (1-(10e-8)))
    return -np.sum(y*np.log(y_pred))

# functions to compute activation and gradient of activations
def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))
def sigmoidGradient(y):
    return y*(1-y)

def relu(x):
    return np.maximum(0, x)
def reluGradient(y):
    return (y>0)*1

def tanH(x):
    numerator = np.exp(x) - np.exp(-x)
    denominator = np.exp(x) + np.exp(-x)
    return numerator/denominator
def tanHGradient(y):
    return 1 - y**2

# no gradient function because this will only be paired with Cross Entropy Loss, and they have a joint gradient function
def softmax(x):
    if len(x.shape)>1:
        normalization = np.max(x, axis=1, keepdims=True)
        numerator = np.exp(x - normalization)
        denominator = np.sum(numerator, axis=1, keepdims=True)
    else:
        normalization = np.max(x)
        numerator = np.exp(x - normalization)
        denominator = np.sum(numerator)
    return numerator / (denominator+10e-8)

# special gradient for softmax & cross entropy loss
def dCdZ(y, y_pred):
    return y_pred-y