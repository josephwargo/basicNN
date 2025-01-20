import numpy as np

# hidden layer with 100 
class neuralNet(object):

    def __init__(self):
        # weights and biases for hidden and output layer
        self.inputShape = 784
        self.hiddenLayerShape = 50
        self.outputShape = 10
        self.w1 = np.random.normal(0,.001, size=(self.inputShape,self.hiddenLayerShape))
        self.b1 = np.random.normal(0, .001)
        self.w2 = np.random.normal(0, .001, size=(self.hiddenLayerShape,self.outputShape))
        self.b2 = np.random.normal(0, .001)
        # hidden layer node
        self.n1 = None
        # output layer node
        self.n2 = None
        # current error
        self.error = None
        # learning rate
        self.learningRate = .025

    def sigmoid(self, x):
        sigmoid = 1 / (1 + np.exp(x))
        return sigmoid

    def forwardPass(self, input, output):
        # hidden layer
        self.n1 = self.sigmoid(np.dot(input, self.w1) + self.b1)
        # output layer
        self.n2 = self.sigmoid(np.dot(self.n1, self.w2) + self.b2)
        # error
        self.error = np.mean(.5*((output - self.n2)**2))
    
    def backwardPass(self, input, output):
        """
        Gradient Notation:
        C = cost function
        H = sigmoid activation
        Z = Wx + b
        W = weights
        B = bias
        X = input from previous layer
        """
        # output layer
        dCdH = -(output - self.n2) # cost function WRT the output of the NN (activation)
        dHdZ = self.n2*(1-self.n2) # activation WRT output node value
        n2LocalError = dCdH * dHdZ # dCdZ - cost function WRT output node value
        dCdW2 = np.outer(self.n1, n2LocalError) / len(self.n1) # np.outer(self.n1, n2LocalError) cost function WRT input weights - value used to update weights
        dCdB2 = np.sum(n2LocalError)
        dCdH = np.dot(self.w2, n2LocalError) # cost function WRT input node value - starting cost for next layer of backprop
        # hidden layer
        dHdZ = self.n1*(1-self.n1)#.reshape(self.hiddenLayerShape,1) # activation WRT output node value
        n1LocalError = dCdH * dHdZ # layer cost WRT output node value
        dCdW1 = np.outer(input, n1LocalError) / len(input) # cost function WRT input weights - value used to update weights
        dCdB1 = np.sum(n1LocalError)
        # updates
        self.w2 += -self.learningRate*dCdW2
        self.b2 += -self.learningRate*dCdB2
        self.w1 += -self.learningRate*dCdW1
        self.b1 += -self.learningRate*dCdB1
    
    def query(self, x):
        node1 = self.sigmoid(np.dot(x, self.w1) + self.b1)
        yPred = self.sigmoid(np.dot(node1, self.w2) + self.b2)
        return yPred

    def trainModel(self, input, output):
        for row in range(len(input)):
            self.forwardPass(input[row], output[row])
            self.backwardPass(input[row], output[row])