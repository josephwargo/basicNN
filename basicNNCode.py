import numpy as np

class neuronLayer(object):
    def __init__(self, prevLayerShape, outputShape):
        self.prevLayerShape = prevLayerShape
        self.outputShape = outputShape
        self.W = np.random.normal(0,.1, size=(self.prevLayerShape,self.outputShape))
        self.b = np.zeros(shape=(outputShape))
        self.N = np.zeros(shape=(outputShape))

class neuralNet(object):

    def __init__(self, inputShape, outputShape, outputActivation, hiddenLayerShapes, 
                 hiddenLayerActivations, learningRate=.001, debug=False):
        if len(hiddenLayerShapes)!=len(hiddenLayerActivations):
            raise Exception('Length of hiddenLayerShapes does not match length of hiddenLayerActivations')
        # variables straight from initialization
        self.debug=debug
        self.learningRate = learningRate
        self.activations = hiddenLayerActivations + [outputActivation]
        self.reverseActivations = self.activations.copy()
        self.reverseActivations.reverse()
        # initializing hidden layers and adding to dictionary of all layers
        hiddenLayer1 = neuronLayer(inputShape, hiddenLayerShapes[0])
        self.allLayers = {'hiddenLayer1': hiddenLayer1}
        if len(hiddenLayerShapes) > 1:
            for count, value in enumerate(hiddenLayerShapes):
                layerNum = count+2
                if count<len(hiddenLayerShapes)-1:
                    self.allLayers["hiddenLayer{}".format(layerNum)] = neuronLayer(value, hiddenLayerShapes[count+1])
                # elif count<len(hiddenLayerShapes)-1:
                #     self.allLayers["hiddenLayer{}".format(layerNum)] = neuronLayer(value, outputShape)
        # adding output layer to dictionary of all layers
        outputLayer = neuronLayer(hiddenLayerShapes[-1], outputShape)
        self.allLayers['outputLayer'] = outputLayer
        # to track error
        self.error = None

    # functions to compute activation and gradient of activations
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoidGradient(self, y):
        return y*(1-y)

    def relu(self, x):
        return np.maximum(0, x)
    
    def reluGradient(self, y):
        return (y>0)*1


    def forwardPass(self, input, output):
        # input = input.reshape(-1,1)
        for count, layerName in enumerate(self.allLayers.keys()):
            layer = self.allLayers[layerName]
            if self.debug:
                print('Layer: ' + layerName)
                print('Input shape: ' + str(input.shape))
                print('Weight shape: ' + str(layer.W.shape))
                print('Bias shape: ' + str(layer.b.shape))
                
            # print(np.dot(input, layer.W).shape)
            if self.activations[count] == 'relu':
                layer.N = self.relu(np.dot(input, layer.W) + layer.b)
            elif self.activations[count]=='sigmoid':
                layer.N = self.sigmoid(np.dot(input, layer.W) + layer.b)
            else:
                raise Exception('Unknown activation function')
            if self.debug:
                print('Output shape: ' + str(layer.N.shape))
                print()
            input = layer.N
        # error
        self.error = np.mean(.5*((output - layer.N)**2))
    
    def costGradient(self, output):
        dCdH = -(output - self.allLayers['outputLayer'].N) # cost function WRT the output of the NN (activation)
        return dCdH

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
        reverseKeys = list(self.allLayers.keys())
        reverseKeys.reverse()
        # lists to hold the update values for weights and biases
        weightUpdates = []
        biasUpdates = []
        # gradient of the cost function to start backpropogation
        dCdH = self.costGradient(output)
        # iterating through layers backwards for backpropogation
        for count, layerName in enumerate(reverseKeys):
            currLayer = self.allLayers[layerName]
            # activation WRT output node value
            if self.reverseActivations[count] == 'relu':
                dHdZ = self.reluGradient(currLayer.N)
            else:
                dHdZ = self.sigmoidGradient(currLayer.N)
            localError = dCdH * dHdZ
            # weight+bias updates, and the dCdH for the next round of backpropogation
            if layerName != 'hiddenLayer1':
                prevLayer = self.allLayers[reverseKeys[count+1]]
                dCdW = np.dot(prevLayer.N.reshape(-1,1), localError.reshape(1,-1)) #/ len(prevLayer.N) # np.outer(self.n1, n2LocalError) cost function WRT input weights - value used to update weights
                dCdB = np.sum(localError, axis=0, keepdims=True)
                dCdH = np.dot(currLayer.W, localError) # cost function WRT input node value - starting cost for next layer of backprop
            # weight and bias updates for when we hit the first hidden layer
            else:
                dCdW = np.dot(input.reshape(-1,1), localError.reshape(1,-1)) #/ len(input) # np.outer(self.n1, n2LocalError) cost function WRT input weights - value used to update weights
                dCdB = np.sum(localError, axis=0, keepdims=True)
            # print(dCdW)
            weightUpdates.append(dCdW)
            biasUpdates.append(dCdB)
        # updating weights and biases
        for count, layerName in enumerate(reverseKeys):
            layer = self.allLayers[layerName]
            layer.W += -self.learningRate*weightUpdates[count]
            layer.b += -self.learningRate*biasUpdates[count]
    
    def query(self, input):
        for count, layerName in enumerate(self.allLayers.keys()):
            layer = self.allLayers[layerName]
            if self.activations[count] == 'relu':
                input = self.relu(np.dot(input, layer.W) + layer.b)
            elif self.activations[count] == 'sigmoid':
                input = self.sigmoid(np.dot(input, layer.W) + layer.b)
        return input

    def trainModel(self, input, output):
        for row in range(len(input)):
            self.forwardPass(input[row], output[row])
            self.backwardPass(input[row], output[row])