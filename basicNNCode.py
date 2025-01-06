import numpy as np

# individual neuron
class neuronLayer(object):
    def __init__(self, prevLayerShape, outputShape, adam):
        self.prevLayerShape = prevLayerShape
        self.outputShape = outputShape
        xavier = np.sqrt(2/(self.prevLayerShape+self.outputShape))
        self.W = np.random.normal(0,xavier, size=(self.prevLayerShape,self.outputShape))
        self.b = np.zeros(shape=(outputShape))
        self.N = np.zeros(shape=(outputShape))
        # adam
        self.adam = adam
        if adam:
            # constants
            self.beta1 = .9
            self.beta1T = self.beta1
            self.beta2 = .999
            self.beta2T = self.beta2
            self.epsilon = 10e-8
            self.t = 1
            # arrays to store 
            self.mdW = np.zeros(shape=(self.prevLayerShape,self.outputShape))
            self.mdB = np.zeros(shape=(outputShape))
            self.vdW = np.zeros(shape=(self.prevLayerShape,self.outputShape))
            self.vdB = np.zeros(shape=(outputShape))
    
    def updateAdam(self, dCdW, dCdB):
        # momentum
        self.mdW = self.beta1*self.mdW + (1-self.beta1)*dCdW
        self.mdB = self.beta1*self.mdB + (1-self.beta1)*dCdB
        # RMSprop
        self.vdW = self.beta2*self.vdW + (1-self.beta2)*(dCdW**2)
        self.vdB = self.beta2*self.vdB + (1-self.beta2)*(dCdB**2)
        # bias correction
        mdWHat = self.mdW / (1-self.beta1**self.t)
        mdBHat = self.mdB / (1-self.beta1**self.t)
        vdWHat = self.vdW / (1-self.beta2**self.t)
        vdBHat = self.vdB / (1-self.beta2**self.t)
        # adam
        newdCdW = mdWHat / (np.sqrt(vdWHat)+self.epsilon)
        newdCdB = mdBHat / (np.sqrt(vdBHat)+self.epsilon)
        self.t+=1
        return newdCdW, newdCdB


# entire net
class neuralNet(object):
    def __init__(self, inputShape, outputShape, outputActivation, hiddenLayerShapes, 
                 hiddenLayerActivations, lossFunction='MSE', learningRate=.001, epochs=1, batchSize=1,
                 adam=False, debug=False):
        # errors
        if len(hiddenLayerShapes)!=len(hiddenLayerActivations):
            raise Exception('Length of hiddenLayerShapes does not match length of hiddenLayerActivations')
        if (lossFunction!='crossEntropyLoss') & (outputActivation!='softmax'):
            raise Exception('A cost function of Cross Entropy Loss and an output layer activation of Softmax must be paired with each other')
        if adam & (learningRate>.01):
            print('Warning: Learning rate may be too high for ADAM optimizer to function properly')
        # variables straight from initialization
        self.batchSize = batchSize
        self.epochs = epochs
        self.debug = debug
        self.adam = adam
        self.learningRate = learningRate
        self.lossFunction = lossFunction
        self.activations = hiddenLayerActivations + [outputActivation]
        self.reverseActivations = self.activations.copy()
        self.reverseActivations.reverse()
        
        # initializing hidden layers and adding to dictionary of all layers
        hiddenLayer1 = neuronLayer(inputShape, hiddenLayerShapes[0], adam)
        self.allLayers = {'hiddenLayer1': hiddenLayer1}
        if len(hiddenLayerShapes) > 1:
            for count, value in enumerate(hiddenLayerShapes):
                layerNum = count+2
                if count<len(hiddenLayerShapes)-1:
                    self.allLayers["hiddenLayer{}".format(layerNum)] = neuronLayer(value, hiddenLayerShapes[count+1], adam)
       
        # adding output layer to dictionary of all layers
        outputLayer = neuronLayer(hiddenLayerShapes[-1], outputShape, adam)
        self.allLayers['outputLayer'] = outputLayer

        # to track error
        self.error = None


    # functions to compute costs and gradients of costs
    def MSE(self, y, y_pred):
        return np.mean(.5*((y-y_pred)**2))
    def MSEgradient(self, y, y_pred):
        return -(y-y_pred)
    
    # no gradient function because this will only be paired with softmax, and they have a joint gradient function
    def crossEntropyLoss(self, y, y_pred):
        y_pred = np.clip(y_pred, 10e-8, (1-(10e-8)))
        return -np.sum(y*np.log(y_pred))

    # functions to compute activation and gradient of activations
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoidGradient(self, y):
        return y*(1-y)

    def relu(self, x):
        return np.maximum(0, x)
    def reluGradient(self, y):
        return (y>0)*1
    
    # no gradient function because this will only be paired with Cross Entropy Loss, and they have a joint gradient function
    def softmax(self, x):
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
    def dCdZ(self, y, y_pred):
        return y_pred-y

    # training methods
    def forwardPass(self, input, output):
        # cycling through each layer
        for count, layerName in enumerate(self.allLayers.keys()):
            layer = self.allLayers[layerName]
            if self.batchSize>1:
                bias = np.tile(layer.b, (self.batchSize, 1))
            else:
                bias = layer.b
            if self.debug:
                print('Layer: ' + layerName)
                print('Input shape: ' + str(input.shape))
                print('Weight shape: ' + str(layer.W.shape))
                print('Bias shape: ' + str(layer.b.shape))

            # calculating dot product + activation
            z = np.dot(input, layer.W) + bias
            if self.activations[count] == 'relu':
                layer.N = self.relu(z)
            elif self.activations[count]=='sigmoid':
                layer.N = self.sigmoid(z)
            elif (self.activations[count]=='softmax'):
                layer.N = self.softmax(z)
            else:
                raise Exception('Unknown activation function')
            if self.debug:
                print('Output shape: ' + str(layer.N.shape))
                print()
            input = layer.N
        # storing final error
        if self.lossFunction == 'MSE':
            self.error = self.MSE(output, layer.N)
        elif self.lossFunction == 'crossEntropyLoss':
            self.error = self.crossEntropyLoss(output, layer.N)
        else:
            raise Exception('Unknown cost function')
    
    # rename and fold into the top
    def costGradient(self, output):
        # gradient of cost function WRT the output of the NN (activation)
        if self.lossFunction == 'MSE':
            dCdH = -(output - self.allLayers['outputLayer'].N)
        elif self.lossFunction == 'crossEntropyLoss':
            dCdH = 0
        return dCdH

    def backwardPass(self, input, output):
        """
        Gradient Notation:
        C = cost function
        H = activation
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
                localError = dCdH * dHdZ
            elif self.reverseActivations[count] == 'sigmoid':
                dHdZ = self.sigmoidGradient(currLayer.N)
                localError = dCdH * dHdZ
            # special case of softmax & cross entropy loss
            elif self.reverseActivations[count] == 'softmax':
                localError = self.dCdZ(output, currLayer.N)
            
            # weight+bias updates, and the dCdH for the next round of backpropogation
            if layerName != 'hiddenLayer1':
                prevLayer = self.allLayers[reverseKeys[count+1]]
                dCdW = np.dot(prevLayer.N.T, localError)
                dCdB = np.sum(localError, axis=0, keepdims=True) / self.batchSize # cost function WRT input biases - value used to update bias
                if currLayer.adam:
                    dCdW, dCdB = currLayer.updateAdam(dCdW, dCdB)
                dCdH = np.dot(localError, currLayer.W.T)
            # weight and bias updates for when we hit the first hidden layer
            else:
                dCdW = np.dot(input.T, localError) / self.batchSize
                dCdB = np.sum(localError, axis=0, keepdims=True) / self.batchSize # cost function WRT input biases - value used to update bias
                if currLayer.adam:
                    dCdW, dCdB = currLayer.updateAdam(dCdW, dCdB)
            weightUpdates.append(dCdW)
            biasUpdates.append(dCdB)
        # updating weights and biases
        for count, layerName in enumerate(reverseKeys):
            layer = self.allLayers[layerName]
            layer.W += -self.learningRate*weightUpdates[count]
            layer.b += -self.learningRate*(biasUpdates[count].reshape(-1,))
    
    def trainModel(self, input, output):
        numSamples = input.shape[0]
        for epoch in range(self.epochs):
            indices = np.random.permutation(numSamples)
            inputShuffled = input[indices]
            outputShuffled = output[indices]
            for start in range(0, numSamples, self.batchSize):
                end = min(start+self.batchSize, numSamples)
                self.forwardPass(inputShuffled[start:end], outputShuffled[start:end])
                self.backwardPass(inputShuffled[start:end], outputShuffled[start:end])

    # return predicted output for a given input
    def query(self, input):
        for count, layerName in enumerate(self.allLayers.keys()):
            layer = self.allLayers[layerName]
            if self.activations[count] == 'relu':
                input = self.relu(np.dot(input, layer.W) + layer.b)
            elif self.activations[count] == 'sigmoid':
                input = self.sigmoid(np.dot(input, layer.W) + layer.b)
            elif self.activations[count] == 'softmax':
                input = self.softmax(np.dot(input, layer.W) + layer.b)
        return input