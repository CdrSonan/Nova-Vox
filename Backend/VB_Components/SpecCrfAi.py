from typing import OrderedDict
import numpy as np
import math
import torch
import torch.nn as nn
import global_consts

import matplotlib.pyplot as plt

class SpecCrfAi(nn.Module):
    """class for generating crossphades between the spectra of different phonemes using AI.
    
    Attributes:
        layer1-4, ReLu1-4: FC and Nonlinear layers of the NN.
        
        learningRate: Learning Rate of the NN
        
        optimizer: Optimization algorithm to use during training. Changes not advised.
        
        criterion: Loss criterion to be used during AI training. Changes not advised.
        
        epoch: training epoch counter displayed in Metadata panels
        
        loss: Torch.Loss object holding Loss data from AI training
        
    Methods:
        __init__: Constructor initialising NN layers and prerequisite properties
        
        forward: Forward NN pass with unprocessed in-and outputs
        
        processData: forward NN pass with data pre-and postprocessing as expected by other classes
        
        train: NN training with forward and backward passes, Loss criterion and optimizer runs based on a dataset of spectral transition samples
        
        dataLoader: helper method for shuffled data loading from an arbitrary dataset
        
        getState: returns the state of the NN, its optimizer and their prerequisites in a Dictionary
        
    The structure of the NN is a forward-feed fully connected NN with ReLU nonlinear activation functions.
    It is designed to process non-negative data. Negative data can still be processed, but may negatively impact performance.
    The size of the NN layers is set to match the batch size and tick rate of the rest of the engine.
    Since network performance deteriorates with skewed data, it internally passes the input through a square root function and squares the output."""
        
        
    def __init__(self, device = None, learningRate=5e-5, hiddenLayerCount = 3):
        """Constructor initialising NN layers and prerequisite attributes.
        
        Arguments:
            learningRate: desired learning rate of the NN as float. supports scientific format.
            
        Returns:
            None"""
            
            
        super(SpecCrfAi, self).__init__()
        self.convolution = nn.Conv1d(5, 5, 1, device = device)
        self.layerStart1 = torch.nn.Linear(5 * global_consts.halfTripleBatchSize + 5, 5 * global_consts.halfTripleBatchSize + 5, device = device)
        self.ReLuStart1 = nn.ReLU()
        self.layerStart2 = torch.nn.Linear(5 * global_consts.halfTripleBatchSize + 5, 4 * global_consts.halfTripleBatchSize, device = device)
        self.ReLuStart2 = nn.ReLU()
        hiddenLayerDict = OrderedDict([])
        for i in range(hiddenLayerCount):
            hiddenLayerDict["layer" + str(i)] = torch.nn.Linear(4 * global_consts.halfTripleBatchSize, 4 * global_consts.halfTripleBatchSize, device = device)
            hiddenLayerDict["ReLu" + str(i)] = nn.ReLU()
        self.hiddenLayers = nn.Sequential(hiddenLayerDict)
        self.layerEnd1 = torch.nn.Linear(4 * global_consts.halfTripleBatchSize, 2 * global_consts.halfTripleBatchSize, device = device)
        self.ReLuEnd1 = nn.ReLU()
        self.layerEnd2 = torch.nn.Linear(2 * global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = device)
        self.ReLuEnd2 = nn.ReLU()
        self.threshold = torch.nn.Threshold(0.001, 0.001)

        self.device = device
        
        self.hiddenLayerCount = hiddenLayerCount
        self.learningRate = learningRate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=0.)
        self.criterion = nn.L1Loss()
        #self.criterion = RelLoss()
        self.epoch = 0
        self.sampleCount = 0
        self.loss = None
        
    def forward(self, spectrum1, spectrum2, spectrum3, spectrum4, factor):
        """Forward NN pass with unprocessed in-and outputs.
        
        Arguments:
            spectrum1, spectrum2: The two spectrum Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object representing the NN output
            
        When performing forward NN runs, it is strongly recommended to use processData() instead of this method."""
        
        spectrum1 = torch.unsqueeze(spectrum1, 1)
        spectrum2 = torch.unsqueeze(spectrum2, 1)
        spectrum3 = torch.unsqueeze(spectrum3, 1)
        spectrum4 = torch.unsqueeze(spectrum4, 1)
        spectra = torch.cat((spectrum1, spectrum2, spectrum3, spectrum4), dim = 1)
        limit = torch.max(spectra, dim = 1)[0]
        #fac = torch.tensor([factor], device = self.device)
        #x = torch.cat((torch.reshape(spectra, (-1,)), fac), dim = 0)
        fac = torch.full((global_consts.halfTripleBatchSize + 1, 1), factor, device = self.device)
        x = torch.cat((spectra, fac), dim = 1)
        x = x.float()
        x = torch.unsqueeze(torch.transpose(x, 0, 1), 0)
        x = self.convolution(x)
        x = torch.reshape(x, (-1,))
        x = self.layerStart1(x)
        x = self.ReLuStart1(x)
        x = self.layerStart2(x)
        x = self.ReLuStart2(x)
        x = self.hiddenLayers(x)
        x = self.layerEnd1(x)
        x = self.ReLuEnd1(x)
        x = self.layerEnd2(x)
        x = self.ReLuEnd2(x)
        x = torch.minimum(x, limit)

        spectralFilterWidth = 4 * global_consts.filterTEEMult
        x = torch.fft.rfft(x, dim = 0)
        cutoffWindow = torch.zeros_like(x)
        cutoffWindow[0:int(spectralFilterWidth / 2)] = 1.
        cutoffWindow[int(spectralFilterWidth / 2):spectralFilterWidth] = torch.linspace(1, 0, spectralFilterWidth - int(spectralFilterWidth / 2))
        x = torch.fft.irfft(cutoffWindow * x, dim = 0, n = global_consts.halfTripleBatchSize + 1)
        x = self.threshold(x)

        """plt.plot(spectrum1)
        plt.plot(spectrum2)
        plt.plot(spectrum3)
        plt.plot(spectrum4)
        plt.plot(limit)
        plt.show()"""
        
        return x
    
    def processData(self, spectrum1, spectrum2, spectrum3, spectrum4, factor):
        """forward NN pass with data pre-and postprocessing as expected by other classes
        
        Arguments:
            spectrum1, spectrum2: The two spectrum Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object containing the interpolated audio spectrum
            
            
        Other than when using forward() directly, this method sets the NN to evaluation mode and applies a square root function to the input and squares the output to
        improve overall performance, especially on the skewed datasets of vocal spectra."""
        
        
        self.eval()
        output = torch.square(torch.squeeze(self(torch.sqrt(spectrum1), torch.sqrt(spectrum2), torch.sqrt(spectrum3), torch.sqrt(spectrum4), factor)))
        return output
    
    def train(self, indata, epochs=1):
        """NN training with forward and backward passes, Loss criterion and optimizer runs based on a dataset of spectral transition samples.
        
        Arguments:
            indata: Tensor, List or other Iterable containing sets of stft-like spectra. Each element should represent a phoneme transition.
            
            epochs: number of epochs to use for training as Integer.
            
        Returns:
            None
            
        Like processData(), train() also takes the square root of the input internally before using the data for inference."""
        
        
        if indata != False:
            if (self.epoch == 0) or self.epoch == epochs:
                self.epoch = epochs
            else:
                self.epoch = None
            for epoch in range(epochs):
                for data in self.dataLoader(indata):
                    print('epoch [{}/{}], switching to next sample'.format(epoch + 1, epochs))
                    #data = torch.sqrt(data.to(device = self.device))
                    data = data.to(device = self.device)
                    data = torch.squeeze(data)
                    spectrum1 = data[0]
                    spectrum2 = data[1]
                    spectrum3 = data[-2]
                    spectrum4 = data[-1]
                    
                    length = data.size()[0]
                    filterWidth = math.ceil(length / 5)
                    threshold = torch.nn.Threshold(0.001, 0.001)
                    data = torch.fft.rfft(data, dim = 0)
                    cutoffWindow = torch.zeros(data.size()[0])
                    cutoffWindow[0:filterWidth] = 1.
                    cutoffWindow[filterWidth] = 0.5
                    data = threshold(torch.fft.irfft(torch.unsqueeze(cutoffWindow, 1) * data, dim = 0, n = length))
                    
                    indexList = np.arange(0, data.size()[0], 1)
                    np.random.shuffle(indexList)
                    for i in indexList:
                        factor = i / float(data.size()[0])
                        spectrumTarget = data[i]
                        output = torch.squeeze(self(spectrum1, spectrum2, spectrum3, spectrum4, factor))
                        loss = self.criterion(output, spectrumTarget)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        print('epoch [{}/{}], sub-sample index {}, loss:{:.4f}'.format(epoch + 1, epochs, i, loss.data))
            self.sampleCount += len(indata)
            self.loss = loss
            
    def dataLoader(self, data):
        """helper method for shuffled data loading from an arbitrary dataset.
        
        Arguments:
            data: Tensor, List or Iterable representing a dataset with several elements
            
        Returns:
            Iterable representing the same dataset, with the order of its elements shuffled"""
        
        
        return torch.utils.data.DataLoader(dataset=data, shuffle=True)
    
    def getState(self):
        """returns the state of the NN, its optimizer and their prerequisites as well as its epoch attribute in a Dictionary.
        
        Arguments:
            None
            
        Returns:
            Dictionary containing the NN's epoch attribute (epoch), weights (state dict), optimizer state (optimizer state dict) and loss object (loss)"""
            
            
        AiState = {'epoch': self.epoch,
                 'model_state_dict': self.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'loss': self.loss,
                 'sampleCount': self.sampleCount
                 }
        return AiState

class LiteSpecCrfAi(nn.Module):
    """A stripped down version of SpecCrfAi only holding the data required for synthesis.
    
    Attributes:
        layer1-4, ReLu1-4: FC and Nonlinear layers of the NN.
        
        epoch: training epoch counter displayed in Metadata panels
        
    Methods:
        __init__: Constructor initialising NN layers and prerequisite properties
        
        forward: Forward NN pass with unprocessed in-and outputs
        
        processData: forward NN pass with data pre-and postprocessing as expected by other classes
        
        getState: returns the state of the NN and its epoch attribute in a Dictionary
        
    This version of the AI can only run data through the NN forward, backpropagation and, by extension, training, are not possible."""

    def __init__(self, specCrfAi = None, device = None):
        """Constructor initialising NN layers and other attributes based on SpecCrfAi base object.
        
        Arguments:
            specCrfAi: SpecCrfAi base object
            
        Returns:
            None"""
            
            
        super(LiteSpecCrfAi, self).__init__()
        
        if specCrfAi == None:
            hiddenLayerCount = 3
        else:
            hiddenLayerCount = 3#specCrfAi.hiddenLayerCount

        self.convolution = nn.Conv1d(5, 5, 1, device = device)
        self.layerStart1 = torch.nn.Linear(5 * global_consts.halfTripleBatchSize + 5, 5 * global_consts.halfTripleBatchSize + 5, device = device)
        self.ReLuStart1 = nn.ReLU()
        self.layerStart2 = torch.nn.Linear(5 * global_consts.halfTripleBatchSize + 5, 4 * global_consts.halfTripleBatchSize, device = device)
        self.ReLuStart2 = nn.ReLU()
        hiddenLayerDict = OrderedDict([])
        for i in range(hiddenLayerCount):
            hiddenLayerDict["layer" + str(i)] = torch.nn.Linear(4 * global_consts.halfTripleBatchSize, 4 * global_consts.halfTripleBatchSize, device = device)
            hiddenLayerDict["ReLu" + str(i)] = nn.ReLU()
        self.hiddenLayers = nn.Sequential(hiddenLayerDict)
        self.layerEnd1 = torch.nn.Linear(4 * global_consts.halfTripleBatchSize, 2 * global_consts.halfTripleBatchSize, device = device)
        self.ReLuEnd1 = nn.ReLU()
        self.layerEnd2 = torch.nn.Linear(2 * global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = device)
        self.ReLuEnd2 = nn.ReLU()
        self.threshold = torch.nn.Threshold(0.001, 0.001)
        
        self.device = device

        if specCrfAi == None:
            self.epoch = 0
            self.sampleCount = 0
        else:
            self.epoch = specCrfAi.getState()['epoch']
            self.sampleCount = specCrfAi.getState()['sampleCount']
            self.load_state_dict(specCrfAi.getState()['model_state_dict'])
            self.eval()
        
    def forward(self, spectrum1, spectrum2, spectrum3, spectrum4, factor):
        """Forward NN pass with unprocessed in-and outputs.
        
        Arguments:
            spectrum1, spectrum2: The two spectrum Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object representing the NN output
            
        When performing forward NN runs, it is strongly recommended to use processData() instead of this method."""
        
        
        spectrum1 = torch.unsqueeze(spectrum1, 1)
        spectrum2 = torch.unsqueeze(spectrum2, 1)
        spectrum3 = torch.unsqueeze(spectrum3, 1)
        spectrum4 = torch.unsqueeze(spectrum4, 1)
        spectra = torch.cat((spectrum1, spectrum2, spectrum3, spectrum4), dim = 1)
        limit = torch.max(spectra, dim = 1)[0]
        #fac = torch.tensor([factor], device = self.device)
        #x = torch.cat((torch.reshape(spectra, (-1,)), fac), dim = 0)
        fac = torch.full((global_consts.halfTripleBatchSize + 1, 1), factor, device = self.device)
        x = torch.cat((spectra, fac), dim = 1)
        x = x.float()
        x = torch.unsqueeze(torch.transpose(x, 0, 1), 0)
        x = self.convolution(x)
        x = torch.reshape(x, (-1,))
        x = self.layerStart1(x)
        x = self.ReLuStart1(x)
        x = self.layerStart2(x)
        x = self.ReLuStart2(x)
        x = self.hiddenLayers(x)
        x = self.layerEnd1(x)
        x = self.ReLuEnd1(x)
        x = self.layerEnd2(x)
        x = self.ReLuEnd2(x)
        x = torch.minimum(x, limit)

        spectralFilterWidth = 4 * global_consts.filterTEEMult
        x = torch.fft.rfft(x, dim = 0)
        cutoffWindow = torch.zeros_like(x)
        cutoffWindow[0:int(spectralFilterWidth / 2)] = 1.
        cutoffWindow[int(spectralFilterWidth / 2):spectralFilterWidth] = torch.linspace(1, 0, spectralFilterWidth - int(spectralFilterWidth / 2))
        x = torch.fft.irfft(cutoffWindow * x, dim = 0, n = global_consts.halfTripleBatchSize + 1)
        x = self.threshold(x)

        """plt.plot(spectrum1)
        plt.plot(spectrum2)
        plt.plot(spectrum3)
        plt.plot(spectrum4)
        plt.plot(x.detach())
        plt.show()"""
        return x
    
    def processData(self, spectrum1, spectrum2, spectrum3, spectrum4, factor):
        """forward NN pass with data pre-and postprocessing as expected by other classes
        
        Arguments:
            spectrum1, spectrum2: The two spectrum Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object containing the interpolated audio spectrum
            
            
        Other than when using forward() directly, this method sets the NN to evaluation mode and applies a square root function to the input and squares the output to
        improve overall performance, especially on the skewed datasets of vocal spectra."""
        
        
        self.eval()
        output = torch.square(torch.squeeze(self(torch.sqrt(spectrum1), torch.sqrt(spectrum2), torch.sqrt(spectrum3), torch.sqrt(spectrum4), factor)))
        return output
    
    def getState(self):
        """returns the state of the NN and its epoch attribute in a Dictionary.
        
        Arguments:
            None
            
        Returns:
            Dictionary containing the NN's weights (state dict) and its epoch attribute (epoch)"""
            
            
        AiState = {'epoch': self.epoch,
                 'sampleCount': self.sampleCount,
                 'model_state_dict': self.state_dict()
                 }
        return AiState

class RelLoss(nn.Module):
    """function for calculating relative loss values between target and actual Tensor objects. Designed to be used with AI optimizers.
    
    Attributes:
        None
        
    Methods:
        __init__: basic class constructor
        
        forward: calculates relative loss based on input and target tensors after successful initialisation."""
    
    
    def __init__(self, weight=None, size_average=True):
        """basic class constructor.
        
        Arguments:
            weight: required by PyTorch in some situations. Unused.
            
            size_average: required by PyTorch in some situations. Unused.
            
        Returns:
            None"""
        
        
        super(RelLoss, self).__init__()
 
    def forward(self, inputs, targets):  
        """calculates relative loss based on input and target tensors after successful initialisation.
        
        Arguments:
            inputs: AI-generated input Tensor
            
            targets: target Tensor
            
        Returns:
            Relative error value calculated from the difference between input and target Tensor as Float"""
        
        
        #inputs = inputs.view(-1)
        #targets = targets.view(-1)
        differences = torch.abs(inputs - targets)
        refs = torch.abs(targets)
        out = (differences / refs).sum() / inputs.size()[0]
        return out