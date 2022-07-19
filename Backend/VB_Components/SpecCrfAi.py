from typing import OrderedDict
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import global_consts
from Backend.Resampler.PhaseShift import phaseInterp

halfHarms = int(global_consts.nHarmonics / 2) + 1

class SpecCrfAi(nn.Module):
    """class for generating crossfades between the spectra of different phonemes using AI.
    
    Attributes:
        layerStart/End 1/2, ReLuStart/End 1/2: leading and trailing FC and Nonlinear layers of the NN.

        hiddenLayers: ordered dictionary containing all layers between the leading and trailing ones
        
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
    Since performance deteriorates with skewed data, it internally passes the input through a square root function and squares the output."""
        
        
    def __init__(self, device:torch.device = None, learningRate:float=5e-5, hiddenLayerCount:int = 3) -> None:
        """Constructor initialising NN layers and prerequisite attributes.
        
        Arguments:
            device: the device the AI is to be loaded on

            learningRate: desired learning rate of the NN as float. supports scientific format.
            
            hiddenLayerCount: number of hidden layers (between leading and trailing layers)

        Returns:
            None"""
            
            
        super(SpecCrfAi, self).__init__()
        self.convolution = nn.Conv1d(6, 6, 1, device = device)
        self.layerStart1 = torch.nn.Linear(6 * global_consts.halfTripleBatchSize + 6, 5 * global_consts.halfTripleBatchSize + 5, device = device)
        self.ReLuStart1 = nn.ReLU()
        self.layerStart2 = torch.nn.Linear(5 * global_consts.halfTripleBatchSize + 5, 4 * global_consts.halfTripleBatchSize, device = device)
        self.ReLuStart2 = nn.ReLU()
        self.harmConvolution = nn.Conv1d(6, 6, 1, device = device)
        self.harmLayerStart1 = torch.nn.Linear(6 * halfHarms, 5 * halfHarms, device = device)
        self.harmReLuStart1 = nn.ReLU()
        self.harmLayerStart2 = torch.nn.Linear(5 * halfHarms, 4 * halfHarms, device = device)
        self.harmReLuStart2 = nn.ReLU()
        hiddenLayerDict = OrderedDict([])
        for i in range(hiddenLayerCount):
            hiddenLayerDict["layer" + str(i)] = torch.nn.Linear(4 * (global_consts.halfTripleBatchSize + halfHarms), 4 * (global_consts.halfTripleBatchSize + halfHarms), device = device)
            hiddenLayerDict["ReLu" + str(i)] = nn.ReLU()
        self.hiddenLayers = nn.Sequential(hiddenLayerDict)
        self.layerEnd1 = torch.nn.Linear(4 * global_consts.halfTripleBatchSize, 2 * global_consts.halfTripleBatchSize, device = device)
        self.ReLuEnd1 = nn.ReLU()
        self.layerEnd2 = torch.nn.Linear(2 * global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = device)
        self.ReLuEnd2 = nn.ReLU()
        self.harmLayerEnd1 = torch.nn.Linear(4 * halfHarms, 2 * halfHarms, device = device)
        self.harmReLuEnd1 = nn.ReLU()
        self.harmLayerEnd2 = torch.nn.Linear(2 * halfHarms, halfHarms, device = device)
        self.harmReLuEnd2 = nn.ReLU()
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

        self.pred = SpecPredAI(device, learningRate)
        self.currPrediction = torch.zeros((1, 1, halfHarms + global_consts.halfTripleBatchSize + 1), device = device)
        
    def forward(self, specharm1:torch.Tensor, specharm2:torch.Tensor, specharm3:torch.Tensor, specharm4:torch.Tensor, factor:float) -> torch.Tensor:
        """Forward NN pass with unprocessed in-and outputs.
        
        Arguments:
            spectrum1, spectrum2: The two spectrum Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object representing the NN output
            
        When performing forward NN runs, it is strongly recommended to use processData() instead of this method."""

        phase1 = specharm1[halfHarms:2 * halfHarms]
        phase2 = specharm2[halfHarms:2 * halfHarms]
        phase3 = specharm3[halfHarms:2 * halfHarms]
        phase4 = specharm4[halfHarms:2 * halfHarms]
        spectrum1 = specharm1[2 * halfHarms:]
        spectrum2 = specharm2[2 * halfHarms:]
        spectrum3 = specharm3[2 * halfHarms:]
        spectrum4 = specharm4[2 * halfHarms:]
        spectrum1 = torch.unsqueeze(spectrum1, 1)
        spectrum2 = torch.unsqueeze(spectrum2, 1)
        spectrum3 = torch.unsqueeze(spectrum3, 1)
        spectrum4 = torch.unsqueeze(spectrum4, 1)
        spectra = torch.cat((spectrum1, spectrum2, spectrum3, spectrum4), dim = 1)
        harm1 = specharm1[:halfHarms]
        harm2 = specharm2[:halfHarms]
        harm3 = specharm3[:halfHarms]
        harm4 = specharm4[:halfHarms]
        harm1 = torch.unsqueeze(harm1, 1)
        harm2 = torch.unsqueeze(harm2, 1)
        harm3 = torch.unsqueeze(harm3, 1)
        harm4 = torch.unsqueeze(harm4, 1)
        harms = torch.cat((harm1, harm2, harm3, harm4), dim = 1)
        limit = torch.max(spectra, dim = 1)[0]
        fac = torch.full((global_consts.halfTripleBatchSize + 1, 1), factor, device = self.device)
        facHarm = torch.full((halfHarms, 1), factor, device = self.device)
        predSpectrum = self.currPrediction[:, :, halfHarms:]
        predSpectrum = torch.squeeze(predSpectrum)
        predSpectrum = torch.unsqueeze(predSpectrum, 1)
        predHarm = self.currPrediction[:, :, :halfHarms]
        predHarm = torch.squeeze(predHarm)
        predHarm = torch.unsqueeze(predHarm, 1)
        x = torch.cat((spectra, fac, predSpectrum), dim = 1)
        x = x.float()
        x = torch.unsqueeze(torch.transpose(x, 0, 1), 0)
        x = self.convolution(x)
        x = torch.reshape(x, (-1,))
        x = self.layerStart1(x)
        x = self.ReLuStart1(x)
        x = self.layerStart2(x)
        x = self.ReLuStart2(x)
        y = torch.cat((harms, facHarm, predHarm), dim = 1)
        y = y.float()
        y = torch.unsqueeze(torch.transpose(y, 0, 1), 0)
        y = self.harmConvolution(y)
        y = torch.reshape(y, (-1,))
        y = self.harmLayerStart1(y)
        y = self.harmReLuStart1(y)
        y = self.harmLayerStart2(y)
        y = self.harmReLuStart2(y)
        x = torch.cat((x, y), 0)
        x = self.hiddenLayers(x)
        y = x[:4 * halfHarms]
        x = x[4 * halfHarms:]
        x = self.layerEnd1(x)
        x = self.ReLuEnd1(x)
        x = self.layerEnd2(x)
        x = self.ReLuEnd2(x)
        y = self.harmLayerEnd1(y)
        y = self.harmReLuEnd1(y)
        y = self.harmLayerEnd2(y)
        y = self.harmReLuEnd2(y)
        x = torch.minimum(x, limit)

        spectralFilterWidth = 4 * global_consts.filterTEEMult
        x = torch.fft.rfft(x, dim = 0)
        cutoffWindow = torch.zeros_like(x)
        cutoffWindow[0:int(spectralFilterWidth / 2)] = 1.
        cutoffWindow[int(spectralFilterWidth / 2):spectralFilterWidth] = torch.linspace(1, 0, spectralFilterWidth - int(spectralFilterWidth / 2))
        x = torch.fft.irfft(cutoffWindow * x, dim = 0, n = global_consts.halfTripleBatchSize + 1)
        x = self.threshold(x)
        
        phases = phaseInterp(phaseInterp(phase1, phase2, 0.5), phaseInterp(phase3, phase4, 0.5), factor)

        result = torch.cat((y, phases, x), 0)
        self.currPrediction = self.stepSpecPred(result)
        return result
    
    def processData(self, specharm1:torch.Tensor, specharm2:torch.Tensor, specharm3:torch.Tensor, specharm4:torch.Tensor, factor:float) -> torch.Tensor:
        """forward NN pass with data pre-and postprocessing as expected by other classes
        
        Arguments:
            spectrum1, spectrum2: The two spectrum Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object containing the interpolated audio spectrum
            
            
        Other than when using forward() directly, this method sets the NN to evaluation mode and applies a square root function to the input and squares the output to
        improve overall performance, especially on the skewed datasets of vocal spectra."""
        
        
        self.eval()
        output = torch.square(torch.squeeze(self(torch.sqrt(specharm1), torch.sqrt(specharm2), torch.sqrt(specharm3), torch.sqrt(specharm4), factor)))
        return output
    
    def train(self, indata, epochs:int=1) -> None:
        """NN training with forward and backward passes, Loss criterion and optimizer runs based on a dataset of spectral transition samples.
        
        Arguments:
            indata: Tensor, List or other Iterable containing sets of stft-like spectra. Each element should represent a phoneme transition.
            
            epochs: number of epochs to use for training as Integer.
            
        Returns:
            None
            
        Like processData(), train() also takes the square root of the input internally before using the data for inference."""
        
        self.pred.train(indata, epochs)
        if indata != False:
            if (self.epoch == 0) or self.epoch == epochs:
                self.epoch = epochs
            else:
                self.epoch = None
            for epoch in range(epochs):
                for data in self.dataLoader(indata):
                    print('epoch [{}/{}], switching to next sample'.format(epoch + 1, epochs))
                    data = torch.sqrt(data.to(device = self.device))
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
                    self.resetSpecPred
                    for i in indexList:
                        factor = i / float(data.size()[0])
                        spectrumTarget = data[i]
                        output = torch.squeeze(self.forward(spectrum1, spectrum2, spectrum3, spectrum4, factor))
                        output = torch.cat((output[:halfHarms], output[2 * halfHarms:]), 0)
                        spectrumTarget = torch.cat((spectrumTarget[:halfHarms], spectrumTarget[2 * halfHarms:]), 0)
                        loss = self.criterion(output, spectrumTarget)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        print('epoch [{}/{}], sub-sample index {}, loss:{:.4f}'.format(epoch + 1, epochs, i, loss.data))
            self.sampleCount += len(indata)
            self.loss = loss

    def stepSpecPred(self, specharm:torch.Tensor) -> torch.Tensor:
        self.pred.state[0].detach()
        self.pred.state[1].detach()
        return self.pred.processData(specharm)

    def resetSpecPred(self) -> None:
        self.pred.resetState()
        self.currPrediction = torch.zeros((1, 1, halfHarms + global_consts.halfTripleBatchSize + 1), device = self.currPrediction.device)
            
    def dataLoader(self, data) -> DataLoader:
        """helper method for shuffled data loading from an arbitrary dataset.
        
        Arguments:
            data: Tensor, List or Iterable representing a dataset with several elements
            
        Returns:
            Iterable representing the same dataset, with the order of its elements shuffled"""
        
        
        return DataLoader(dataset=data, shuffle=True)
    
    def getState(self) -> dict:
        """returns the state of the NN, its optimizer and their prerequisites as well as its epoch attribute in a Dictionary.
        
        Arguments:
            None
            
        Returns:
            Dictionary containing the NN's epoch attribute (epoch), weights (state dict), optimizer state (optimizer state dict) and loss object (loss)"""
            
            
        AiState = {'epoch': self.epoch,
                 'model_state_dict': self.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'pred_model_state_dict': self.pred.state_dict(),
                 'pred_optimizer_state_dict': self.pred.optimizer.state_dict(),
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

    def __init__(self, specCrfAi:SpecCrfAi = None, device:torch.device = None) -> None:
        """Constructor initialising NN layers and other attributes based on SpecCrfAi base object.
        
        Arguments:
            specCrfAi: SpecCrfAi base object
            
        Returns:
            None"""
            
            
        super(LiteSpecCrfAi, self).__init__()
        
        if specCrfAi == None:
            hiddenLayerCount = 3
        else:
            hiddenLayerCount = specCrfAi.hiddenLayerCount

        self.convolution = nn.Conv1d(6, 6, 1, device = device)
        self.layerStart1 = torch.nn.Linear(5 * global_consts.halfTripleBatchSize + 5, 5 * global_consts.halfTripleBatchSize + 5, device = device)
        self.ReLuStart1 = nn.ReLU()
        self.layerStart2 = torch.nn.Linear(5 * global_consts.halfTripleBatchSize + 5, 4 * global_consts.halfTripleBatchSize, device = device)
        self.ReLuStart2 = nn.ReLU()
        self.harmConvolution = nn.Conv1d(6, 6, 1, device = device)
        self.harmLayerStart1 = torch.nn.Linear(6 * halfHarms, 5 * halfHarms, device = device)
        self.harmReLuStart1 = nn.ReLU()
        self.harmLayerStart2 = torch.nn.Linear(5 * halfHarms, 4 * halfHarms, device = device)
        self.harmReLuStart2 = nn.ReLU()
        hiddenLayerDict = OrderedDict([])
        for i in range(hiddenLayerCount):
            hiddenLayerDict["layer" + str(i)] = torch.nn.Linear(4 * (global_consts.halfTripleBatchSize + halfHarms), 4 * (global_consts.halfTripleBatchSize + halfHarms), device = device)
            hiddenLayerDict["ReLu" + str(i)] = nn.ReLU()
        self.hiddenLayers = nn.Sequential(hiddenLayerDict)
        self.layerEnd1 = torch.nn.Linear(4 * global_consts.halfTripleBatchSize, 2 * global_consts.halfTripleBatchSize, device = device)
        self.ReLuEnd1 = nn.ReLU()
        self.layerEnd2 = torch.nn.Linear(2 * global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = device)
        self.ReLuEnd2 = nn.ReLU()
        self.harmLayerEnd1 = torch.nn.Linear(4 * halfHarms, 2 * halfHarms, device = device)
        self.harmReLuEnd1 = nn.ReLU()
        self.harmLayerEnd2 = torch.nn.Linear(2 * halfHarms, halfHarms, device = device)
        self.harmReLuEnd2 = nn.ReLU()
        self.threshold = torch.nn.Threshold(0.001, 0.001)
        
        self.device = device

        self.pred = SpecPredAI(device)

        if specCrfAi == None:
            self.epoch = 0
            self.sampleCount = 0
        else:
            self.epoch = specCrfAi.getState()['epoch']
            self.sampleCount = specCrfAi.getState()['sampleCount']
            self.load_state_dict(specCrfAi.getState()['model_state_dict'])
            self.pred.load_state_dict(specCrfAi.getState()['pred_model_state_dict'])
            self.eval()
        
    def forward(self, specharm1:torch.Tensor, specharm2:torch.Tensor, specharm3:torch.Tensor, specharm4:torch.Tensor, factor:float) -> torch.Tensor:
        """Forward NN pass with unprocessed in-and outputs.
        
        Arguments:
            spectrum1, spectrum2: The two spectrum Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object representing the NN output
            
        When performing forward NN runs, it is strongly recommended to use processData() instead of this method."""

        phase1 = specharm1[halfHarms:2 * halfHarms]
        phase2 = specharm2[halfHarms:2 * halfHarms]
        phase3 = specharm3[halfHarms:2 * halfHarms]
        phase4 = specharm4[halfHarms:2 * halfHarms]
        spectrum1 = specharm1[2 * halfHarms:]
        spectrum2 = specharm2[2 * halfHarms:]
        spectrum3 = specharm3[2 * halfHarms:]
        spectrum4 = specharm4[2 * halfHarms:]
        spectrum1 = torch.unsqueeze(spectrum1, 1)
        spectrum2 = torch.unsqueeze(spectrum2, 1)
        spectrum3 = torch.unsqueeze(spectrum3, 1)
        spectrum4 = torch.unsqueeze(spectrum4, 1)
        spectra = torch.cat((spectrum1, spectrum2, spectrum3, spectrum4), dim = 1)
        harm1 = specharm1[:halfHarms]
        harm2 = specharm2[:halfHarms]
        harm3 = specharm3[:halfHarms]
        harm4 = specharm4[:halfHarms]
        harm1 = torch.unsqueeze(harm1, 1)
        harm2 = torch.unsqueeze(harm2, 1)
        harm3 = torch.unsqueeze(harm3, 1)
        harm4 = torch.unsqueeze(harm4, 1)
        harms = torch.cat((harm1, harm2, harm3, harm4), dim = 1)
        limit = torch.max(spectra, dim = 1)[0]
        fac = torch.full((global_consts.halfTripleBatchSize + 1, 1), factor, device = self.device)
        facHarm = torch.full((halfHarms, 1), factor, device = self.device)
        predSpectrum = self.currPrediction[halfHarms:]
        predSpectrum = torch.unsqueeze(predSpectrum, 1)
        predHarm = self.currPrediction[:halfHarms]
        predHarm = torch.unsqueeze(predHarm, 1)
        x = torch.cat((spectra, fac, predSpectrum), dim = 1)
        x = x.float()
        x = torch.unsqueeze(torch.transpose(x, 0, 1), 0)
        x = self.convolution(x)
        x = torch.reshape(x, (-1,))
        x = self.layerStart1(x)
        x = self.ReLuStart1(x)
        x = self.layerStart2(x)
        x = self.ReLuStart2(x)
        y = torch.cat((harms, facHarm, predHarm), dim = 1)
        y = y.float()
        y = torch.unsqueeze(torch.transpose(y, 0, 1), 0)
        y = self.harmConvolution(y)
        y = torch.reshape(y, (-1,))
        y = self.harmLayerStart1(y)
        y = self.harmReLuStart1(y)
        y = self.harmLayerStart2(y)
        y = self.harmReLuStart2(y)
        x = torch.cat((x, y), 0)
        x = self.hiddenLayers(x)
        y = x[:4 * halfHarms]
        x = x[4 * halfHarms:]
        x = self.layerEnd1(x)
        x = self.ReLuEnd1(x)
        x = self.layerEnd2(x)
        x = self.ReLuEnd2(x)
        y = self.harmLayerEnd1(y)
        y = self.harmReLuEnd1(y)
        y = self.harmLayerEnd2(y)
        y = self.harmReLuEnd2(y)
        x = torch.minimum(x, limit)

        spectralFilterWidth = 4 * global_consts.filterTEEMult
        x = torch.fft.rfft(x, dim = 0)
        cutoffWindow = torch.zeros_like(x)
        cutoffWindow[0:int(spectralFilterWidth / 2)] = 1.
        cutoffWindow[int(spectralFilterWidth / 2):spectralFilterWidth] = torch.linspace(1, 0, spectralFilterWidth - int(spectralFilterWidth / 2))
        x = torch.fft.irfft(cutoffWindow * x, dim = 0, n = global_consts.halfTripleBatchSize + 1)
        x = self.threshold(x)
        
        phases = phaseInterp(phaseInterp(phase1, phase2, 0.5), phaseInterp(phase3, phase4, 0.5), factor)

        result = torch.cat((y, phases, x), 0)
        self.currPrediction = self.stepSpecPred(result)
        return result
    
    def processData(self, specharm1:torch.Tensor, specharm2:torch.Tensor, specharm3:torch.Tensor, specharm4:torch.Tensor, factor:float) -> torch.Tensor:
        """forward NN pass with data pre-and postprocessing as expected by other classes
        
        Arguments:
            spectrum1, spectrum2: The two spectrum Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object containing the interpolated audio spectrum
            
            
        Other than when using forward() directly, this method sets the NN to evaluation mode and applies a square root function to the input and squares the output to
        improve overall performance, especially on the skewed datasets of vocal spectra."""
        
        
        self.eval()
        output = torch.square(torch.squeeze(self(torch.sqrt(specharm1), torch.sqrt(specharm2), torch.sqrt(specharm3), torch.sqrt(specharm4), factor)))
        return output

    def stepSpecPred(self, specharm:torch.Tensor) -> torch.Tensor:
        self.pred.state[0].detach()
        self.pred.state[1].detach()
        return self.pred.processData(specharm)

    def resetSpecPred(self) -> None:
        self.pred.resetState()
        self.currPrediction = torch.zeros((halfHarms + global_consts.halfTripleBatchSize + 1,), device = self.currPrediction.device)
    
    def getState(self) -> dict:
        """returns the state of the NN and its epoch attribute in a Dictionary.
        
        Arguments:
            None
            
        Returns:
            Dictionary containing the NN's weights (state dict) and its epoch attribute (epoch)"""
            
            
        AiState = {'epoch': self.epoch,
                 'sampleCount': self.sampleCount,
                 'model_state_dict': self.state_dict(),
                 'pred_model_state_dict': self.pred.state_dict()
                 }
        return AiState

class RelLoss(nn.Module):
    """function for calculating relative loss values between target and actual Tensor objects. Designed to be used with AI optimizers. Currently unused.
    
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
 
    def forward(self, inputs:torch.Tensor, targets:torch.Tensor) -> float:  
        """calculates relative loss based on input and target tensors after successful initialisation.
        
        Arguments:
            inputs: AI-generated input Tensor
            
            targets: target Tensor
            
        Returns:
            Relative error value calculated from the difference between input and target Tensor as Float"""
        
        differences = torch.abs(inputs - targets)
        refs = torch.abs(targets)
        out = (differences / refs).sum() / inputs.size()[0]
        return out

class SpecPredAI(nn.Module):
    def __init__(self, device:torch.device = None, learningRate:float=5e-5) -> None:
        super().__init__()
        self.layer1Harm = torch.nn.Linear(halfHarms, halfHarms)
        self.layer1Spec = torch.nn.Linear(global_consts.halfTripleBatchSize + 1, global_consts.halfTripleBatchSize + 1, device = device)
        self.ReLuHarm1 = nn.ReLU()
        self.ReLuSpec1 = nn.ReLU()
        recSize =  halfHarms + global_consts.halfTripleBatchSize + 1
        self.sharedRecurrency = nn.LSTM(input_size = recSize, hidden_size = recSize, num_layers = 2)
        self.layer2Harm = torch.nn.Linear(halfHarms, halfHarms)
        self.layer2Spec = torch.nn.Linear(global_consts.halfTripleBatchSize + 1, global_consts.halfTripleBatchSize + 1, device = device)
        self.layer3Harm = torch.nn.Linear(halfHarms, halfHarms)
        self.layer3Spec = torch.nn.Linear(global_consts.halfTripleBatchSize + 1, global_consts.halfTripleBatchSize + 1, device = device)

        self.learningRate = learningRate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=0.)
        self.criterion = nn.L1Loss()
        self.state = (torch.zeros((2, 1, recSize), device = device), torch.zeros((2, 1, recSize), device = device))

    def forward(self, specharm:torch.Tensor) -> torch.Tensor:
        harmonics = specharm[:halfHarms]
        spectrum = specharm[halfHarms:]
        spectrum = torch.unsqueeze(spectrum)
        harmonics = torch.unsqueeze(harmonics)
        harmonics = self.layer1Harm(harmonics)
        spectrum = self.layer1Spec(spectrum)
        harmonics = self.ReLuHarm1(harmonics)
        spectrum = self.ReLuSpec1(spectrum)
        x = torch.cat((harmonics, spectrum), 0)
        x, self.state = self.sharedRecurrency(x, self.state)
        harmonics = x[:halfHarms]
        spectrum = x[halfHarms:]
        harmonics = self.layer2Harm(harmonics)
        spectrum = self.layer2Spec(spectrum)
        harmonics = self.layer3Harm(harmonics)
        spectrum = self.layer3Spec(spectrum)

        spectralFilterWidth = 4 * global_consts.filterTEEMult
        spectrum = torch.fft.rfft(spectrum, dim = 0)
        cutoffWindow = torch.zeros_like(spectrum)
        cutoffWindow[0:int(spectralFilterWidth / 2)] = 1.
        cutoffWindow[int(spectralFilterWidth / 2):spectralFilterWidth] = torch.linspace(1, 0, spectralFilterWidth - int(spectralFilterWidth / 2))
        spectrum = torch.fft.irfft(cutoffWindow * spectrum, dim = 0, n = global_consts.halfTripleBatchSize + 1)
        spectrum = self.threshold(spectrum)
        return torch.cat((harmonics, spectrum), 0)

    def processData(self, specharm:torch.Tensor) -> torch.Tensor:
        harmonics = specharm[:halfHarms]
        spectrum = specharm[2 * halfHarms:]
        harmonics = self.layer1Harm(harmonics)
        spectrum = self.layer1Spec(spectrum)
        harmonics = self.ReLuHarm1(harmonics)
        spectrum = self.ReLuSpec1(spectrum)
        x = torch.cat((harmonics, spectrum), 0)
        x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)
        x, self.state = self.sharedRecurrency(x, self.state)
        return x

    def train(self, indata, epochs:int=1) -> None:
        if indata == False:
            return
        for epoch in range(epochs):
            for data in self.dataLoader(indata):
                self.resetState()
                for i in range(len(data) - 1):
                    input = torch.cat((data[i, :halfHarms], data[i, 2 * halfHarms:]), 0)
                    target = torch.cat((data[i + 1, :halfHarms], data[i + 1, 2 * halfHarms:]), 0)
                    output = self.forward(input)
                    loss = self.criterion(output, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

    def resetState(self) -> None:
        recSize =  halfHarms + global_consts.halfTripleBatchSize + 1
        self.state = (torch.zeros((2, 1, recSize), device = self.state[0].device), torch.zeros((2, 1, recSize), device = self.state[1].device))

    def dataLoader(self, data) -> DataLoader:
        """helper method for shuffled data loading from an arbitrary dataset.
        
        Arguments:
            data: Tensor, List or Iterable representing a dataset with several elements
            
        Returns:
            Iterable representing the same dataset, with the order of its elements shuffled"""
        
        
        return DataLoader(dataset=data, shuffle=True)
