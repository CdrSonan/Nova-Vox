import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import global_consts

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
        
        
    def __init__(self, device = None, learningRate=1e-4):
        """Constructor initialising NN layers and prerequisite attributes.
        
        Arguments:
            learningRate: desired learning rate of the NN as float. supports scientific format.
            
        Returns:
            None"""
            
            
        super(SpecCrfAi, self).__init__()

        self.layer1 = torch.nn.Linear(3 * global_consts.halfTripleBatchSize + 4, global_consts.tripleBatchSize + 3, device = device)
        self.ReLu1 = nn.ReLU()
        self.layer2 = torch.nn.Linear(global_consts.tripleBatchSize + 3, global_consts.tripleBatchSize + 3, device = device)
        self.ReLu2 = nn.ReLU()
        self.layer3 = torch.nn.Linear(global_consts.tripleBatchSize + 3, 2 * global_consts.tripleBatchSize, device = device)
        self.ReLu3 = nn.ReLU()
        self.layer4 = torch.nn.Linear(2 * global_consts.tripleBatchSize, global_consts.tripleBatchSize + 3, device = device)
        self.ReLu4 = nn.ReLU()
        self.layer5 = torch.nn.Linear(global_consts.tripleBatchSize + 3, global_consts.halfTripleBatchSize + 1, device = device)
        
        self.device = device
        
        self.learningRate = learningRate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=0.)
        #self.criterion = nn.L1Loss()
        self.criterion = RelLoss()
        self.epoch = 0
        self.sampleCount = 0
        self.loss = None
        
    def forward(self, spectrum1, spectrum2, factor):
        """Forward NN pass with unprocessed in-and outputs.
        
        Arguments:
            spectrum1, spectrum2: The two spectrum Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object representing the NN output
            
        When performing forward NN runs, it is strongly recommended to use processData() instead of this method."""
        
        
        fac = torch.tensor([factor], device = self.device)
        interpolated = (spectrum1 * (1. - fac)) + (spectrum2 * fac)
        x = torch.cat((spectrum1, spectrum2, interpolated, fac), dim = 0)
        x = x.float()
        x = self.layer1(x)
        x = self.ReLu1(x)
        x = self.layer2(x)
        x = self.ReLu2(x)
        x = self.layer3(x)
        x = self.ReLu3(x)
        x = self.layer4(x)
        x = self.ReLu4(x)
        x = self.layer5(x)
        return x
    
    def processData(self, spectrum1, spectrum2, factor):
        """forward NN pass with data pre-and postprocessing as expected by other classes
        
        Arguments:
            spectrum1, spectrum2: The two spectrum Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object containing the interpolated audio spectrum
            
            
        Other than when using forward() directly, this method sets the NN to evaluation mode and applies a square root function to the input and squares the output to
        improve overall performance, especially on the skewed datasets of vocal spectra."""
        
        
        self.eval()
        output = torch.square(torch.squeeze(self(torch.sqrt(spectrum1), torch.sqrt(spectrum2), factor)))
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
            sampleCount = len(indata)
            for epoch in range(epochs):
                for data in self.dataLoader(indata):
                    print('epoch [{}/{}], switching to next sample'.format(epoch + 1, epochs))
                    data = torch.sqrt(data.to(device = self.device))
                    data = torch.squeeze(data)
                    spectrum1 = data[0]
                    spectrum2 = data[-1]
                    indexList = np.arange(0, data.size()[0], 1)
                    np.random.shuffle(indexList)
                    for i in indexList:
                        factor = i / float(data.size()[0])
                        spectrumTarget = data[i]
                        output = torch.squeeze(self(spectrum1, spectrum2, factor))
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
        
        self.layer1 = torch.nn.Linear(3 * global_consts.halfTripleBatchSize + 4, global_consts.tripleBatchSize + 3, device = device)
        self.ReLu1 = nn.ReLU()
        self.layer2 = torch.nn.Linear(global_consts.tripleBatchSize + 3, global_consts.tripleBatchSize + 3, device = device)
        self.ReLu2 = nn.ReLU()
        self.layer3 = torch.nn.Linear(global_consts.tripleBatchSize + 3, 2 * global_consts.tripleBatchSize, device = device)
        self.ReLu3 = nn.ReLU()
        self.layer4 = torch.nn.Linear(2 * global_consts.tripleBatchSize, global_consts.tripleBatchSize + 3, device = device)
        self.ReLu4 = nn.ReLU()
        self.layer5 = torch.nn.Linear(global_consts.tripleBatchSize + 3, global_consts.halfTripleBatchSize + 1, device = device)
        
        self.device = device

        if specCrfAi == None:
            self.epoch = 0
            self.sampleCount = 0
        else:
            self.epoch = specCrfAi.getState()['epoch']
            self.sampleCount = specCrfAi.getState()['sampleCount']
            self.load_state_dict(specCrfAi.getState()['model_state_dict'])
            self.eval()
        
    def forward(self, spectrum1, spectrum2, factor):
        """Forward NN pass with unprocessed in-and outputs.
        
        Arguments:
            spectrum1, spectrum2: The two spectrum Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object representing the NN output
            
        When performing forward NN runs, it is strongly recommended to use processData() instead of this method."""
        
        
        fac = torch.tensor([factor], device = self.device)
        interpolated = (spectrum1 * (1. - fac)) + (spectrum2 * fac)
        x = torch.cat((spectrum1, spectrum2, interpolated, fac), dim = 0)
        x = x.float()
        x = self.layer1(x)
        x = self.ReLu1(x)
        x = self.layer2(x)
        x = self.ReLu2(x)
        x = self.layer3(x)
        x = self.ReLu3(x)
        x = self.layer4(x)
        x = self.ReLu4(x)
        x = self.layer5(x)
        return x
    
    def processData(self, spectrum1, spectrum2, factor):
        """forward NN pass with data pre-and postprocessing as expected by other classes
        
        Arguments:
            spectrum1, spectrum2: The two spectrum Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object containing the interpolated audio spectrum
            
            
        Other than when using forward() directly, this method sets the NN to evaluation mode and applies a square root function to the input and squares the output to
        improve overall performance, especially on the skewed datasets of vocal spectra."""
        
        
        self.eval()
        output = torch.square(torch.squeeze(self(torch.sqrt(spectrum1), torch.sqrt(spectrum2), factor)))
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
        
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        differences = torch.abs(inputs - targets)
        refs = torch.abs(targets)
        out = (differences / refs).sum() / inputs.size()[0]
        return out