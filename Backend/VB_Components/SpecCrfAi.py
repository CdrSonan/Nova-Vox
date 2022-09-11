from typing import OrderedDict
import numpy as np
import math
from os import path, getenv
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import global_consts
from Backend.Resampler.PhaseShift import phaseInterp

halfHarms = int(global_consts.nHarmonics / 2) + 1

class SpecCrfAi(nn.Module):
    """class for generating crossfades between the spectra of different phonemes using AI.
    
    Attributes:
        convolution: leading 1-dimensional convolution layer of the NN. Used for spectral processing.

        harmConvolution: leading 1-dimensional convolution layer of the NN. Used for harmonic amplitudes processing.

        layerStart/End 1/2, ReLuStart/End 1/2: leading and trailing FC and Nonlinear layers of the NN. Used for spectral processing.

        harmLayerStart/End 1/2, harmReLuStart/End 1/2: leading and trailing FC and Nonlinear layers of the NN. Used for harmonic amplitudes processing.

        hiddenLayers: torch.nn.Sequential object containing all layers between the leading and trailing ones
        
        threshold: final threshold layer applied to the output spectrum, to prevent overshooting

        device: the torch.device the AI is loaded on

        hiddenLayerCount: integer indicating the number of hidden layers of the NN

        learningRate: Learning Rate of the NN
        
        optimizer: Optimization algorithm to use during training. Changes not advised.
        
        criterion: Loss criterion to be used during AI training. Changes not advised.
        
        epoch: training epoch counter displayed in Metadata panels

        sampleCount: integer indicating the number of samples the AI has been trained with.
        
        loss: float accumulating and normalizing recent loss values during AI training.

        pred: The AI's 'LSTM predictor'. SpecPredAi object used for temporal awareness.

        currPrediction: the most recent prediction of pred. Used as input to the main NN.
        
    Methods:
        __init__: Constructor initialising NN layers and prerequisite properties
        
        forward: Forward NN pass with unprocessed in-and outputs
        
        processData: forward NN pass with data pre-and postprocessing as expected by other classes
        
        train: NN training with forward and backward passes, Loss criterion and optimizer runs based on a dataset of spectral transition samples

        test: performs a set of tests for performance evaluation, and saves the results to a TensorBoard file.

        stepSpecPred: sends a specharm Tensor to the AI's LSTM predictor, and saves its updated prediction to currPrediction.

        resetSpecPred: resets the hidden states and cell states of the AI's LSTM predictor.
        
        dataLoader: helper method for shuffled data loading from an arbitrary dataset
        
        getState: returns the state of the NN, its optimizer and their prerequisites in a Dictionary
        
    The structure of the NN is a forward-feed fully connected NN with ReLU nonlinear activation functions.
    It is designed to process non-negative data. Negative data can still be processed, but may negatively impact performance.
    The size of the NN layers is set to process specharm Tensors, matching the format, batch size and tick rate used by the rest of the engine.
    Internally, each specharm is decomposed into its spectral and harmonics parts, which are sent through separate NN layers. They are afterwards
    combined with the current prediction of the AI's LSTM predictor, and sent through a set of shared layers. Afterwards, the spectral and harmonic
    components are processed by separate sets of layers once again. The final output is then fed back into the AI's LSTM predictor, updating
    its prediction for the next frame.
    Since performance deteriorates with skewed data, the NN internally passes the input through a square root function and squares the output."""
        
        
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
        #self.criterion = RelLoss()
        self.epoch = 0
        self.sampleCount = 0
        
    def forward(self, specharm1:torch.Tensor, specharm2:torch.Tensor, specharm3:torch.Tensor, specharm4:torch.Tensor, currPred:torch.Tensor, factor:float) -> torch.Tensor:
        """Forward NN pass with unprocessed in- and outputs.
        
        Arguments:
            specharm1-4: The sets of two spectrum + harmonics Tensors to perform the interpolation between, preceding and following the transition that is to be calculated, respectively.
            
            factor: Float between 0 and 1 determining the "position" within the interpolation. When using a value of 0 the output will be extremely similar to specharm 1 and 2,
            while a values of 1 will result in output extremely similar to specharm 3 and 4.
            
        Returns:
            Tensor object representing the NN output
            
        When performing forward NN runs, it is strongly recommended to use processData() instead of this method."""

        if factor.__class__ == torch.Tensor:
            factor = factor.item()

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
        predSpectrum = currPred[:, :, halfHarms:]
        predSpectrum = torch.squeeze(predSpectrum)
        predSpectrum = torch.unsqueeze(predSpectrum, 1)
        predHarm = currPred[:, :, :halfHarms]
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
        return result


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
    """Class for providing additional time awareness to a spectral crossfade Ai.
    
    Methods:
        forward: processes a specharm tensor, updating the internal states and returning the predicted next spectrum
        
        processData: processes a specharm tensor, updating the internal states and returning the immediate LSTM output
        
        train: trains the AI using recorded sequences of specharms
        
        resetState: resets the hidden states and cell states of the internal LSTM layers
        
        dataLoader: utility function to assist with data loading for training
        
    The NN is trained on sequences of specharms, and always aims to predict the next specharm in the sequence using its internal LSTM layers.
    When used with a SpecCrfAi, it instead returns the immediate output of the lowest LSTM layer, for use as input for the SpecCrfAi.
    This is because the SpecCrfAi is using the data about already processed specharms in a more abstract way similar to representation
    learning, rather than an estimated specharm."""


    def __init__(self, device:torch.device = None, learningRate:float=5e-5) -> None:
        """basic constructor accepting the learning rate hyperparameter as input"""

        super().__init__()
        self.layer1Harm = torch.nn.Linear(halfHarms, halfHarms)
        self.layer1Spec = torch.nn.Linear(global_consts.halfTripleBatchSize + 1, global_consts.halfTripleBatchSize + 1, device = device)
        self.ReLuHarm1 = nn.ReLU()
        self.ReLuSpec1 = nn.ReLU()
        recSize =  halfHarms + global_consts.halfTripleBatchSize + 1
        self.sharedRecurrency = nn.LSTM(input_size = recSize, hidden_size = recSize, num_layers = 2, batch_first = True)
        self.layer2Harm = torch.nn.Linear(halfHarms, halfHarms)
        self.layer2Spec = torch.nn.Linear(global_consts.halfTripleBatchSize + 1, global_consts.halfTripleBatchSize + 1, device = device)
        self.layer3Harm = torch.nn.Linear(halfHarms, halfHarms)
        self.layer3Spec = torch.nn.Linear(global_consts.halfTripleBatchSize + 1, global_consts.halfTripleBatchSize + 1, device = device)

        self.learningRate = learningRate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=0.)
        self.criterion = nn.L1Loss()
        self.loss = 0
        self.state = (torch.zeros((2, 1, recSize), device = device), torch.zeros((2, 1, recSize), device = device))

        self.threshold = torch.nn.Threshold(0.001, 0.001)

    def forward(self, specharm:torch.Tensor) -> torch.Tensor:
        """forward pass through the entire NN, aiming to predict the next specharm in a sequence"""

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
        encoded = x
        x = torch.squeeze(x)
        harmonics = x[:, :halfHarms]
        spectrum = x[:, halfHarms:]
        harmonics = self.layer2Harm(harmonics)
        spectrum = self.layer2Spec(spectrum)
        harmonics = self.layer3Harm(harmonics)
        spectrum = self.layer3Spec(spectrum)

        spectralFilterWidth = 4 * global_consts.filterTEEMult
        spectrum = torch.fft.rfft(spectrum, dim = 1)
        cutoffWindow = torch.zeros_like(spectrum)
        cutoffWindow[:, 0:int(spectralFilterWidth / 2)] = 1.
        cutoffWindow[:, int(spectralFilterWidth / 2):spectralFilterWidth] = torch.linspace(1, 0, spectralFilterWidth - int(spectralFilterWidth / 2))
        spectrum = torch.fft.irfft(cutoffWindow * spectrum, dim = 1, n = global_consts.halfTripleBatchSize + 1)
        spectrum = self.threshold(spectrum)
        return encoded, torch.cat((harmonics, spectrum), 1)

    def resetState(self) -> None:
        """resets the hidden states and cell states of the LSTM layers. Should be called between training or inference runs."""

        recSize =  halfHarms + global_consts.halfTripleBatchSize + 1
        self.state = (torch.zeros((2, 1, recSize), device = self.state[0].device), torch.zeros((2, 1, recSize), device = self.state[1].device))

class AIWrapper():
    def __init__(self, device = torch.device("cpu")) -> None:
        self.crfAi = SpecCrfAi(device = device)
        self.predAi = SpecPredAI(device = device)
        self.currPred = torch.zeros((halfHarms + global_consts.halfTripleBatchSize + 1,), device = device)
        self.device = device
        self.crfAiOptimizer = torch.optim.Adam(self.crfAi.parameters(), lr=self.crfAi.learningRate, weight_decay=0.)
        self.predAiOptimizer = torch.optim.Adam(self.predAi.parameters(), lr=self.predAi.learningRate, weight_decay=0.)
        self.criterion = nn.L1Loss()
    
    @staticmethod
    def dataLoader(data) -> DataLoader:
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
            
        if self.final:
            AiState = {'crfAi_epoch': self.crfAi.epoch,
                'crfAi_model_state_dict': self.crfAi.state_dict(),
                'crfAi_sampleCount': self.crfAi.sampleCount,
                'predAi_epoch': self.predAi.epoch,
                'predAi_model_state_dict': self.predAi.state_dict(),
                'predAi_sampleCount': self.predAi.sampleCount,
                'final': True
            }
        else:
            AiState = {'crfAi_epoch': self.crfAi.epoch,
                'crfAi_model_state_dict': self.crfAi.state_dict(),
                'crfAi_optimizer_state_dict': self.crfAiOptimizer.state_dict(),
                'crfAi_sampleCount': self.crfAi.sampleCount,
                'predAi_epoch': self.predAi.epoch,
                'predAi_model_state_dict': self.predAi.state_dict(),
                'predAi_optimizer_state_dict': self.predAiOptimizer.state_dict(),
                'predAi_sampleCount': self.predAi.sampleCount,
                'final': False
            }
        return AiState

    def loadState(self, AiState:dict) -> None:
        if AiState["final"]:
            pass
        else:
            pass

    def interpolate(self, specharm1:torch.Tensor, specharm2:torch.Tensor, specharm3:torch.Tensor, specharm4:torch.Tensor, factor:float) -> torch.Tensor:
        """forward NN pass with data pre- and postprocessing as expected by other classes
        
        Arguments:
            specharm1, specharm2: The two specharm Tensors to perform the interpolation between
            
            factor: Float between 0 and 1 determining the "position" in the interpolation. A value of 0 matches spectrum 1, a values of 1 matches spectrum 2.
            
        Returns:
            Tensor object containing the interpolated audio spectrum
            
            
        Other than when using forward() directly, this method sets the NN to evaluation mode, applies a square root function to the input and squares the output to
        improve overall performance, especially on the skewed datasets of vocal spectra."""
        
        
        self.crfAi.eval()
        self.predAi.eval()
        output = torch.square(torch.squeeze(self.crfAi(torch.sqrt(specharm1), torch.sqrt(specharm2), torch.sqrt(specharm3), torch.sqrt(specharm4), self.currPred, factor)))
        self.currPred, prediction = self.predAi(output)
        return output, prediction

    def predict(self, specharm:torch.Tensor):
        self.predAi.eval()
        self.currPred, prediction = self.predAi(specharm)
        return prediction

    def reset(self) -> None:
        """resets the hidden states and cell states of the AI's LSTM Predictor subnet."""

        self.predAi.resetState()
        self.currPred = torch.zeros((1, 1, halfHarms + global_consts.halfTripleBatchSize + 1), device = self.device)

    def finalize(self):
        self.final = True

    def trainCrf(self, indata, epochs:int=1, logging:bool = False) -> None:
        """NN training with forward and backward passes, loss criterion and optimizer runs based on a dataset of spectral transition samples.
        
        Arguments:
            indata: Tensor, List or other Iterable containing sets of specharm data. Each element should represent a phoneme transition.
            
            epochs: number of epochs to use for training as Integer.
            
        Returns:
            None
            
        Like processData(), train() also takes the square root of the input internally before using the data."""
        
        if logging:
            writer = SummaryWriter(path.join(getenv("APPDATA"), "Nova-Vox", "Logs"))
            #writer.add_graph(self, (indata[0][0], indata[0][1], indata[0][-2], indata[0][-1], torch.tensor([0.5])))
        else:
            writer = None

        if (self.epoch == 0) or self.epoch == epochs:
            self.epoch = epochs
        else:
            self.epoch = None
        reportedLoss = 0.
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
                    output = torch.squeeze(self.forward(spectrum1, spectrum2, spectrum3, spectrum4, factor)[0])
                    output = torch.cat((output[:halfHarms], output[2 * halfHarms:]), 0)
                    spectrumTarget = torch.cat((spectrumTarget[:halfHarms], spectrumTarget[2 * halfHarms:]), 0)
                    loss = self.criterion(output, spectrumTarget)
                    self.crfAiOptimizer.zero_grad()
                    loss.backward()
                    self.crfAiOptimizer.step()
                    print('epoch [{}/{}], sub-sample index {}, loss:{:.4f}'.format(epoch + 1, epochs, i, loss.data))
            if writer != None:
                writer.add_scalar("loss", loss.data)
            self.crfAi.sampleCount += len(indata)
            reportedLoss = (reportedLoss * 99 + loss.data) / 100
        hparams = dict()
        hparams["epochs"] = epochs
        hparams["learning rate"] = self.crfAi.learningRate
        hparams["hidden layer count"] = self.crfAi.hiddenLayerCount
        metrics = dict()
        metrics["acc. sample count"] = self.crfAi.sampleCount
        metrics["wtd. train loss"] = reportedLoss
        if writer != None:
            writer.add_hparams(hparams, metrics)
            writer.close()
    
    def trainPred(self, indata, epochs:int=1, writer:SummaryWriter = None) -> None:
        """trains the NN based on a dataset of specharm sequences"""

        reportedLoss = 0.
        for epoch in range(epochs):
            for data in self.dataLoader(indata):
                self.reset()
                input = torch.cat((data[:, :-1, :halfHarms], data[:, :-1, 2 * halfHarms:]), 2)
                target = torch.cat((data[:, 1:, :halfHarms], data[:, 1:, 2 * halfHarms:]), 2)
                input = torch.squeeze(input)
                target = torch.squeeze(target)
                output = self.predAi(input)
                loss = self.criterion(output, target)
                self.predAiOptimizer.zero_grad()
                loss.backward()
                self.predAiOptimizer.step()
                reportedLoss = (reportedLoss * 99 + loss.data) / 100
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss.data))
            if writer != None:
                writer.add_scalar("loss", loss.data)
        hparams = dict()
        hparams["epochs"] = epochs
        metrics = dict()
        metrics["wtd. train loss"] = self.loss
        if writer != None:
            writer.add_hparams(hparams, metrics)
            writer.close()
