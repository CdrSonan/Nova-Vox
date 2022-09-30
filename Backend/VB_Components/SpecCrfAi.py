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
        
        
    def __init__(self, device:torch.device = None, learningRate:float=5e-5, hiddenLayerCount:int = 3, hiddenLayerSize:int = 4 * (global_consts.halfTripleBatchSize + halfHarms), regularization:float=1e-5) -> None:
        """Constructor initialising NN layers and prerequisite attributes.
        
        Arguments:
            device: the device the AI is to be loaded on

            learningRate: desired learning rate of the NN as float. supports scientific format.
            
            hiddenLayerCount: number of hidden layers (between leading and trailing layers)

        Returns:
            None"""
            
            
        super(SpecCrfAi, self).__init__()
        self.convolution = nn.Conv1d(6, 6, 1, device = device)
        self.layerStart1 = torch.nn.Linear(6 * global_consts.halfTripleBatchSize + 6, int(3 * global_consts.halfTripleBatchSize + 3 + hiddenLayerSize / 4), device = device)
        self.ReLuStart1 = nn.ReLU()
        self.layerStart2 = torch.nn.Linear(int(3 * global_consts.halfTripleBatchSize + 3 + hiddenLayerSize / 4), math.ceil(hiddenLayerSize / 2), device = device)
        self.ReLuStart2 = nn.ReLU()
        self.harmConvolution = nn.Conv1d(6, 6, 1, device = device)
        self.harmLayerStart1 = torch.nn.Linear(6 * halfHarms, int(3 * halfHarms + hiddenLayerSize / 4), device = device)
        self.harmReLuStart1 = nn.ReLU()
        self.harmLayerStart2 = torch.nn.Linear(int(3 * halfHarms + hiddenLayerSize / 4), math.floor(hiddenLayerSize / 2), device = device)
        self.harmReLuStart2 = nn.ReLU()
        hiddenLayerDict = OrderedDict([])
        for i in range(hiddenLayerCount):
            hiddenLayerDict["layer" + str(i)] = torch.nn.Linear(hiddenLayerSize, hiddenLayerSize, device = device)
            hiddenLayerDict["ReLu" + str(i)] = nn.ReLU()
        self.hiddenLayers = nn.Sequential(hiddenLayerDict)
        self.layerEnd1 = torch.nn.Linear(math.ceil(hiddenLayerSize / 2), int(hiddenLayerSize / 4 + global_consts.halfTripleBatchSize / 2), device = device)
        self.ReLuEnd1 = nn.ReLU()
        self.layerEnd2 = torch.nn.Linear(int(hiddenLayerSize / 4 + global_consts.halfTripleBatchSize / 2), global_consts.halfTripleBatchSize + 1, device = device)
        self.ReLuEnd2 = nn.ReLU()
        self.harmLayerEnd1 = torch.nn.Linear(math.floor(hiddenLayerSize / 2), int(hiddenLayerSize / 4 + halfHarms / 2), device = device)
        self.harmReLuEnd1 = nn.ReLU()
        self.harmLayerEnd2 = torch.nn.Linear(int(hiddenLayerSize / 4 + halfHarms / 2), halfHarms, device = device)
        self.harmReLuEnd2 = nn.ReLU()
        self.threshold = torch.nn.Threshold(0.001, 0.001)

        self.device = device
        self.learningRate = learningRate
        self.hiddenLayerCount = hiddenLayerCount
        self.hiddenLayerSize = hiddenLayerSize
        self.regularization = regularization
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
        currPred = currPred.detach()
        predSpectrum = currPred[halfHarms:]
        predSpectrum = torch.unsqueeze(predSpectrum, 1)
        predHarm = currPred[:halfHarms]
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
        y = x[:math.ceil(self.hiddenLayerSize / 2)]
        x = x[math.ceil(self.hiddenLayerSize / 2):]
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


    def __init__(self, device:torch.device = None, learningRate:float=5e-5, recLayerCount:int=1, recSize:int=halfHarms + global_consts.halfTripleBatchSize + 1, regularization:float=1e-5) -> None:
        """basic constructor accepting the learning rate hyperparameter as input"""

        super().__init__()
        self.layer1Harm = torch.nn.Linear(halfHarms, halfHarms)
        self.layer1Spec = torch.nn.Linear(global_consts.halfTripleBatchSize + 1, global_consts.halfTripleBatchSize + 1, device = device)
        self.ReLuHarm1 = nn.ReLU()
        self.ReLuSpec1 = nn.ReLU()
        self.recSize = recSize
        self.sharedRecurrency1 = nn.LSTMCell(halfHarms + global_consts.halfTripleBatchSize + 1, recSize, device = device)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.sharedRecurrency2 = nn.LSTMCell(recSize, recSize, device = device)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.sharedRecurrency3 = nn.LSTMCell(recSize, halfHarms + global_consts.halfTripleBatchSize + 1, device = device)
        self.layer2Harm = torch.nn.Linear(halfHarms, halfHarms)
        self.layer2Spec = torch.nn.Linear(global_consts.halfTripleBatchSize + 1, global_consts.halfTripleBatchSize + 1, device = device)
        self.layer3Harm = torch.nn.Linear(halfHarms, halfHarms)
        self.layer3Spec = torch.nn.Linear(global_consts.halfTripleBatchSize + 1, global_consts.halfTripleBatchSize + 1, device = device)
        self.threshold = torch.nn.Threshold(0.001, 0.001)

        self.hiddenState1 = torch.zeros((1, recSize), device = device)
        self.cellState1 = torch.zeros((1, recSize), device = device)
        self.hiddenState2 = torch.zeros((1, recSize), device = device)
        self.cellState2 = torch.zeros((1, recSize), device = device)
        self.hiddenState3 = torch.zeros((1, halfHarms + global_consts.halfTripleBatchSize + 1), device = device)
        self.cellState3 = torch.zeros((1, halfHarms + global_consts.halfTripleBatchSize + 1), device = device)
        
        self.device = device
        self.learningRate = learningRate
        self.recLayerCount = recLayerCount
        self.recSize = recSize
        self.regularization = regularization
        self.epoch = 0
        self.sampleCount = 0
        

    def forward(self, specharm:torch.Tensor) -> torch.Tensor:
        """forward pass through the entire NN, aiming to predict the next specharm in a sequence"""

        harmonics = specharm[:halfHarms]
        spectrum = specharm[2 * halfHarms:]
        harmonics = self.layer1Harm(harmonics)
        spectrum = self.layer1Spec(spectrum)
        harmonics = self.ReLuHarm1(harmonics)
        spectrum = self.ReLuSpec1(spectrum)
        x = torch.unsqueeze(torch.cat((harmonics, spectrum), 0), 0)
        self.hiddenState1, self.cellState1 = self.sharedRecurrency1(x, (self.hiddenState1, self.cellState1))
        self.hiddenState1 = self.dropout1(self.hiddenState1)
        self.hiddenState2, self.cellState2 = self.sharedRecurrency1(self.hiddenState1, (self.hiddenState2, self.cellState2))
        self.hiddenState2 = self.dropout2(self.hiddenState2)
        self.hiddenState3, self.cellState3 = self.sharedRecurrency1(self.hiddenState2, (self.hiddenState3, self.cellState3))
        x = self.hiddenState3
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
        return torch.squeeze(self.hiddenState3), torch.cat((harmonics, spectrum), 1)

    def resetState(self) -> None:
        """resets the hidden states and cell states of the LSTM layers. Should be called between training or inference runs."""

        self.hiddenState1 = torch.zeros((1, self.recSize), device = self.device)
        self.cellState1 = torch.zeros((1, self.recSize), device = self.device)
        self.hiddenState2 = torch.zeros((1, self.recSize), device = self.device)
        self.cellState2 = torch.zeros((1, self.recSize), device = self.device)
        self.hiddenState3 = torch.zeros((1, halfHarms + global_consts.halfTripleBatchSize + 1), device = self.device)
        self.cellState3 = torch.zeros((1, halfHarms + global_consts.halfTripleBatchSize + 1), device = self.device)

class AIWrapper():
    def __init__(self, device = torch.device("cpu"), hparams:dict = None) -> None:
        self.hparams = {
            "crf_lr": 0.001,
            "crf_reg": 0.0001,
            "crf_hlc": 5,
            "crf_hls": 512,
            "pred_lr": 0.01,
            "pred_reg": 0.001,
            "pred_rs": 512
        }
        for i in hparams.keys():
            self.hparams[i] = hparams[i]
        self.crfAi = SpecCrfAi(device = device, learningRate=self.hparams["crf_lr"], regularization=self.hparams["crf_reg"], hiddenLayerCount=self.hparams["crf_hlc"], hiddenLayerSize=self.hparams["crf_hls"])
        self.predAi = SpecPredAI(device = device, learningRate=self.hparams["pred_lr"], regularization=self.hparams["pred_reg"], recSize=self.hparams["rs"])
        self.currPred = torch.zeros((halfHarms + global_consts.halfTripleBatchSize + 1,), device = device)
        self.device = device
        self.final = False
        self.crfAiOptimizer = torch.optim.Adam(self.crfAi.parameters(), lr=self.crfAi.learningRate, weight_decay=self.crfAi.regularization)
        self.predAiOptimizer = torch.optim.Adam(self.predAi.parameters(), lr=self.predAi.learningRate, weight_decay=self.predAi.regularization)
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
            aiState = {'crfAi_epoch': self.crfAi.epoch,
                'crfAi_model_state_dict': self.crfAi.state_dict(),
                'crfAi_sampleCount': self.crfAi.sampleCount,
                'predAi_epoch': self.predAi.epoch,
                'predAi_model_state_dict': self.predAi.state_dict(),
                'predAi_sampleCount': self.predAi.sampleCount,
                'final': True
            }
        else:
            aiState = {'crfAi_epoch': self.crfAi.epoch,
                'crfAi_model_state_dict': self.crfAi.state_dict(),
                'crfAi_optimizer_state_dict': self.crfAiOptimizer.state_dict(),
                'crfAi_sampleCount': self.crfAi.sampleCount,
                'predAi_epoch': self.predAi.epoch,
                'predAi_model_state_dict': self.predAi.state_dict(),
                'predAi_optimizer_state_dict': self.predAiOptimizer.state_dict(),
                'predAi_sampleCount': self.predAi.sampleCount,
                'final': False
            }
        return aiState

    def loadState(self, aiState:dict, mode:str = None, reset:bool=False) -> None:
        if aiState["final"]:
            self.final = True
            pass
        else:
            pass

        if (mode == None) or (mode == "crf"):
            if reset:
                self.crfAi = SpecCrfAi(device = self.device, learningRate=self.hparams["crf_lr"], regularization=self.hparams["crf_reg"], hiddenLayerCount=self.hparams["crf_hlc"], hiddenLayerSize=self.hparams["crf_hls"])
                self.crfAiOptimizer = torch.optim.Adam(self.crfAi.parameters(), lr=self.crfAi.learningRate, weight_decay=self.crfAi.regularization)
            self.crfAi.epoch = aiState['crfAi_epoch']
            self.crfAi.sampleCount = aiState["crfAi_sampleCount"]
            self.crfAi.load_state_dict(aiState['crfAi_model_state_dict'])
            self.crfAiOptimizer.load_state_dict(aiState['crfAi_optimizer_state_dict'])
        if (mode == None) or (mode == "pred"):
            if reset:
                self.predAi = SpecPredAI(device = self.device, learningRate=self.hparams["pred_lr"], regularization=self.hparams["pred_reg"], recSize=self.hparams["rs"])
                self.predAiOptimizer = torch.optim.Adam(self.predAi.parameters(), lr=self.predAi.learningRate, weight_decay=self.predAi.regularization)
            self.predAi.epoch = aiState["predAi_epoch"]
            self.predAi.sampleCount = aiState["predAi_sampleCount"]
            self.predAi.load_state_dict(aiState['predAi_model_state_dict'])
            self.predAiOptimizer.load_state_dict(aiState['predAi_optimizer_state_dict'])
        self.crfAi.eval()
        self.predAi.eval()

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
        self.currPred = torch.zeros((halfHarms + global_consts.halfTripleBatchSize + 1,), device = self.device)

    def prefetch(self, specharm:torch.Tensor) -> None:
        harmonics = specharm[:halfHarms]
        spectrum = specharm[2 * halfHarms:]
        self.predAi.hiddenState3 = torch.unsqueeze(torch.cat((harmonics, spectrum), 0), 0)

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
        

        self.crfAi.train()
        if logging:
            writer = SummaryWriter(path.join(getenv("APPDATA"), "Nova-Vox", "Logs"))
            #writer.add_graph(self, (indata[0][0], indata[0][1], indata[0][-2], indata[0][-1], torch.tensor([0.5])))
        else:
            writer = None

        if (self.crfAi.epoch == 0) or self.crfAi.epoch == epochs:
            self.crfAi.epoch = epochs
        else:
            self.crfAi.epoch = None
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
                
                self.reset()
                self.prefetch(data[0])
                for i in range(1, data.size()[0]):
                    factor = i / float(data.size()[0])
                    spectrumTarget = data[i]
                    output = self.crfAi(spectrum1, spectrum2, spectrum3, spectrum4, self.currPred, factor)
                    self.currPred, prediction = self.predAi(output)
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
    
    def trainPred(self, indata, epochs:int=1, logging:bool = False) -> None:
        """trains the NN based on a dataset of specharm sequences"""

        if logging:
            writer = SummaryWriter(path.join(getenv("APPDATA"), "Nova-Vox", "Logs"))
            #writer.add_graph(self, (indata[0][0], indata[0][1], indata[0][-2], indata[0][-1], torch.tensor([0.5])))
        else:
            writer = None
        if (self.predAi.epoch == 0) or self.epoch == epochs:
            self.predAi.epoch = epochs
        else:
            self.predAi.epoch = None
        reportedLoss = 0.
        for epoch in range(epochs):
            for data in self.dataLoader(indata):
                self.reset()
                self.prefetch(data[0])
                input = torch.cat((data[:-1, :halfHarms], data[:-1, 2 * halfHarms:]), 1)
                target = torch.cat((data[1:, :halfHarms], data[1:, 2 * halfHarms:]), 1)
                input = torch.squeeze(input)
                target = torch.squeeze(target)
                self.currPred, output = self.predAi(input)
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
