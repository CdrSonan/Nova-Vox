#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import math
from os import path, getenv
from csv import DictWriter
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import global_consts
from Backend.VB_Components.Ai.CrfAi import SpecCrfAi
from Backend.VB_Components.Ai.PredAi import SpecPredAi, SpecPredDiscriminator, DataGenerator
from Backend.VB_Components.Ai.VAE import VAE
from Backend.Resampler.PhaseShift import phaseInterp
from Backend.Resampler.CubicSplineInter import interp
from Util import dec2bin

halfHarms = int(global_consts.nHarmonics / 2) + 1


class AIWrapper():
    """Wrapper class for the mandatory AI components of a Voicebank. Controls data pre- and postprocessing, state loading and saving, Hyperparameters, and both training and inference."""

    def __init__(self, voicebank, device = torch.device("cpu"), hparams:dict = None) -> None:
        """constructor taking a target device and dictionary of hyperparameters as input"""

        self.hparams = {
            "crf_lr": 0.000055,
            "crf_reg": 0.,
            "crf_hlc": 1,
            "crf_hls": 4000,
            "crf_def_thrh" : 0.05,
            "pred_lr": 1.,
            "pred_reg": 0.,
            "pred_rlc": 3,
            "pred_rs": 1024,
            "pred_drp":0.5,
            "preddisc_lr": 1.,
            "preddisc_reg": 0.,
            "preddisc_rlc": 3,
            "preddisc_rs": 1024,
            "preddisc_drp":0.25,
            "pred_guide_wgt": 1.,
            "pred_train_asym": 4,
            "latent_dim": 128
        }
        if hparams:
            for i in hparams.keys():
                self.hparams[i] = hparams[i]
        self.voicebank = voicebank
        self.crfAi = SpecCrfAi(device = device, learningRate=self.hparams["crf_lr"], regularization=self.hparams["crf_reg"], hiddenLayerCount=int(self.hparams["crf_hlc"]), hiddenLayerSize=int(self.hparams["crf_hls"]))
        self.predAi = SpecPredAi(device = device, learningRate=self.hparams["pred_lr"], regularization=self.hparams["pred_reg"], recSize=int(self.hparams["pred_rs"]), recLayerCount=int(self.hparams["pred_rlc"]), dropout = self.hparams["pred_drp"])
        self.predAiDisc = SpecPredDiscriminator(device = device, learningRate=self.hparams["preddisc_lr"], regularization=self.hparams["preddisc_reg"], recSize=int(self.hparams["preddisc_rs"]), recLayerCount=int(self.hparams["preddisc_rlc"]), dropout = self.hparams["preddisc_drp"])
        self.predAiGenerator = DataGenerator(self.voicebank, self.crfAi)
        self.VAE = VAE(device = device, latent_dim = self.hparams["latent_dim"])
        self.device = device
        self.final = False
        self.defectiveCrfBins = []
        self.crfAiOptimizer = torch.optim.NAdam(self.crfAi.parameters(), lr=self.crfAi.learningRate, weight_decay=self.crfAi.regularization)
        self.predAiOptimizer = torch.optim.Adadelta(self.predAi.parameters(), lr=self.predAi.learningRate, weight_decay=self.predAi.regularization)
        self.predAiDiscOptimizer = torch.optim.Adadelta(self.predAiDisc.parameters(), lr=self.predAiDisc.learningRate, weight_decay=self.predAiDisc.regularization)
        self.VAEOptimizer = torch.optim.Adam(self.VAE.parameters(), lr=0.0001)
        self.criterion = nn.L1Loss()
        self.guideCriterion = nn.MSELoss()
        self.pretrainCriterion = nn.BCELoss()
        self.deskewingPremul = torch.ones((global_consts.halfTripleBatchSize + halfHarms + 1,), device = self.device)
    
    @staticmethod
    def dataLoader(data) -> DataLoader:
        """helper method for shuffled data loading from an arbitrary dataset.
        
        Arguments:
            data: Tensor, List or Iterable representing a dataset with several elements
            
        Returns:
            Iterable representing the same dataset, with the order of its elements shuffled"""
        
        
        return DataLoader(dataset=data, shuffle=True)

    def getState(self) -> dict:
        """returns the state of the NN, its optimizer and their prerequisites as well as its epoch and sample count attributes in a Dictionary.
        
        Arguments:
            None
            
        Returns:
            Dictionary containing the NN's epoch attribute (epoch), weights (state dict), optimizer state (optimizer state dict) and sample count attribute (sampleCount)"""
            
        if self.final:
            aiState = {'crfAi_epoch': self.crfAi.epoch,
                'crfAi_model_state_dict': self.crfAi.state_dict(),
                'crfAi_sampleCount': self.crfAi.sampleCount,
                'predAi_epoch': self.predAi.epoch,
                'predAi_model_state_dict': self.predAi.state_dict(),
                'VAE_model_state_dict': self.VAE.state_dict(),
                'predAi_sampleCount': self.predAi.sampleCount,
                'deskew_premul': self.deskewingPremul,
                'defective_crf_bins': self.defectiveCrfBins,
                'final': True
            }
        else:
            aiState = {'crfAi_epoch': self.crfAi.epoch,
                'crfAi_model_state_dict': self.crfAi.state_dict(),
                'crfAi_optimizer_state_dict': self.crfAiOptimizer.state_dict(),
                'crfAi_sampleCount': self.crfAi.sampleCount,
                'predAi_epoch': self.predAi.epoch,
                'predAi_model_state_dict': self.predAi.state_dict(),
                'predAiDisc_model_state_dict': self.predAiDisc.state_dict(),
                'VAE_model_state_dict': self.VAE.state_dict(),
                'predAi_optimizer_state_dict': self.predAiOptimizer.state_dict(),
                'predAiDisc_optimizer_state_dict': self.predAiDiscOptimizer.state_dict(),
                'VAE_optimizer_state_dict': self.VAEOptimizer.state_dict(),
                'predAi_sampleCount': self.predAi.sampleCount,
                'deskew_premul': self.deskewingPremul,
                'defective_crf_bins': self.defectiveCrfBins,
                'final': False
            }
        return aiState

    def loadState(self, aiState:dict, mode:str = None, reset:bool=False) -> None:
        """loads the weights of the NNs managed by the wrapper from a dictionary, and reinitializes the NNs and/or their optimizers if required.
        
        Arguments:
            aiState: Dictionary in the same format as returned by getState(), containing all necessary information about the NNs
            
            mode: whether to load the weights for both NNs (None), only the phoneme crossfade Ai (crf), or only the prediction Ai (pred)
            
            reset: indicates whether the NNs and their optimizers should be reset before applying changed weights to them. Must be True when the dictionary contains weights
            for a NN using different hyperparameters than the currently active one."""

        if (mode == None) or (mode == "crf"):
            if reset:
                self.crfAi = SpecCrfAi(device = self.device, learningRate=self.hparams["crf_lr"], regularization=self.hparams["crf_reg"], hiddenLayerCount=self.hparams["crf_hlc"], hiddenLayerSize=self.hparams["crf_hls"])
                if aiState["final"]:
                    self.crfAiOptimizer = torch.optim.NAdam(self.crfAi.parameters(), lr=self.crfAi.learningRate, weight_decay=self.crfAi.regularization)
            self.crfAi.epoch = aiState['crfAi_epoch']
            self.crfAi.sampleCount = aiState["crfAi_sampleCount"]
            self.crfAi.load_state_dict(aiState['crfAi_model_state_dict'])
            if not aiState["final"]:
                self.crfAiOptimizer.load_state_dict(aiState['crfAi_optimizer_state_dict'])
        if (mode == None) or (mode == "pred"):
            if reset:
                self.predAi = SpecPredAi(device = self.device, learningRate=self.hparams["pred_lr"], regularization=self.hparams["pred_reg"], recSize=self.hparams["pred_rs"], recLayerCount=int(self.hparams["pred_rlc"]), dropout = self.hparams["pred_drp"])
                self.predAiDisc = SpecPredDiscriminator(device = self.device, learningRate=self.hparams["preddisc_lr"], regularization=self.hparams["preddisc_reg"], recSize=int(self.hparams["preddisc_rs"]), recLayerCount=int(self.hparams["preddisc_rlc"]), dropout = self.hparams["preddisc_drp"])
                if aiState["final"]:
                    self.predAiOptimizer = torch.optim.Adadelta(self.predAi.parameters(), lr=self.predAi.learningRate, weight_decay=self.predAi.regularization)
                    self.predAiDiscOptimizer = torch.optim.Adadelta(self.predAiDisc.parameters(), lr=self.predAiDisc.learningRate, weight_decay=self.predAiDisc.regularization)
            self.predAi.epoch = aiState["predAi_epoch"]
            self.predAi.sampleCount = aiState["predAi_sampleCount"]
            self.predAi.load_state_dict(aiState['predAi_model_state_dict'])
            self.VAE.load_state_dict(aiState['VAE_model_state_dict'])
            self.deskewingPremul = aiState["deskew_premul"]
            self.defectiveCrfBins = aiState["defective_crf_bins"]
            if not aiState["final"]:
                self.predAiDisc.load_state_dict(aiState['predAiDisc_model_state_dict'])
                self.predAiOptimizer.load_state_dict(aiState['predAi_optimizer_state_dict'])
                self.predAiDiscOptimizer.load_state_dict(aiState['predAiDisc_optimizer_state_dict'])
                self.VAEOptimizer.load_state_dict(aiState['VAE_optimizer_state_dict'])
        self.crfAi.eval()
        self.predAi.eval()

    def interpolate(self, specharm1:torch.Tensor, specharm2:torch.Tensor, specharm3:torch.Tensor, specharm4:torch.Tensor, embedding1:torch.Tensor, embedding2:torch.Tensor, outputSize:int, pitchCurve:torch.Tensor, slopeFactor:int) -> torch.Tensor:
        """forward pass of both NNs for generating a transition between two phonemes, with data pre- and postprocessing
        
        Arguments:
            specharm1-4: The four specharm Tensors to perform the interpolation between
            
            outputSize: length of the generated transition in engine ticks.

            pitchCurve: Tensor containing the pitch curve during the transition. Used to correctly calculate harmonic amplitudes during the transition.

            slopeFactor: integer expected to be between 0 and outputSize. Used for time remapping before the transition is calculated. Indicates where on the timeline 50% of the transition should have occured.
            
        Returns:
            tuple of two Tensor objects, containing the interpolated audio spectrum without and with the prediction Ai applied to it, respectively."""

        
        self.crfAi.eval()
        self.crfAi.requires_grad_(False)
        phase1 = specharm1[halfHarms:2 * halfHarms]
        phase2 = specharm2[halfHarms:2 * halfHarms]
        phase3 = specharm3[halfHarms:2 * halfHarms]
        phase4 = specharm4[halfHarms:2 * halfHarms]
        spectrum1 = specharm1[2 * halfHarms:]
        spectrum2 = specharm2[2 * halfHarms:]
        spectrum3 = specharm3[2 * halfHarms:]
        spectrum4 = specharm4[2 * halfHarms:]
        harm1 = specharm1[:halfHarms]
        harm2 = specharm2[:halfHarms]
        harm3 = specharm3[:halfHarms]
        harm4 = specharm4[:halfHarms]
        factor = math.log(0.5, slopeFactor / outputSize)
        factor = torch.pow(torch.linspace(0, 1, outputSize, device = self.device), factor)
        spectrum = torch.squeeze(self.crfAi(spectrum1, spectrum2, spectrum3, spectrum4, dec2bin(torch.tensor(embedding1, device = self.device), 32), dec2bin(torch.tensor(embedding2, device = self.device), 32), factor)).transpose(0, 1)
        for i in self.defectiveCrfBins:
            spectrum[:, i] = torch.mean(torch.cat((spectrum[:, i - 1].unsqueeze(1), spectrum[:, i + 1].unsqueeze(1)), 1), 1)
        borderRange = torch.zeros((outputSize,), device = self.device)
        borderLimit = min(global_consts.crfBorderAbs, math.ceil(outputSize * global_consts.crfBorderRel))
        borderRange[:borderLimit] = torch.linspace(1, 0, borderLimit, device = self.device)
        spectrum *= (1. - borderRange.unsqueeze(1))
        spectrum += torch.matmul(borderRange.unsqueeze(1), ((spectrum1 + spectrum2) / 2).unsqueeze(0))
        borderRange = torch.flip(borderRange, (0,))
        spectrum *= (1. - borderRange.unsqueeze(1))
        spectrum += torch.matmul(borderRange.unsqueeze(1), ((spectrum3 + spectrum4) / 2).unsqueeze(0))
        phases = torch.empty(outputSize, phase1.size()[0], device = self.device)
        nativePitch = math.ceil(global_consts.tripleBatchSize / pitchCurve[0])
        originSpace = torch.min(torch.linspace(nativePitch, int(global_consts.nHarmonics / 2) * nativePitch, int(global_consts.nHarmonics / 2) + 1, device = self.device), torch.tensor([global_consts.halfTripleBatchSize,], device = self.device))
        factors = interp(torch.linspace(0, global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = self.device), (spectrum1 + spectrum2) / 2, originSpace)
        harmsStart = (harm1 + harm2) * 0.5 / factors
        nativePitch = math.ceil(global_consts.tripleBatchSize / pitchCurve[-1])
        originSpace = torch.min(torch.linspace(nativePitch, int(global_consts.nHarmonics / 2) * nativePitch, int(global_consts.nHarmonics / 2) + 1, device = self.device), torch.tensor([global_consts.halfTripleBatchSize,], device = self.device))
        factors = interp(torch.linspace(0, global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = self.device), (spectrum3 + spectrum4) / 2, originSpace)
        harmsEnd = (harm3 + harm4) * 0.5 / factors
        harms = torch.empty((outputSize, halfHarms), device = self.device)
        harmLimit = torch.max(torch.cat((harm1.unsqueeze(1), harm2.unsqueeze(1), harm3.unsqueeze(1), harm4.unsqueeze(1)), 1))
        for i in range(outputSize):
            phases[i] = phaseInterp(phaseInterp(phase1, phase2, 0.5), phaseInterp(phase3, phase4, 0.5), i / (outputSize - 1))
            nativePitch = math.ceil(global_consts.tripleBatchSize / pitchCurve[i])
            originSpace = torch.min(torch.linspace(nativePitch, int(global_consts.nHarmonics / 2) * nativePitch, int(global_consts.nHarmonics / 2) + 1, device = self.device), torch.tensor([global_consts.halfTripleBatchSize,], device = self.device))
            factors = interp(torch.linspace(0, global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = self.device), spectrum[i], originSpace)
            harms[i] = ((1. - (i + 1) / (outputSize + 1)) * harmsStart + (i + 1) / (outputSize + 1) * harmsEnd) * factors
            harms[i] = torch.min(harms[i], harmLimit)
            harms[i] = torch.max(harms[i], torch.tensor([0.,], device = self.device))
        output = torch.cat((harms, phases, spectrum), 1)
        prediction = self.predict(output)
        return output, torch.squeeze(prediction)

    def predict(self, specharm:torch.Tensor):
        """forward pass through the prediction Ai, taking a specharm as input and predicting the next one in a sequence. Includes data pre- and postprocessing."""

        self.predAi.eval()
        self.predAi.requires_grad_(False)
        latent = self.VAE.encoder(specharm)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        prediction = self.predAi(latent, self.deskewingPremul, True)
        output = self.VAE.decoder(torch.squeeze(prediction))
        return torch.squeeze(output)

    def reset(self) -> None:
        """resets the hidden states and cell states of the AI's LSTM layers."""

        self.predAi.resetState()
        self.predAiDisc.resetState()

    def finalize(self):
        self.final = True

    def trainCrf(self, indata, epochs:int=1, logging:bool = False, reset:bool = False) -> None:
        """NN training with forward and backward passes, loss criterion and optimizer runs based on a dataset of spectral transition samples.
        
        Arguments:
            indata: Tensor, List or other Iterable containing sets of specharm data. Each element should represent a phoneme transition.
            
            epochs: number of epochs to use for training as Integer.

            logging: Flag indicating whether to write telemetry to a .csv log

            reset: Flag indicating whether the Ai should be reset before training
            
        Returns:
            None"""
        

        if reset:
            self.crfAi = SpecCrfAi(self.device, self.hparams["crf_lr"], self.hparams["crf_hlc"], self.hparams["crf_hls"], self.hparams["crf_reg"])
            self.crfAiOptimizer = torch.optim.NAdam(self.crfAi.parameters(), lr=self.crfAi.learningRate, weight_decay=self.crfAi.regularization)
        self.crfAi.train()
        if logging:
            csvFile = open(path.join(getenv("APPDATA"), "Nova-Vox", "Logs", "AI_train_crf.csv"), 'w', newline='')
            fieldnames = ["epochs", "learning rate", "hidden layer count", "loss", "acc. sample count", "wtd. train loss"]
            writer = DictWriter(csvFile, fieldnames)
        else:
            writer = None

        if (self.crfAi.epoch == 0) or self.crfAi.epoch == epochs:
            self.crfAi.epoch = epochs
        else:
            self.crfAi.epoch = None
        reportedLoss = 0.
        for epoch in tqdm(range(epochs), desc = "training", position = 0, unit = "epochs"):
            for data in tqdm(self.dataLoader(indata), desc = "epoch " + str(epoch), position = 1, total = len(indata), unit = "samples"):
                embedding1 = dec2bin(torch.tensor(data[1][0], device = self.device), 32)
                embedding2 = dec2bin(torch.tensor(data[1][1], device = self.device), 32)
                data = data[0].to(device = self.device)
                data = torch.squeeze(data)
                spectrum1 = data[2, 2 * halfHarms:]
                spectrum2 = data[3, 2 * halfHarms:]
                spectrum3 = data[-2, 2 * halfHarms:]
                spectrum4 = data[-1, 2 * halfHarms:]
                outputSize = data.size()[0] - 2
                factor = torch.linspace(0, 1, outputSize, device = self.device)
                output = self.crfAi(spectrum1, spectrum2, spectrum3, spectrum4, embedding1, embedding2, factor).transpose(0, 1)
                target = data[2:, 2 * halfHarms:]
                loss = self.criterion(output, target)
                self.crfAiOptimizer.zero_grad()
                loss.backward()
                self.crfAiOptimizer.step()
            tqdm.write('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss.data))
            self.crfAi.sampleCount += len(indata)
            reportedLoss = (reportedLoss * 99 + loss.data) / 100
            if writer != None:
                results = {
                    "epochs": epoch,
                    "learning rate": self.crfAi.learningRate,
                    "hidden layer count": self.crfAi.hiddenLayerCount,
                    "loss": loss.data,
                    "acc. sample count": self.crfAi.sampleCount,
                    "wtd. train loss": reportedLoss
                }
                writer.writerow(results)
        if writer != None:
            writer.close()
        criterion = torch.zeros((global_consts.halfTripleBatchSize + 1,), device = self.device)
        criterionSteps = 0
        with torch.no_grad():
            for data in self.dataLoader(indata):
                embedding1 = dec2bin(data[1][0].clone().to(self.device), 32)
                embedding2 = dec2bin(data[1][1].clone().to(self.device), 32)
                data = data[0].to(device = self.device)
                data = torch.squeeze(data)
                spectrum1 = data[2, 2 * halfHarms:]
                spectrum2 = data[3, 2 * halfHarms:]
                spectrum3 = data[-2, 2 * halfHarms:]
                spectrum4 = data[-1, 2 * halfHarms:]
                outputSize = data.size()[0] - 2
                factor = torch.linspace(0, 1, outputSize, device = self.device)
                output = self.crfAi(spectrum1, spectrum2, spectrum3, spectrum4, embedding1, embedding2, factor).transpose(0, 1)
                criterionA = torch.cat((torch.ones((outputSize, 1), device = self.device), output[:, 1:] / output[:, :-1]), 1)
                criterionB = torch.cat((output[:, :-1] / output[:, 1:], torch.ones((outputSize, 1), device = self.device)), 1)
                criterion += torch.mean(criterionA + criterionB, dim = 0)
                criterionSteps += 1
            criterion /= criterionSteps
            criterion = torch.less(criterion, torch.tensor([self.hparams["crf_def_thrh"],], device = self.device))
        self.defectiveCrfBins = criterion.to_sparse().coalesce().indices()
        print("defective Crf frequency bins:", self.defectiveCrfBins)
    
    def trainPred(self, indata, epochs:int=1, logging:bool = False, reset:bool = False) -> None:
        """trains the NN based on a dataset of specharm sequences"""

        scaler = torch.cuda.amp.GradScaler()
        if reset:
            self.predAi = SpecPredAi(self.device, self.hparams["pred_lr"], self.hparams["pred_rlc"], self.hparams["pred_rs"], self.hparams["pred_reg"])
            self.predAiDisc = SpecPredDiscriminator(device = self.device, learningRate=self.hparams["preddisc_lr"], regularization=self.hparams["preddisc_reg"], recSize=int(self.hparams["preddisc_rs"]), recLayerCount=int(self.hparams["preddisc_rlc"]))
            self.predAiGenerator = DataGenerator(self.voicebank, self.crfAi)
            self.predAiOptimizer = torch.optim.Adadelta(self.predAi.parameters(), lr=self.predAi.learningRate, weight_decay=self.predAi.regularization)
            self.predAiDiscOptimizer = torch.optim.Adadelta(self.predAiDisc.parameters(), lr=self.predAiDisc.learningRate, weight_decay=self.predAiDisc.regularization)
            self.VAE = VAE(device = self.device, latent_dim = self.hparams["latent_dim"])
        else:
            self.predAiGenerator.rebuildPool()
        self.predAi.train()
        self.predAi.requires_grad_(True)
        self.predAiDisc.train()
        self.predAiDisc.requires_grad_(True)
        self.VAE.train()
        self.VAE.requires_grad_(True)
        if logging:
            csvFile = open(path.join(getenv("APPDATA"), "Nova-Vox", "Logs", "AI_train_pred.csv"), 'w', newline='')
            fieldnames = ["epochs", "learning rate", "hidden layer count", "spec. loss", "harm. loss", "acc. sample count", "wtd. spec. train loss" "wtd. harm. train loss"]
            writer = DictWriter(csvFile, fieldnames)
        else:
            writer = None
        if (self.predAi.epoch == 0) or self.predAi.epoch == epochs:
            self.predAi.epoch = epochs
        else:
            self.predAi.epoch = None
        total = 0
        for phoneme in self.voicebank.phonemeDict.values():
            if torch.isnan(phoneme[0].avgSpecharm).any():
                continue
            self.deskewingPremul += phoneme[0].avgSpecharm.to(self.device)
            total += 1
        self.deskewingPremul /= total * 2.25
        targetLength = 0
        total = 0
        for data in self.dataLoader(indata):
            data = torch.squeeze(data)
            targetLength += data.size()[0]
            total += 1
        targetLength /= total
        
        self.trainVAE(indata, writer, epochs)

        for epoch in tqdm(range(epochs), desc = "training", position = 0, unit = "epochs"):
            for index, data in enumerate(tqdm(self.dataLoader(indata), desc = "epoch " + str(epoch), position = 1, total = len(indata), unit = "samples")):
                data = self.VAE.encoder(torch.squeeze(data))
                self.reset()
                synthBase = self.VAE.encoder(self.predAiGenerator.synthesize([0.2, 0.3, 0.4, 0.5], targetLength, 12))
                synthInput = self.predAi(synthBase, self.deskewingPremul, True)
                
                self.predAiDiscOptimizer.zero_grad()
                self.predAiDisc.resetState()
                posDiscriminatorLoss = self.predAiDisc(data, self.deskewingPremul, True)[-1]
                negDiscriminatorLoss = self.predAiDisc(synthInput.detach(), self.deskewingPremul, True)[-1]
                discriminatorLoss = posDiscriminatorLoss - negDiscriminatorLoss
                scaler.scale(discriminatorLoss).backward()
                scaler.step(self.predAiDiscOptimizer)
                
                if index % self.hparams["pred_train_asym"] == 0:
                    self.predAiOptimizer.zero_grad()
                    self.predAiDisc.resetState()
                    generatorLoss = self.predAiDisc(synthInput, self.deskewingPremul, True)[-1]
                    guideLoss = self.hparams["pred_guide_wgt"] * self.guideCriterion(synthBase, synthInput)
                    scaler.scale(generatorLoss + guideLoss).backward()
                    scaler.step(self.predAiOptimizer)

                tqdm.write("losses: pos.:{}, neg.:{}, disc.:{}, gen.:{}".format(posDiscriminatorLoss.data.__repr__(), negDiscriminatorLoss.data.__repr__(), discriminatorLoss.data.__repr__(), generatorLoss.data.__repr__()))
                
            tqdm.write('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, generatorLoss.data))
            self.predAi.sampleCount += len(indata)
            if writer != None:
                results = {
                    "epochs": epoch,
                    "learning rate": self.predAi.learningRate,
                    "hidden layer count": self.predAi.recLayerCount,
                    "gen. loss": generatorLoss.data,
                    "disc. loss": discriminatorLoss.data,
                    "acc. sample count": self.predAi.sampleCount,
                }
                writer.writerow(results)
        if writer != None:
            writer.close()

    def trainVAE(self, indata, writer, epochs:int=1) -> None:
        scaler = torch.cuda.amp.GradScaler()
        for epoch in tqdm(range(epochs), desc = "training", position = 0, unit = "epochs"):
            for index, data in enumerate(tqdm(self.dataLoader(indata), desc = "epoch " + str(epoch), position = 1, total = len(indata), unit = "samples")):
                data = torch.squeeze(data)
                self.VAEOptimizer.zero_grad()
                loss = self.VAE.training_step(data)
                scaler.scale(loss).backward()
                scaler.step(self.VAEOptimizer)
            if writer != None:
                results = {
                    "epochs": epoch,
                    "learning rate": self.predAi.learningRate,
                    "hidden layer count": self.predAi.recLayerCount,
                    "gen. loss": loss.data,
                    "disc. loss": "VAE",
                    "acc. sample count": self.predAi.sampleCount,
                }
                writer.writerow(results)
