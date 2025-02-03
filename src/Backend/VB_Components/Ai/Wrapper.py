#Copyright 2022 - 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import math
from statistics import mean
from os import path, getenv
from csv import DictWriter
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import global_consts
from Backend.VB_Components.Ai.TrAi import TrAi
from Backend.VB_Components.Ai.MainAi import MainAi, MainCritic, DataGenerator
from Backend.VB_Components.Ai.Util import gradientPenalty, GuideRelLoss, newEmbedding
from Backend.Resampler.PhaseShift import phaseInterp
from Backend.Resampler.CubicSplineInter import interp
from Util import dec2bin

halfHarms = int(global_consts.nHarmonics / 2) + 1

def dataLoader_collate(data):
    return data[0]
    """This is exactly as dumb as it looks."""

class AIWrapper():
    """Wrapper class for the mandatory AI components of a Voicebank. Controls data pre- and postprocessing, state loading and saving, Hyperparameters, and both training and inference."""

    def __init__(self, voicebank, device = torch.device("cpu"), hparams:dict = None, inferOnly:bool = False) -> None:
        """constructor taking a target device and dictionary of hyperparameters as input"""

        self.inferOnly = inferOnly
        self.hparams = {
            "tr_lr": 0.000055,
            "tr_reg": 0.,
            "tr_hlc": 1,
            "tr_hls": 4000,
            "tr_def_thrh" : 0.05,
            "latent_dim": 256,
            "main_blkA": [256, 192],
            "main_blkB": [256, 256],
            "main_blkC": [256, 256],
            "main_lr": 0.0012,
            "main_reg": 0.,
            "main_drp":0.5,
            "crt_blkA": [256, 192],
            "crt_blkB": [256, 256],
            "crt_blkC": [256, 256],
            "crt_out_wgt": 0.1,
            "crt_lr": 0.0008,
            "crt_reg": 0.001,
            "crt_drp":0.5,
            "gan_guide_wgt": 0.1,
            "gan_train_asym": 1,
            "fargan_interval": 10,
            "embeddingDim": 8,
        }
        if hparams:
            for i in hparams.keys():
                self.hparams[i] = hparams[i]
        self.voicebank = voicebank
        self.device = device
        self.final = False
        self.defectiveTrBins = []
        self.trAi = TrAi(device = self.device, learningRate=self.hparams["tr_lr"], regularization=self.hparams["tr_reg"], hiddenLayerCount=int(self.hparams["tr_hlc"]), hiddenLayerSize=int(self.hparams["tr_hls"]))
        self.mainAi = MainAi(device = self.device, dim = self.hparams["latent_dim"], embedDim = self.hparams["embeddingDim"], blockA = self.hparams["main_blkA"], blockB = self.hparams["main_blkB"], blockC = self.hparams["main_blkC"], learningRate=self.hparams["main_lr"], regularization=self.hparams["main_reg"], dropout = self.hparams["main_drp"])
        self.mainEmbedding = {"": torch.zeros((self.hparams["embeddingDim"],), device = self.device)}
        if not self.inferOnly:
            self.mainCritic = MainCritic(device = self.device, dim = self.hparams["latent_dim"], embedDim = self.hparams["embeddingDim"], blockA = self.hparams["crt_blkA"], blockB = self.hparams["crt_blkB"], blockC = self.hparams["crt_blkC"], outputWeight = self.hparams["crt_out_wgt"], learningRate=self.hparams["crt_lr"], regularization=self.hparams["crt_reg"], dropout = self.hparams["crt_drp"])
            self.mainGenerator = DataGenerator(self.voicebank, self.trAi)
            self.trAiOptimizer = torch.optim.NAdam(self.trAi.parameters(), lr=self.trAi.learningRate, weight_decay=self.trAi.regularization)
            """self.mainAiOptimizer = [torch.optim.AdamW([*self.mainAi.baseEncoder.parameters(), *self.mainAi.baseDecoder.parameters()], lr=self.mainAi.learningRate, weight_decay=self.mainAi.regularization),
                                    torch.optim.AdamW([*self.mainAi.encoderA.parameters(), *self.mainAi.decoderA.parameters()], lr=self.mainAi.learningRate, weight_decay=self.mainAi.regularization),
                                    torch.optim.AdamW([*self.mainAi.encoderB.parameters(), *self.mainAi.decoderB.parameters()], lr=self.mainAi.learningRate * 4, weight_decay=self.mainAi.regularization),
                                    torch.optim.AdamW([*self.mainAi.encoderC.parameters(), *self.mainAi.decoderC.parameters()], lr=self.mainAi.learningRate * 16, weight_decay=self.mainAi.regularization)]
            self.mainCriticOptimizer = [torch.optim.AdamW([*self.mainCritic.baseEncoder.parameters(), *self.mainCritic.baseDecoder.parameters(), *self.mainCritic.final.parameters()], lr=self.mainCritic.learningRate, weight_decay=self.mainCritic.regularization),
                                        torch.optim.AdamW([*self.mainCritic.encoderA.parameters(), *self.mainCritic.decoderA.parameters()], lr=self.mainCritic.learningRate, weight_decay=self.mainCritic.regularization),
                                        torch.optim.AdamW([*self.mainCritic.encoderB.parameters(), *self.mainCritic.decoderB.parameters()], lr=self.mainCritic.learningRate * 4, weight_decay=self.mainCritic.regularization),
                                        torch.optim.AdamW([*self.mainCritic.encoderC.parameters(), *self.mainCritic.decoderC.parameters()], lr=self.mainCritic.learningRate * 16, weight_decay=self.mainCritic.regularization)]"""
            self.mainAiOptimizer = torch.optim.NAdam([*self.mainAi.parameters()], lr=self.mainAi.learningRate, weight_decay=self.mainAi.regularization)
            self.mainCriticOptimizer = torch.optim.NAdam([*self.mainCritic.parameters()], lr=self.mainCritic.learningRate, weight_decay=self.mainCritic.regularization)
            self.criterion = nn.L1Loss()
            self.guideCriterion = GuideRelLoss(device = self.device, threshold = 0.6)
        self.deskewingPremul = torch.full((global_consts.halfTripleBatchSize + global_consts.halfHarms + 1,), 0.01, device = self.device)
    
    @staticmethod
    def dataLoader(data) -> DataLoader:
        """helper method for shuffled data loading from an arbitrary dataset.
        
        Arguments:
            data: Tensor, List or Iterable representing a dataset with several elements
            
        Returns:
            Iterable representing the same dataset, with the order of its elements shuffled"""
        
        return DataLoader(dataset=data, shuffle=True, collate_fn = dataLoader_collate, batch_size=1, num_workers=4)

    def getState(self) -> dict:
        """returns the state of the NN, its optimizer and their prerequisites as well as its epoch and sample count attributes in a Dictionary.
        
        Arguments:
            None
            
        Returns:
            Dictionary containing the NN's epoch attribute (epoch), weights (state dict), optimizer state (optimizer state dict) and sample count attribute (sampleCount)"""
            
        if self.final:
            aiState = {'trAi_epoch': self.trAi.epoch,
                'trAi_model_state_dict': self.trAi.state_dict(),
                'trAi_sampleCount': self.trAi.sampleCount,
                'mainAi_epoch': self.mainAi.epoch,
                'mainAi_model_state_dict': self.mainAi.state_dict(),
                'mainAi_sampleCount': self.mainAi.sampleCount,
                'mainAI_embedding': self.mainEmbedding,
                'deskew_premul': self.deskewingPremul,
                'defective_tr_bins': self.defectiveTrBins,
                'final': True
            }
        else:
            aiState = {'trAi_epoch': self.trAi.epoch,
                'trAi_model_state_dict': self.trAi.state_dict(),
                'trAi_optimizer_state_dict': self.trAiOptimizer.state_dict(),
                'trAi_sampleCount': self.trAi.sampleCount,
                'mainAi_epoch': self.mainAi.epoch,
                'mainAi_model_state_dict': self.mainAi.state_dict(),
                'mainCritic_model_state_dict': self.mainCritic.state_dict(),
                'mainAi_optimizer_state_dict': self.mainAiOptimizer.state_dict(),
                'mainCritic_optimizer_state_dict': self.mainCriticOptimizer.state_dict(),
                'mainAi_sampleCount': self.mainAi.sampleCount,
                'mainAI_embedding': self.mainEmbedding,
                'deskew_premul': self.deskewingPremul,
                'defective_tr_bins': self.defectiveTrBins,
                'final': False
            }
        return aiState

    def loadState(self, aiState:dict, mode:str = None, reset:bool=False) -> None:
        """loads the weights of the NNs managed by the wrapper from a dictionary, and reinitializes the NNs and/or their optimizers if required.
        
        Arguments:
            aiState: Dictionary in the same format as returned by getState(), containing all necessary information about the NNs
            
            mode: whether to load the weights for both NNs (None), only the phoneme transition Ai (tr), or only the main Ai (main)
            
            reset: indicates whether the NNs and their optimizers should be reset before applying changed weights to them. Must be True when the dictionary contains weights
            for a NN using different hyperparameters than the currently active one."""
        if (mode == None) or (mode == "tr"):
            if reset:
                self.trAi = TrAi(device = self.device, learningRate=self.hparams["tr_lr"], regularization=self.hparams["tr_reg"], hiddenLayerCount=int(self.hparams["tr_hlc"]), hiddenLayerSize=int(self.hparams["tr_hls"]))
                if not aiState["final"] and not self.inferOnly:
                    self.trAiOptimizer = torch.optim.NAdam(self.trAi.parameters(), lr=self.trAi.learningRate, weight_decay=self.trAi.regularization)
            self.trAi.epoch = aiState['trAi_epoch']
            self.trAi.sampleCount = aiState["trAi_sampleCount"]
            self.trAi.load_state_dict(aiState['trAi_model_state_dict'])
            if not aiState["final"] and not self.inferOnly:
                self.trAiOptimizer.load_state_dict(aiState['trAi_optimizer_state_dict'])
            self.defectiveTrBins = aiState["defective_tr_bins"]
        if (mode == None) or (mode == "main"):
            if reset:
                self.mainAi = MainAi(device = self.device, dim = self.hparams["latent_dim"], embedDim = self.hparams["embeddingDim"], blockA = self.hparams["main_blkA"], blockB = self.hparams["main_blkB"], blockC = self.hparams["main_blkC"], learningRate=self.hparams["main_lr"], regularization=self.hparams["main_reg"], dropout = self.hparams["main_drp"])
                if not aiState["final"] and not self.inferOnly:
                    self.mainCritic = MainCritic(device = self.device, dim = self.hparams["latent_dim"], embedDim = self.hparams["embeddingDim"], blockA = self.hparams["crt_blkA"], blockB = self.hparams["crt_blkB"], blockC = self.hparams["crt_blkC"], outputWeight = self.hparams["crt_out_wgt"], learningRate=self.hparams["crt_lr"], regularization=self.hparams["crt_reg"], dropout = self.hparams["crt_drp"])
                    self.mainAiOptimizer = torch.optim.NAdam([*self.mainAi.parameters()], lr=self.mainAi.learningRate, weight_decay=self.mainAi.regularization)
                    self.mainCriticOptimizer = torch.optim.NAdam([*self.mainCritic.parameters()], lr=self.mainCritic.learningRate, weight_decay=self.mainCritic.regularization)
            self.mainAi.epoch = aiState["mainAi_epoch"]
            self.mainAi.sampleCount = aiState["mainAi_sampleCount"]
            self.mainAi.load_state_dict(aiState['mainAi_model_state_dict'])
            self.mainEmbedding = aiState["mainAI_embedding"]
            for i in self.mainEmbedding.keys():
                self.mainEmbedding[i] = self.mainEmbedding[i].to(self.device)
            self.deskewingPremul = aiState["deskew_premul"]
            if not aiState["final"] and not self.inferOnly:
                self.mainCritic.load_state_dict(aiState['mainCritic_model_state_dict'])
                #self.mainAiOptimizer.load_state_dict(aiState['mainAi_optimizer_state_dict'])
                #self.mainCriticOptimizer.load_state_dict(aiState['mainCritic_optimizer_state_dict'])
        self.trAi.eval()
        self.mainAi.eval()

    def interpolate(self, specharm1:torch.Tensor, specharm2:torch.Tensor, specharm3:torch.Tensor, specharm4:torch.Tensor, embedding1:torch.Tensor, embedding2:torch.Tensor, expression1:str, expression2:str, outputSize:int, pitchCurve:torch.Tensor, slopeFactor:int) -> torch.Tensor:
        """forward pass of both NNs for generating a transition between two phonemes, with data pre- and postprocessing
        
        Arguments:
            specharm1-4: The four specharm Tensors to perform the interpolation between
            
            outputSize: length of the generated transition in engine ticks.

            pitchCurve: Tensor containing the pitch curve during the transition. Used to correctly calculate harmonic amplitudes during the transition.

            slopeFactor: integer expected to be between 0 and outputSize. Used for time remapping before the transition is calculated. Indicates where on the timeline 50% of the transition should have occured.
            
        Returns:
            tuple of two Tensor objects, containing the interpolated audio spectrum without and with the main Ai applied to it, respectively."""

        
        self.trAi.eval()
        self.trAi.requires_grad_(False)
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
        spectrum = torch.squeeze(self.trAi(spectrum1, spectrum2, spectrum3, spectrum4, dec2bin(embedding1.to(self.device), 32), dec2bin(embedding2.to(self.device), 32), factor)).transpose(0, 1)
        for i in self.defectiveTrBins:
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
        if expression1 == expression2:
            expression = expression1
        else:
            expression = ""
        refined = self.refine(output, expression)
        return output, torch.squeeze(refined)

    def refine(self, specharm:torch.Tensor, expression:str = "") -> torch.Tensor:
        """forward pass through the main Ai, taking a specharm as input and refining it to sound more natural. Includes data pre- and postprocessing."""

        self.mainAi.eval()
        self.mainAi.requires_grad_(False)
        phases = specharm[:, halfHarms:2 * halfHarms]
        remainder = torch.cat((specharm[:, :halfHarms], specharm[:, 2 * halfHarms:]), 1)
        latent = remainder / self.deskewingPremul
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if expression not in self.mainEmbedding.keys():
            expression = ""
        refined = self.mainAi(latent, 4, self.mainEmbedding[expression])
        output = torch.squeeze(refined) * self.deskewingPremul
        return torch.cat((output[:, :halfHarms], phases, output[:, halfHarms:]), 1)

    def reset(self) -> None:
        """resets the hidden states and cell states of the AI's LSTM layers."""

        self.mainAi.resetState()
        if not self.inferOnly:
            self.mainCritic.resetState()

    def finalize(self):
        self.final = True

    def trainTr(self, indata, epochs:int=1, logging:bool = False, reset:bool = False) -> None:
        """NN training with forward and backward passes, loss criterion and optimizer runs based on a dataset of spectral transition samples.
        
        Arguments:
            indata: Tensor, List or other Iterable containing sets of specharm data. Each element should represent a phoneme transition.
            
            epochs: number of epochs to use for training as Integer.

            logging: Flag indicating whether to write telemetry to a .csv log

            reset: Flag indicating whether the Ai should be reset before training
            
        Returns:
            None"""
        
        if self.inferOnly:
            raise Exception("Cannot start training since wrapper was initialized in inference-only mode")
        if reset:
            self.trAi = TrAi(device = self.device, learningRate=self.hparams["tr_lr"], regularization=self.hparams["tr_reg"], hiddenLayerCount=int(self.hparams["tr_hlc"]), hiddenLayerSize=int(self.hparams["tr_hls"]))
            self.trAiOptimizer = torch.optim.NAdam(self.trAi.parameters(), lr=self.trAi.learningRate, weight_decay=self.trAi.regularization)
        self.trAi.train()
        if logging:
            csvFile = open(path.join(getenv("APPDATA"), "Nova-Vox", "Logs", "AI_train_tr.csv"), 'w', newline='')
            fieldnames = ["epochs", "learning rate", "hidden layer count", "loss", "acc. sample count", "wtd. train loss"]
            writer = DictWriter(csvFile, fieldnames)
        else:
            writer = None

        if (self.trAi.epoch == 0) or self.trAi.epoch == epochs:
            self.trAi.epoch = epochs
        else:
            self.trAi.epoch = None
        reportedLoss = 0.
        loader = self.dataLoader(indata)
        for epoch in tqdm(range(epochs), desc = "training", position = 0, unit = "epochs"):
            for data in tqdm(loader, desc = "epoch " + str(epoch), position = 1, total = len(indata), unit = "samples"):
                avgSpecharm = torch.cat((data.avgSpecharm[:int(global_consts.nHarmonics / 2) + 1], torch.zeros([int(global_consts.nHarmonics / 2) + 1]), data.avgSpecharm[int(global_consts.nHarmonics / 2) + 1:]), 0)
                embedding1 = dec2bin(data.embedding[0].to(self.device), 32)
                embedding2 = dec2bin(data.embedding[1].to(self.device), 32)
                data = (avgSpecharm + data.specharm).to(device = self.device)
                data = torch.squeeze(data)
                spectrum1 = data[2, 2 * halfHarms:]
                spectrum2 = data[3, 2 * halfHarms:]
                spectrum3 = data[-2, 2 * halfHarms:]
                spectrum4 = data[-1, 2 * halfHarms:]
                outputSize = data.size()[0] - 2
                factor = torch.linspace(0, 1, outputSize, device = self.device)
                output = self.trAi(spectrum1, spectrum2, spectrum3, spectrum4, embedding1, embedding2, factor).transpose(0, 1)
                target = data[2:, 2 * halfHarms:]
                loss = self.criterion(output, target)
                self.trAiOptimizer.zero_grad()
                loss.backward()
                self.trAiOptimizer.step()
            tqdm.write('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss.data))
            self.trAi.sampleCount += len(indata)
            reportedLoss = (reportedLoss * 99 + loss.data) / 100
            if writer != None:
                results = {
                    "epochs": epoch,
                    "learning rate": self.trAi.learningRate,
                    "hidden layer count": self.trAi.hiddenLayerCount,
                    "loss": loss.data,
                    "acc. sample count": self.trAi.sampleCount,
                    "wtd. train loss": reportedLoss
                }
                writer.writerow(results)
        if writer != None:
            csvFile.close()
        criterion = torch.zeros((global_consts.halfTripleBatchSize + 1,), device = self.device)
        criterionSteps = 0
        with torch.no_grad():
            for data in loader:
                avgSpecharm = torch.cat((data.avgSpecharm[:int(global_consts.nHarmonics / 2) + 1], torch.zeros([int(global_consts.nHarmonics / 2) + 1]), data.avgSpecharm[int(global_consts.nHarmonics / 2) + 1:]), 0)
                embedding1 = dec2bin(data.embedding[0].clone().to(self.device), 32)
                embedding2 = dec2bin(data.embedding[1].clone().to(self.device), 32)
                data = (avgSpecharm + data.specharm).to(device = self.device)
                data = torch.squeeze(data)
                spectrum1 = data[2, 2 * halfHarms:]
                spectrum2 = data[3, 2 * halfHarms:]
                spectrum3 = data[-2, 2 * halfHarms:]
                spectrum4 = data[-1, 2 * halfHarms:]
                outputSize = data.size()[0] - 2
                factor = torch.linspace(0, 1, outputSize, device = self.device)
                output = self.trAi(spectrum1, spectrum2, spectrum3, spectrum4, embedding1, embedding2, factor).transpose(0, 1)
                criterionA = torch.cat((torch.ones((outputSize, 1), device = self.device), output[:, 1:] / output[:, :-1]), 1)
                criterionB = torch.cat((output[:, :-1] / output[:, 1:], torch.ones((outputSize, 1), device = self.device)), 1)
                criterion += torch.mean(criterionA + criterionB, dim = 0)
                criterionSteps += 1
            criterion /= criterionSteps
            criterion = torch.less(criterion, torch.tensor([self.hparams["tr_def_thrh"],], device = self.device))
        self.defectiveTrBins = criterion.to_sparse().coalesce().indices()
        print("defective Tr. frequency bins:", self.defectiveTrBins)
    
    def trainMain(self, indata, epochs:int=1, logging:bool = False, reset:bool = False, generatorMode:str = "reclist") -> None:
        """trains the NN based on a dataset of specharm sequences"""

        if self.inferOnly:
            raise Exception("Cannot start training since wrapper was initialized in inference-only mode")
        if reset:
            self.mainAi = MainAi(device = self.device, dim = self.hparams["latent_dim"], embedDim = self.hparams["embeddingDim"], blockA = self.hparams["main_blkA"], blockB = self.hparams["main_blkB"], blockC = self.hparams["main_blkC"], learningRate=self.hparams["main_lr"], regularization=self.hparams["main_reg"], dropout = self.hparams["main_drp"])
            self.mainCritic = MainCritic(device = self.device, dim = self.hparams["latent_dim"], embedDim = self.hparams["embeddingDim"], blockA = self.hparams["crt_blkA"], blockB = self.hparams["crt_blkB"], blockC = self.hparams["crt_blkC"], outputWeight = self.hparams["crt_out_wgt"], learningRate=self.hparams["crt_lr"], regularization=self.hparams["crt_reg"], dropout = self.hparams["crt_drp"])
            self.mainGenerator = DataGenerator(self.voicebank, self.trAi, mode = generatorMode)
            self.mainEmbedding = {"": torch.zeros((self.hparams["embeddingDim"],), device = self.device)}
            """self.mainAiOptimizer = [torch.optim.AdamW([*self.mainAi.baseEncoder.parameters(), *self.mainAi.baseDecoder.parameters()], lr=self.mainAi.learningRate, weight_decay=self.mainAi.regularization),
                                    torch.optim.AdamW([*self.mainAi.encoderA.parameters(), *self.mainAi.decoderA.parameters()], lr=self.mainAi.learningRate, weight_decay=self.mainAi.regularization),
                                    torch.optim.AdamW([*self.mainAi.encoderB.parameters(), *self.mainAi.decoderB.parameters()], lr=self.mainAi.learningRate * 4, weight_decay=self.mainAi.regularization),
                                    torch.optim.AdamW([*self.mainAi.encoderC.parameters(), *self.mainAi.decoderC.parameters()], lr=self.mainAi.learningRate * 16, weight_decay=self.mainAi.regularization)]
            self.mainCriticOptimizer = [torch.optim.AdamW([*self.mainCritic.baseEncoder.parameters(), *self.mainCritic.baseDecoder.parameters(), *self.mainCritic.final.parameters()], lr=self.mainCritic.learningRate, weight_decay=self.mainCritic.regularization),
                                        torch.optim.AdamW([*self.mainCritic.encoderA.parameters(), *self.mainCritic.decoderA.parameters()], lr=self.mainCritic.learningRate, weight_decay=self.mainCritic.regularization),
                                        torch.optim.AdamW([*self.mainCritic.encoderB.parameters(), *self.mainCritic.decoderB.parameters()], lr=self.mainCritic.learningRate * 4, weight_decay=self.mainCritic.regularization),
                                        torch.optim.AdamW([*self.mainCritic.encoderC.parameters(), *self.mainCritic.decoderC.parameters()], lr=self.mainCritic.learningRate * 16, weight_decay=self.mainCritic.regularization)]"""
            self.mainAiOptimizer = torch.optim.NAdam([*self.mainAi.parameters()], lr=self.mainAi.learningRate, weight_decay=self.mainAi.regularization)
            self.mainCriticOptimizer = torch.optim.NAdam([*self.mainCritic.parameters()], lr=self.mainCritic.learningRate, weight_decay=self.mainCritic.regularization)
        else:
            if self.mainGenerator.mode != generatorMode:
                self.mainGenerator.mode = generatorMode
            self.mainGenerator.rebuildPool()
        self.mainAi.train()
        self.mainAi.requires_grad_(True)
        self.mainCritic.train()
        self.mainCritic.requires_grad_(True)
        if logging:
            csvFile = open(path.join(getenv("APPDATA"), "Nova-Vox", "Logs", "AI_train_main.csv"), 'w', newline='')
            fieldnames = ["pos.", "neg.", "disc.", "gen.", "encA", "encB", "encC", "decA", "decB", "decC", "baseEnc", "baseDec", "CEncA", "CEncB", "CEncC", "CDecA", "CDecB", "CDecC", "CBaseEnc", "CBaseDec", "final"]
            writer = DictWriter(csvFile, fieldnames)
        else:
            writer = None
        if (self.mainAi.epoch == 0) or self.mainAi.epoch == epochs:
            self.mainAi.epoch = epochs
        else:
            self.mainAi.epoch = None
        total = 0
        self.deskewingPremul = torch.full((global_consts.halfTripleBatchSize + global_consts.halfHarms + 1,), 0.01, device = self.device)
        for key in self.voicebank.phonemeDict.keys():
            for phoneme in self.voicebank.phonemeDict[key]:
                self.deskewingPremul = torch.max(self.deskewingPremul, phoneme.avgSpecharm.to(self.device))
                total += 1
            expression = key.split("_")
            if len(expression) == 1:
                expression = ""
            else:
                expression = expression[-1]
            if expression not in self.mainEmbedding.keys():
                self.mainEmbedding[expression] = newEmbedding(len(self.mainEmbedding), self.hparams["embeddingDim"], self.device)
        
        fargan_smp = None
        fargan_embedding = None
        fargan_expression = None
        fargan_score = math.inf
        
        phases = [(6, "train"),]
                 #(1, "train"),(1, "train"),(1, "train"),(1, "train"),(1, "train"),(1, "train"),(1, "train"),(1, "train"),(1, "train"),(1, "train"),]
        
        #bceloss = nn.BCEWithLogitsLoss()
        
        for phaseIdx, phase in enumerate(phases):
            for epoch in tqdm(range(epochs), desc = "training phase " + str(phaseIdx + 1), position = 1, unit = "epochs"):
                for index, data in enumerate(tqdm(self.dataLoader(indata), desc = "epoch " + str(epoch), position = 0, total = len(indata), unit = "samples")):
                    avgSpecharm = torch.cat((data.avgSpecharm[:int(global_consts.nHarmonics / 2) + 1], torch.zeros([int(global_consts.nHarmonics / 2) + 1]), data.avgSpecharm[int(global_consts.nHarmonics / 2) + 1:]), 0)
                    key = data.key
                    data = (avgSpecharm + data.specharm).to(device = self.device)
                    if index % self.hparams["fargan_interval"] == self.hparams["fargan_interval"] - 1:
                        embedding = fargan_embedding
                        expression = fargan_expression
                        data = fargan_smp
                        fargan_score = math.inf
                    else:
                        expression = key.split("_")
                        if len(expression) == 1:
                            expression = ""
                        else:
                            expression = expression[-1]
                        if expression not in self.mainEmbedding.keys():
                            expression = ""
                        embedding = self.mainEmbedding[expression]
                        data = torch.squeeze(data)
                        data = torch.cat((data[:, :halfHarms], data[:, 2 * halfHarms:]), 1)
                        data /= self.deskewingPremul
                    self.reset()
                    synthBase = self.mainGenerator.synthesize([0.1, 0., 0., 0.], data.size()[0], 14, expression)
                    synthBase = torch.cat((synthBase[:, :halfHarms], synthBase[:, 2 * halfHarms:]), 1)
                    synthBase /= self.deskewingPremul
                    self.mainAi.resetState()
                    if synthBase.isnan().any():
                        print("NaN encountered in synthetic sample")
                        synthBase = torch.where(synthBase.isnan(), torch.zeros_like(synthBase), synthBase)
                    synthInput = self.mainAi(synthBase, phase[0], embedding)
                    print(torch.mean(data), torch.max(data))
                    print(torch.mean(synthBase), torch.max(synthBase))
                    print(torch.mean(synthInput), torch.max(synthInput))
                    print("\n\n\n")
                    
                    if phase[1] == "pretrain":
                        self.mainAi.zero_grad()
                        generatorLoss = torch.mean(torch.square(synthInput - synthBase))
                        generatorLoss.backward()
                        self.mainAiOptimizer.step()
                    elif index % self.hparams["gan_train_asym"] == 0:
                        self.mainAi.zero_grad()
                        self.mainCritic.zero_grad()
                        self.mainCritic.resetState()
                        generatorLoss = torch.square(self.mainCritic(synthInput, phase[0], embedding))
                        guideLoss = self.hparams["gan_guide_wgt"] * self.guideCriterion(synthInput, synthBase)
                        (generatorLoss + guideLoss).backward()
                        #generatorLoss.backward()
                        self.mainAiOptimizer.step()
                    
                    self.mainCritic.zero_grad()
                    self.mainCritic.resetState()
                    posDiscriminatorLoss = torch.square(self.mainCritic(data, phase[0], embedding))
                    negDiscriminatorLoss = torch.square(self.mainCritic(synthInput.detach(), phase[0], embedding) - 1)
                    discriminatorLoss = posDiscriminatorLoss + negDiscriminatorLoss
                    if phase[1] == "pretrain":
                        discriminatorLoss *= 0.1
                    discriminatorLoss.backward()
                    self.mainCriticOptimizer.step()
                    
                    if index % self.hparams["fargan_interval"] != self.hparams["fargan_interval"] - 1 and negDiscriminatorLoss < fargan_score:
                        fargan_smp = synthInput.detach().clone()
                        fargan_embedding = embedding.clone()
                        fargan_expression = expression
                        fargan_score = negDiscriminatorLoss

                    tqdm.write("losses: pos.:{}, neg.:{}, disc.:{}, gen.:{}".format(posDiscriminatorLoss.data.__repr__(), negDiscriminatorLoss.data.__repr__(), discriminatorLoss.data.__repr__(), generatorLoss.data.__repr__()))
                    if writer != None:
                        results = {
                            "pos.": posDiscriminatorLoss.data.item(),
                            "neg.": negDiscriminatorLoss.data.item(),
                            "disc.": discriminatorLoss.data.item(), 
                            "gen.": generatorLoss.data.item(),
                            "encA": mean([torch.mean(torch.abs(i.weight.grad)).item() for i in self.mainAi.encoderA.modules() if isinstance(i, nn.Linear)]) if phase[0] > 0 else 0.,
                            "encB": mean([torch.mean(torch.abs(i.weight.grad)).item() for i in self.mainAi.encoderB.modules() if isinstance(i, nn.Linear)]) if phase[0] > 1 else 0.,
                            "encC": mean([torch.mean(torch.abs(i.weight.grad)).item() for i in self.mainAi.encoderC.modules() if isinstance(i, nn.Linear)]) if phase[0] > 2 else 0.,
                            "decA": mean([torch.mean(torch.abs(i.weight.grad)).item() for i in self.mainAi.decoderA.modules() if isinstance(i, nn.Linear)]) if phase[0] > 3 else 0.,
                            "decB": mean([torch.mean(torch.abs(i.weight.grad)).item() for i in self.mainAi.decoderB.modules() if isinstance(i, nn.Linear)]) if phase[0] > 4 else 0.,
                            "decC": mean([torch.mean(torch.abs(i.weight.grad)).item() for i in self.mainAi.decoderC.modules() if isinstance(i, nn.Linear)]) if phase[0] > 5 else 0.,
                            "baseEnc": mean([torch.mean(torch.abs(i.weight.grad)).item() for i in self.mainAi.preNet.modules() if isinstance(i, nn.Linear)]),
                            "baseDec": mean([torch.mean(torch.abs(i.weight.grad)).item() for i in self.mainAi.postNet.modules() if isinstance(i, nn.Linear)]),
                            "CEncA": 0.,#mean([torch.mean(torch.abs(i.weight.grad)).item() for i in self.mainCritic.mainNet[0].modules() if isinstance(i, nn.Linear)]) if phase[0] > 0 else 0.,
                            "CEncB": 0.,#mean([torch.mean(torch.abs(i.weight.grad)).item() for i in self.mainCritic.mainNet[1].modules() if isinstance(i, nn.Linear)]) if phase[0] > 1 else 0.,
                            "CEncC": 0.,#mean([torch.mean(torch.abs(i.weight.grad)).item() for i in self.mainCritic.mainNet[2].modules() if isinstance(i, nn.Linear)]) if phase[0] > 2 else 0.,
                            "CDecA": 0.,#mean([torch.mean(torch.abs(i.weight.grad)).item() for i in self.mainCritic.mainNet[3].modules() if isinstance(i, nn.Linear)]) if phase[0] > 3 else 0.,
                            "CDecB": 0.,#mean([torch.mean(torch.abs(i.weight.grad)).item() for i in self.mainCritic.mainNet[4].modules() if isinstance(i, nn.Linear)]) if phase[0] > 4 else 0.,
                            "CDecC": 0.,#mean([torch.mean(torch.abs(i.weight.grad)).item() for i in self.mainCritic.mainNet[5].modules() if isinstance(i, nn.Linear)]) if phase[0] > 5 else 0.,
                            "CBaseEnc": 0.,#mean([torch.mean(torch.abs(i.weight.grad)).item() for i in self.mainCritic.preNet.modules() if isinstance(i, nn.Linear)]),
                            "CBaseDec": 0.,#mean([torch.mean(torch.abs(i.weight.grad)).item() for i in self.mainCritic.postNet.modules() if isinstance(i, nn.Linear)]),
                            "final": 0.,#torch.mean(torch.abs(self.mainCritic.final.parametrizations.weight.original.grad)).item()
                        }
                        writer.writerow(results)
                    
                tqdm.write('epoch [{}/{}], loss:{}'.format(epoch + 1, epochs, generatorLoss.data.__repr__()))
                self.mainAi.sampleCount += len(indata)
        
        if writer != None:
            csvFile.close()
