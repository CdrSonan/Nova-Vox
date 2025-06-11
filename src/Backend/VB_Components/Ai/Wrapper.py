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
            "latent_dim": 512,
            "main_blkA": [256, 192],
            "main_blkB": [256, 256],
            "main_blkC": [256, 256],
            "main_lr": 0.0001,
            "main_reg": 0., #prev. 0.00001.
            "main_drp":0.5,
            "crt_blkA": [256, 192],
            "crt_blkB": [256, 256],
            "crt_blkC": [256, 256],
            "crt_out_wgt": 0.1,
            "crt_lr": 0.0001,
            "crt_reg": 0., #prev. 0.00001,
            "crt_drp":0.1,
            "gan_guide_wgt": 0.2,
            "gan_train_asym": 1,
            "fargan_interval": 50,
            "embeddingDim": 8,
        }
        if hparams:
            for i in hparams.keys():
                self.hparams[i] = hparams[i]
        self.voicebank = voicebank
        self.device = device
        self.final = False
        self.defectiveTrBins = []
        self.trAi = TrAi()
        self.mainAi = MainAi(device = self.device, dim = self.hparams["latent_dim"], embedDim = self.hparams["embeddingDim"], blockA = self.hparams["main_blkA"], blockB = self.hparams["main_blkB"], blockC = self.hparams["main_blkC"], learningRate=self.hparams["main_lr"], regularization=self.hparams["main_reg"], dropout = self.hparams["main_drp"])
        self.mainEmbedding = {"": torch.zeros((self.hparams["embeddingDim"],), device = self.device)}
        if not self.inferOnly:
            self.mainCritic = MainCritic(device = self.device, dim = self.hparams["latent_dim"], embedDim = self.hparams["embeddingDim"], blockA = self.hparams["crt_blkA"], blockB = self.hparams["crt_blkB"], blockC = self.hparams["crt_blkC"], outputWeight = self.hparams["crt_out_wgt"], learningRate=self.hparams["crt_lr"], regularization=self.hparams["crt_reg"], dropout = self.hparams["crt_drp"])
            self.mainGenerator = DataGenerator(self.voicebank, self.trAi)
            """self.mainAiOptimizer = [torch.optim.AdamW([*self.mainAi.baseEncoder.parameters(), *self.mainAi.baseDecoder.parameters()], lr=self.mainAi.learningRate, weight_decay=self.mainAi.regularization),
                                    torch.optim.AdamW([*self.mainAi.encoderA.parameters(), *self.mainAi.decoderA.parameters()], lr=self.mainAi.learningRate, weight_decay=self.mainAi.regularization),
                                    torch.optim.AdamW([*self.mainAi.encoderB.parameters(), *self.mainAi.decoderB.parameters()], lr=self.mainAi.learningRate * 4, weight_decay=self.mainAi.regularization),
                                    torch.optim.AdamW([*self.mainAi.encoderC.parameters(), *self.mainAi.decoderC.parameters()], lr=self.mainAi.learningRate * 16, weight_decay=self.mainAi.regularization)]
            self.mainCriticOptimizer = [torch.optim.AdamW([*self.mainCritic.baseEncoder.parameters(), *self.mainCritic.baseDecoder.parameters(), *self.mainCritic.final.parameters()], lr=self.mainCritic.learningRate, weight_decay=self.mainCritic.regularization),
                                        torch.optim.AdamW([*self.mainCritic.encoderA.parameters(), *self.mainCritic.decoderA.parameters()], lr=self.mainCritic.learningRate, weight_decay=self.mainCritic.regularization),
                                        torch.optim.AdamW([*self.mainCritic.encoderB.parameters(), *self.mainCritic.decoderB.parameters()], lr=self.mainCritic.learningRate * 4, weight_decay=self.mainCritic.regularization),
                                        torch.optim.AdamW([*self.mainCritic.encoderC.parameters(), *self.mainCritic.decoderC.parameters()], lr=self.mainCritic.learningRate * 16, weight_decay=self.mainCritic.regularization)]"""
            self.mainAiOptimizer = torch.optim.NAdam([*self.mainAi.parameters()], lr=self.mainAi.learningRate, weight_decay=self.mainAi.regularization, betas = (0.96, 0.999))
            self.mainCriticOptimizer = torch.optim.NAdam([*self.mainCritic.parameters()], lr=self.mainCritic.learningRate, weight_decay=self.mainCritic.regularization, betas = (0.96, 0.999))
            self.criterion = nn.MSELoss()
            self.guideCriterion = GuideRelLoss(device = self.device)
        self.deskewingPremul = torch.full((global_consts.frameSize,), 0.01, device = self.device)
    
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
            aiState = {
                'mainAi_epoch': self.mainAi.epoch,
                'mainAi_model_state_dict': self.mainAi.state_dict(),
                'mainAi_sampleCount': self.mainAi.sampleCount,
                'mainAI_embedding': self.mainEmbedding,
                'deskew_premul': self.deskewingPremul,
                'final': True
            }
        else:
            aiState = {
                'mainAi_epoch': self.mainAi.epoch,
                'mainAi_model_state_dict': self.mainAi.state_dict(),
                'mainCritic_model_state_dict': self.mainCritic.state_dict(),
                'mainAi_optimizer_state_dict': self.mainAiOptimizer.state_dict(),
                'mainCritic_optimizer_state_dict': self.mainCriticOptimizer.state_dict(),
                'mainAi_sampleCount': self.mainAi.sampleCount,
                'mainAI_embedding': self.mainEmbedding,
                'deskew_premul': self.deskewingPremul,
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
        if reset:
            self.mainAi = MainAi(device = self.device, dim = self.hparams["latent_dim"], embedDim = self.hparams["embeddingDim"], blockA = self.hparams["main_blkA"], blockB = self.hparams["main_blkB"], blockC = self.hparams["main_blkC"], learningRate=self.hparams["main_lr"], regularization=self.hparams["main_reg"], dropout = self.hparams["main_drp"])
            if not aiState["final"] and not self.inferOnly:
                self.mainCritic = MainCritic(device = self.device, dim = self.hparams["latent_dim"], embedDim = self.hparams["embeddingDim"], blockA = self.hparams["crt_blkA"], blockB = self.hparams["crt_blkB"], blockC = self.hparams["crt_blkC"], outputWeight = self.hparams["crt_out_wgt"], learningRate=self.hparams["crt_lr"], regularization=self.hparams["crt_reg"], dropout = self.hparams["crt_drp"])
                self.mainAiOptimizer = torch.optim.NAdam([*self.mainAi.parameters()], lr=self.mainAi.learningRate, weight_decay=self.mainAi.regularization, betas = (0.96, 0.999))
                self.mainCriticOptimizer = torch.optim.NAdam([*self.mainCritic.parameters()], lr=self.mainCritic.learningRate, weight_decay=self.mainCritic.regularization, betas = (0.96, 0.999))
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

        
        factor = math.log(0.5, slopeFactor / outputSize)
        factor = torch.pow(torch.linspace(0, 1, outputSize, device = self.device), factor)
        specharm1 = torch.clone(specharm1) / self.deskewingPremul
        specharm2 = torch.clone(specharm2) / self.deskewingPremul
        specharm3 = torch.clone(specharm3) / self.deskewingPremul
        specharm4 = torch.clone(specharm4) / self.deskewingPremul
        specharm = torch.squeeze(self.trAi(specharm1, specharm2, specharm3, specharm4, dec2bin(embedding1.to(self.device), 32), dec2bin(embedding2.to(self.device), 32), factor)).to(self.device)
        specharm *= self.deskewingPremul
        if expression1 == expression2:
            expression = expression1
        else:
            expression = ""
        refined = self.refine(specharm, expression)
        return specharm, torch.squeeze(refined)

    def refine(self, specharm:torch.Tensor, expression:str = "") -> torch.Tensor:
        """forward pass through the main Ai, taking a specharm as input and refining it to sound more natural. Includes data pre- and postprocessing."""

        self.mainAi.eval()
        self.mainAi.requires_grad_(False)
        latent = specharm / self.deskewingPremul
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        latent[:, :halfHarms] = torch.max(latent[:, :halfHarms], torch.zeros_like(latent[:, :halfHarms]))
        latent[:, :halfHarms] = torch.log(latent[:, :halfHarms] + 0.001)
        latent[:, 2 * halfHarms:] = torch.max(latent[:, 2 * halfHarms:], torch.zeros_like(latent[:, 2 * halfHarms:]))
        latent[:, 2 * halfHarms:] = torch.log(latent[:, 2 * halfHarms:] + 0.001)
        if expression not in self.mainEmbedding.keys():
            expression = ""
        refined = self.mainAi(latent, 4, self.mainEmbedding[expression])
        refined[:, :halfHarms] = torch.exp(refined[:, :halfHarms]) - 0.001
        refined[:, :halfHarms] = torch.max(refined[:, :halfHarms], torch.zeros_like(refined[:, :halfHarms]))
        refined[:, 2 * halfHarms:] = torch.exp(refined[:, 2 * halfHarms:]) - 0.001
        refined[:, 2 * halfHarms:] = torch.max(refined[:, 2 * halfHarms:], torch.zeros_like(refined[:, 2 * halfHarms:]))
        
        refined[:, halfHarms:2 * halfHarms] = torch.remainder(refined[:, halfHarms:2 * halfHarms], 2 * math.pi)
        refined[:, halfHarms:2 * halfHarms] = torch.where(refined[:, halfHarms:2 * halfHarms] > math.pi, refined[:, halfHarms:2 * halfHarms] - 2 * math.pi, refined[:, halfHarms:2 * halfHarms])
        return torch.squeeze(refined) * self.deskewingPremul

    def reset(self) -> None:
        """resets the hidden states and cell states of the AI's LSTM layers."""

        self.mainAi.resetState()
        if not self.inferOnly:
            self.mainCritic.resetState()

    def finalize(self):
        self.final = True
    
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
            self.mainAiOptimizer = torch.optim.NAdam([*self.mainAi.parameters()], lr=self.mainAi.learningRate, weight_decay=self.mainAi.regularization, betas = (0.96, 0.999))
            self.mainCriticOptimizer = torch.optim.NAdam([*self.mainCritic.parameters()], lr=self.mainCritic.learningRate, weight_decay=self.mainCritic.regularization, betas = (0.96, 0.999))
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
            fieldnames = ["pos.", "neg.", "disc.", "gen."]
            writer = DictWriter(csvFile, fieldnames)
        else:
            writer = None
        if (self.mainAi.epoch == 0) or self.mainAi.epoch == epochs:
            self.mainAi.epoch = epochs
        else:
            self.mainAi.epoch = None
        total = 0
        self.deskewingPremul = torch.full((global_consts.frameSize,), 0.01, device = self.device)
        for key in self.voicebank.phonemeDict.keys():
            for phoneme in self.voicebank.phonemeDict[key]:
                if torch.isnan(phoneme.avgSpecharm).any() or torch.isnan(phoneme.specharm).any():
                    continue
                self.deskewingPremul = torch.max(self.deskewingPremul, torch.cat((phoneme.avgSpecharm.to(self.device)[:global_consts.halfHarms], torch.ones([global_consts.halfHarms,]).to(self.device), phoneme.avgSpecharm.to(self.device)[global_consts.halfHarms:]), 0))
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
        
        for phaseIdx, phase in enumerate(phases):
            for epoch in tqdm(range(epochs), desc = "training phase " + str(phaseIdx + 1), position = 1, unit = "epochs"):
                for index, data in enumerate(tqdm(self.dataLoader(indata), desc = "epoch " + str(epoch), position = 0, total = len(indata), unit = "samples")):
                    avgSpecharm = torch.cat((data.avgSpecharm[:int(global_consts.nHarmonics / 2) + 1], torch.zeros([int(global_consts.nHarmonics / 2) + 1]), data.avgSpecharm[int(global_consts.nHarmonics / 2) + 1:]), 0)
                    key = data.key
                    data = (avgSpecharm + data.specharm).to(device = self.device)
                    if torch.isnan(data).any():
                        tqdm.write("NaN in data")
                        continue
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
                        embedding = torch.zeros_like(self.mainEmbedding[expression])
                        data = torch.squeeze(data)
                        data = self.mainGenerator.augment(data)
                        data /= self.deskewingPremul
                        data[:, :global_consts.halfHarms] = torch.max(data[:, :global_consts.halfHarms], torch.zeros_like(data[:, :global_consts.halfHarms]))
                        data[:, :global_consts.halfHarms] = torch.log(data[:, :global_consts.halfHarms] + 0.001)
                        data[:, global_consts.nHarmonics + 2:] = torch.max(data[:, global_consts.nHarmonics + 2:], torch.zeros_like(data[:, global_consts.nHarmonics + 2:]))
                        data[:, global_consts.nHarmonics + 2:] = torch.log(data[:, global_consts.nHarmonics + 2:] + 0.001)
                    self.reset()
                    synthBase = self.mainGenerator.synthesize([0.25, 0., 0., 0.], data.size()[0], 10, expression).to(device = self.device)
                    synthBase /= self.deskewingPremul
                    synthBase[:, :global_consts.halfHarms] = torch.max(synthBase[:, :global_consts.halfHarms], torch.zeros_like(synthBase[:, :global_consts.halfHarms]))
                    synthBase[:, :global_consts.halfHarms] = torch.log(synthBase[:, :global_consts.halfHarms] + 0.001)
                    synthBase[:, global_consts.nHarmonics + 2:] = torch.max(synthBase[:, global_consts.nHarmonics + 2:], torch.zeros_like(synthBase[:, global_consts.nHarmonics + 2:]))
                    synthBase[:, global_consts.nHarmonics + 2:] = torch.log(synthBase[:, global_consts.nHarmonics + 2:] + 0.001)
                    self.mainAi.resetState()
                    if synthBase.isnan().any():
                        tqdm.write("NaN encountered in synthetic sample")
                        synthBase = torch.where(synthBase.isnan(), torch.zeros_like(synthBase), synthBase)
                    synthInput = self.mainAi(synthBase, phase[0], embedding)
                    
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
                            "gen.": generatorLoss.data.item()
                        }
                        writer.writerow(results)
                    
                tqdm.write('epoch [{}/{}], loss:{}'.format(epoch + 1, epochs, generatorLoss.data.__repr__()))
                self.mainAi.sampleCount += len(indata)
        
        if writer != None:
            csvFile.close()
