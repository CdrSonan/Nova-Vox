#Copyright 2022 - 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import random
from math import floor, ceil, log
from copy import copy
from tkinter import Tk, filedialog

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import global_consts
from Backend.VB_Components.Ai.TrAi import TrAi
from Backend.VB_Components.Ai.Util import init_weights_logistic, init_weights_rectifier, init_weights_rectifier_leaky, norm_attention
from Backend.DataHandler.VocalSequence import VocalSequence
from Backend.DataHandler.VocalSegment import VocalSegment
from Backend.Resampler.Resamplers import getSpecharm
from Backend.Resampler.CubicSplineInter import interp
from Backend.Resampler.PhaseShift import phaseInterp
from Backend.ESPER.PitchCalculator import calculatePitch
from Backend.ESPER.SpectralCalculator import calculateSpectra
from Util import dec2bin

halfHarms = int(global_consts.nHarmonics / 2) + 1
input_dim = global_consts.halfTripleBatchSize + halfHarms + 1


class EncoderBlock(nn.Module):
    
    def __init__(self, dim:int, proj_dim:int, numLayers:int, attnExtension:int, device:torch.device) -> None:
        super().__init__()
        self.dim = dim
        self.proj_dim = proj_dim
        self.numLayers = numLayers
        self.attnExtension = attnExtension
        self.device = device
        self.cnn = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(self.dim, self.dim, 3, padding = 1, device = self.device),
                nn.Softplus()
              ) for _ in range(self.numLayers)],
        )
        self.norm = nn.LayerNorm([self.dim,], device = self.device, elementwise_affine = False)
        self.norm_proj = nn.LayerNorm([self.proj_dim,], device = self.device, elementwise_affine = False)
        if self.attnExtension is not None:
            self.nPosEmbeddings = ceil(log(2 * attnExtension + 5, 2))
            self.attention = nn.MultiheadAttention(embed_dim = self.proj_dim, kdim = self.dim + self.nPosEmbeddings, vdim = self.dim + self.nPosEmbeddings, num_heads = 4, dropout = 0.05, device = self.device)
        else:
            self.nPosEmbeddings = 0
        self.projector = nn.Linear((self.dim + self.nPosEmbeddings) * 5, self.proj_dim, device = self.device)
        self.resDropout = nn.Linear(self.dim, self.dim, device = self.device)#nn.Dropout(0.1)
        self.skip = nn.Linear(self.dim, self.dim, device = self.device)#nn.Dropout(0.1)
        self.apply(init_weights_rectifier_leaky)
        
    def forward(self, input:torch.Tensor) -> (torch.Tensor, torch.Tensor):
        posEmbeddings = torch.empty((input.size()[0], self.nPosEmbeddings), device = self.device)
        for i in range(self.nPosEmbeddings):
            posEmbeddings[:, i] = torch.arange(0, input.size()[0], device = self.device) % (2 ** (i + 1)) / (2 ** i)
        src = torch.cat((self.norm(self.cnn(input.transpose(0, 1)).transpose(0, 1) + self.skip(input)), posEmbeddings), 1)
        #src = torch.cat((self.cnn(input.transpose(0, 1)).transpose(0, 1), posEmbeddings), 1)
        tgt = self.projector(src.clone().reshape((int(src.size()[0] / 5), src.size()[1] * 5)))
        if self.attnExtension is None:
            return self.resDropout(src[:, :self.dim]), tgt
        mask = torch.ones((tgt.size()[0], src.size()[0]), device = self.device, dtype = torch.bool)
        for i in range(tgt.size()[0]):
            lower = max(0, i * 5 - self.attnExtension)
            upper = min(src.size()[0], (i + 1) * 5 + self.attnExtension)
            mask[i, lower:upper] = False
        attnOutput = self.attention(tgt, src, src, attn_mask = mask, need_weights = False)[0]
        return self.resDropout(src[:, :self.dim]), self.norm_proj(attnOutput)

class DecoderBlock(nn.Module):
    
    def __init__(self, dim:int, proj_dim:int, numLayers:int, attnExtension:int, device:torch.device) -> None:
        super().__init__()
        self.dim = dim
        self.proj_dim = proj_dim
        self.numLayers = numLayers
        self.attnExtension = attnExtension
        self.device = device
        self.cnn = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(self.dim, self.dim, 3, padding = 1, device = self.device),
                nn.Softplus(),
              ) for _ in range(self.numLayers)],
        )
        self.norm = nn.LayerNorm([self.dim,], device = self.device, elementwise_affine = False)
        self.norm_proj = nn.LayerNorm([self.dim,], device = self.device, elementwise_affine = False)
        if self.attnExtension is not None:
            self.nPosEmbeddings = ceil(log(2 * attnExtension + 5, 2))
            self.attention = nn.MultiheadAttention(embed_dim = self.dim, kdim = self.proj_dim + self.nPosEmbeddings, vdim = self.proj_dim + self.nPosEmbeddings, num_heads = 4, dropout = 0.05, device = self.device)
        else:
            self.nPosEmbeddings = 0
        self.projector = nn.Linear(self.proj_dim + self.nPosEmbeddings, self.dim * 5, device = self.device)
        self.skip = nn.Linear(self.dim, self.dim, device = self.device)#nn.Dropout(0.1)
        self.apply(init_weights_rectifier_leaky)
        
    def forward(self, input:torch.Tensor, residual:torch.Tensor) -> torch.Tensor:
        posEmbeddings = torch.empty((input.size()[0], self.nPosEmbeddings), device = self.device)
        for i in range(self.nPosEmbeddings):
            posEmbeddings[:, i] = torch.arange(0, input.size()[0], device = self.device) % (2 ** (i + 1)) / (2 ** i)
        src = torch.cat((input, posEmbeddings), 1)
        tgt = self.projector(src.clone()).reshape((-1, self.dim))
        if self.attnExtension is None:
            cnnInput = tgt + residual
        else:
            mask = torch.ones((tgt.size()[0], src.size()[0]), device = self.device, dtype = torch.bool)
            for i in range(src.size()[0]):
                lower = max(0, i * 5 - self.attnExtension)
                upper = min(tgt.size()[0], (i + 1) * 5 + self.attnExtension)
                mask[lower:upper, i] = False
            attnOutput = self.attention(tgt, src, src, attn_mask = mask, need_weights = False)[0]
            cnnInput = self.norm_proj(attnOutput) + residual
        return self.norm(self.cnn(cnnInput.transpose(0, 1)).transpose(0, 1) + self.skip(cnnInput))

class NormEncoderBlock(nn.Module):
    
    def __init__(self, dim:int, proj_dim:int, numLayers:int, attnExtension:int, device:torch.device) -> None:
        super().__init__()
        self.dim = dim
        self.proj_dim = proj_dim
        self.numLayers = numLayers
        self.attnExtension = attnExtension
        self.device = device
        self.cnn = nn.Sequential(
            *[nn.Sequential(
                nn.utils.parametrizations.spectral_norm(nn.Conv1d(self.dim, self.dim, 3, padding = 1, device = self.device)),
                nn.Tanh(),
              ) for i in range(self.numLayers)],
        )
        self.norm = nn.LayerNorm([self.dim,], device = self.device, elementwise_affine = False)
        self.norm_proj = nn.LayerNorm([self.proj_dim,], device = self.device, elementwise_affine = False)
        if self.attnExtension is not None:
            self.nPosEmbeddings = ceil(log(2 * attnExtension + 5, 2))
            self.attention = norm_attention(nn.MultiheadAttention(embed_dim = self.proj_dim, kdim = self.dim + self.nPosEmbeddings, vdim = self.dim + self.nPosEmbeddings, num_heads = 4, dropout = 0.05, device = self.device))
        else:
            self.nPosEmbeddings = 0
        self.projector = nn.utils.parametrizations.spectral_norm(nn.Linear((self.dim + self.nPosEmbeddings) * 5, self.proj_dim, device = self.device))
        self.resDropout = nn.Linear(self.dim, self.dim, device = self.device)#nn.Dropout(0.5)
        self.skip = nn.Linear(self.dim, self.dim, device = self.device)#nn.Dropout(0.5)
        self.apply(init_weights_logistic)
        
    def forward(self, input:torch.Tensor) -> (torch.Tensor, torch.Tensor):
        posEmbeddings = torch.empty((input.size()[0], self.nPosEmbeddings), device = self.device)
        for i in range(self.nPosEmbeddings):
            posEmbeddings[:, i] = torch.arange(0, input.size()[0], device = self.device) % (2 ** (i + 1)) / (2 ** i)
        src = torch.cat((self.norm(self.cnn(input.transpose(0, 1)).transpose(0, 1) + self.skip(input)), posEmbeddings), 1)
        #src = torch.cat((self.norm(self.cnn(input.transpose(0, 1)).transpose(0, 1)), posEmbeddings), 1)
        tgt = self.projector(src.clone().reshape((int(src.size()[0] / 5), src.size()[1] * 5)))
        if self.attnExtension is None:
            return self.resDropout(src[:, :self.dim]), tgt
        mask = torch.ones((tgt.size()[0], src.size()[0]), device = self.device, dtype = torch.bool)
        for i in range(tgt.size()[0]):
            lower = max(0, i * 5 - self.attnExtension)
            upper = min(src.size()[0], (i + 1) * 5 + self.attnExtension)
            mask[i, lower:upper] = False
        attnOutput = self.attention(tgt, src, src, attn_mask = mask, need_weights = False)[0]
        return self.resDropout(src[:, :self.dim]), self.norm_proj(attnOutput)

class NormDecoderBlock(nn.Module):
    
    def __init__(self, dim:int, proj_dim:int, numLayers:int, attnExtension:int, device:torch.device) -> None:
        super().__init__()
        self.dim = dim
        self.proj_dim = proj_dim
        self.numLayers = numLayers
        self.attnExtension = attnExtension
        self.device = device
        self.cnn = nn.Sequential(
            *[nn.Sequential(
                nn.utils.parametrizations.spectral_norm(nn.Conv1d(self.dim, self.dim, 3, padding = 1, device = self.device)),
                nn.Tanh(),
              ) for i in range(self.numLayers)],
        )
        self.norm = nn.LayerNorm([self.dim,], device = self.device, elementwise_affine = False)
        self.norm_proj = nn.LayerNorm([self.dim,], device = self.device, elementwise_affine = False)
        if self.attnExtension is not None:
            self.nPosEmbeddings = ceil(log(2 * attnExtension + 5, 2))
            self.attention = norm_attention(nn.MultiheadAttention(embed_dim = self.dim, kdim = self.proj_dim + self.nPosEmbeddings, vdim = self.proj_dim + self.nPosEmbeddings, num_heads = 4, dropout = 0.05, device = self.device))
        else:
            self.nPosEmbeddings = 0
        self.projector = nn.utils.parametrizations.spectral_norm(nn.Linear(self.proj_dim + self.nPosEmbeddings, self.dim * 5, device = self.device))
        self.skip = nn.Linear(self.dim, self.dim, device = self.device)#nn.Dropout(0.5)
        self.apply(init_weights_logistic)
        
    def forward(self, input:torch.Tensor, residual:torch.Tensor) -> torch.Tensor:
        posEmbeddings = torch.empty((input.size()[0], self.nPosEmbeddings), device = self.device)
        for i in range(self.nPosEmbeddings):
            posEmbeddings[:, i] = torch.arange(0, input.size()[0], device = self.device) % (2 ** (i + 1)) / (2 ** i)
        src = torch.cat((input, posEmbeddings), 1)
        tgt = self.projector(src.clone()).reshape((-1, self.dim))
        if self.attnExtension is None:
            cnnInput = tgt + residual
        else:
            mask = torch.ones((tgt.size()[0], src.size()[0]), device = self.device, dtype = torch.bool)
            for i in range(src.size()[0]):
                lower = max(0, i * 5 - self.attnExtension)
                upper = min(tgt.size()[0], (i + 1) * 5 + self.attnExtension)
                mask[lower:upper, i] = False
            attnOutput = self.attention(tgt, src, src, attn_mask = mask, need_weights = False)[0]
            cnnInput = self.norm_proj(attnOutput) + residual
        #return self.norm(self.cnn(cnnInput.transpose(0, 1)).transpose(0, 1))
        return self.norm(self.cnn(cnnInput.transpose(0, 1)).transpose(0, 1) + self.skip(cnnInput))


class MainAi(nn.Module):
    """Class for the Ai postprocessing/spectral prediction component.
    
    Methods:
        forward: processes a spectrum tensor, updating the internal states and returning the predicted next spectrum
        
        resetState: resets the hidden states and cell states of the internal LSTM layers"""


    def __init__(self, dim:int, embedDim:int, blockA:list, blockB:list, blockC:list, device:torch.device = None, learningRate:float=5e-5, regularization:float=1e-5, dropout:float=0.05, compile:bool = True) -> None:
        """basic constructor accepting the learning rate and other hyperparameters as input"""

        super().__init__()
        
        self.baseEncoder = nn.Sequential(
            nn.Linear(input_dim + embedDim, dim * 2, device = device),
            nn.Softplus(),
            nn.Linear(dim * 2, dim, device = device),
            nn.Softplus()
        )
        self.baseDecoder = nn.Sequential(
            nn.Linear(dim, dim * 2, device = device),
            nn.Softplus(),
            nn.Linear(dim * 2, input_dim, device = device),
            nn.Softplus()
        )
        
        self.baseResidual = nn.Dropout(0.1)
        
        self.encoderA = EncoderBlock(dim, 2 * dim, blockA[0], blockA[1], device)
        
        self.decoderA = DecoderBlock(dim, 2 * dim, blockA[0], blockA[1], device)
        
        self.encoderB = EncoderBlock(2 * dim, 3 * dim, blockB[0], blockB[1], device)
        
        self.decoderB = DecoderBlock(2 * dim, 3 * dim, blockB[0], blockB[1], device)
        
        self.encoderC = EncoderBlock(3 * dim, 4 * dim, blockC[0], blockC[1], device)
        
        self.decoderC = DecoderBlock(3 * dim, 4 * dim, blockC[0], blockC[1], device)
        
        self.baseEncoder.apply(init_weights_rectifier_leaky)
        self.baseDecoder.apply(init_weights_rectifier_leaky)

        self.device = device
        self.learningRate = learningRate
        self.dim = dim
        self.blockAHParams = blockA
        self.blockBHParams = blockB
        self.blockCHParams = blockC
        self.regularization = regularization
        self.epoch = 0
        self.sampleCount = 0
    
    def __new__(cls, dim:int, embedDim:int, blockA:list, blockB:list, blockC:list, device:torch.device = None, learningRate:float=5e-5, regularization:float=1e-5, dropout:float=0.05, compile:bool = False):
        instance = super().__new__(cls)
        if compile:
            instance = torch.compile(instance, dynamic = True, mode = "reduce-overhead")
        return instance

    def forward(self, input:torch.Tensor, level:int, embedding:torch.Tensor) -> torch.Tensor:
        """forward pass through the entire NN, aiming to predict the next spectrum in a sequence"""

        latent = self.baseEncoder(torch.cat((input, embedding.unsqueeze(0).tile((input.size()[0], 1))), 1))
        
        if latent.size()[0] % 125 != 0:
            padded = torch.cat((latent, torch.zeros((125 - latent.size()[0] % 125, *latent.size()[1:]), device = self.device, dtype = latent.dtype)), 0)
        else:
            padded = latent

        if level > 0:
            resA, encA = self.encoderA(padded)
            if level > 1:
                resB, encB = self.encoderB(encA)
                if level > 2:
                    resC, encC = self.encoderC(encB)
                    if level == 3:
                        encC = torch.zeros_like(encC)
                    decC = self.decoderC(encC, resC)
                else:
                    decC = torch.zeros_like(encB)
                decB = self.decoderB(decC, resB)
            else:
                decB = torch.zeros_like(encA)
            decA = self.decoderA(decB, resA)
        else:
            decA = torch.zeros_like(padded)
        
        return self.baseDecoder(decA[:latent.size()[0]] + self.baseResidual(latent)) + input

    def resetState(self) -> None:
        """resets the hidden states and cell states of the LSTM layers. Should be called between training or inference runs."""

        pass

class MainCritic(nn.Module):
    
    def __init__(self, dim:int, embedDim:int, blockA:list, blockB:list, blockC:list, outputWeight:int = 0.9, device:torch.device = None, learningRate:float=5e-5, regularization:float=1e-5, dropout:float=0.05, compile:bool = True) -> None:
        super().__init__()
        
        self.baseEncoder = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Linear(input_dim + embedDim, dim * 2, device = device)),
            nn.LeakyReLU(),
            nn.utils.parametrizations.spectral_norm(nn.Linear(dim * 2, dim, device = device)),
            nn.LeakyReLU()
        )
        self.baseDecoder = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Linear(dim, dim * 2, device = device)),
            nn.LeakyReLU(),
            nn.utils.parametrizations.spectral_norm(nn.Linear(dim * 2, dim, device = device)),
            nn.LeakyReLU()
        )
        
        self.baseResidual = nn.Dropout(0.1)
        
        self.encoderA = NormEncoderBlock(dim, 2 * dim, blockA[0], blockA[1], device)
        
        self.decoderA = NormDecoderBlock(dim, 2 * dim, blockA[0], blockA[1], device)
        
        self.encoderB = NormEncoderBlock(2 * dim, 3 * dim, blockB[0], blockB[1], device)
        
        self.decoderB = NormDecoderBlock(2 * dim, 3 * dim, blockB[0], blockB[1], device)
        
        self.encoderC = NormEncoderBlock(3 * dim, 4 * dim, blockC[0], blockC[1], device)
        
        self.decoderC = NormDecoderBlock(3 * dim, 4 * dim, blockC[0], blockC[1], device)
        
        self.final = nn.utils.parametrizations.spectral_norm(nn.Linear(dim, 1, bias = False, device = device))
        
        self.baseEncoder.apply(init_weights_rectifier_leaky)
        self.baseDecoder.apply(init_weights_rectifier_leaky)
        self.final.apply(init_weights_rectifier)

        self.device = device
        self.learningRate = learningRate
        self.dim = dim
        self.blockAHParams = blockA
        self.blockBHParams = blockB
        self.blockCHParams = blockC
        self.regularization = regularization
        self.outputWeight = outputWeight
        self.epoch = 0
        self.sampleCount = 0
        
    def __new__(cls, dim:int, embedDim:int, blockA:list, blockB:list, blockC:list, outputWeight:int = 0.9, device:torch.device = None, learningRate:float=5e-5, regularization:float=1e-5, dropout:float=0.05, compile:bool = False):
        instance = super().__new__(cls)
        if compile:
            instance = torch.compile(instance, dynamic = True, mode = "reduce-overhead")
        return instance

    def forward(self, input:torch.Tensor, level:int, embedding:torch.Tensor) -> torch.Tensor:
        """forward pass through the entire NN, aiming to predict the next spectrum in a sequence"""

        latent = self.baseEncoder(torch.cat((input, embedding.unsqueeze(0).tile((input.size()[0], 1))), 1))
        
        if latent.size()[0] % 125 != 0:
            padded = torch.cat((latent, torch.zeros((125 - latent.size()[0] % 125, *latent.size()[1:]), device = self.device)), 0)
        else:
            padded = latent

        if level > 0:
            resA, encA = self.encoderA(padded)
            if level > 1:
                resB, encB = self.encoderB(encA)
                if level > 2:
                    resC, encC = self.encoderC(encB)
                    if level == 3:
                        encC = torch.zeros_like(encC)
                    decC = self.decoderC(encC, resC)
                else:
                    decC = torch.zeros_like(encB)
                decB = self.decoderB(decC, resB)
            else:
                decB = torch.zeros_like(encA)
            decA = self.decoderA(decB, resA)
        else:
            decA = torch.zeros_like(padded)
        
        output = self.final(self.baseDecoder(decA[:latent.size()[0]] + self.baseResidual(latent)))
        return self.outputWeight * torch.max(output) + (1 - self.outputWeight) * torch.mean(output)

    def resetState(self) -> None:
        """resets the hidden states and cell states of the LSTM layers. Should be called between training or inference runs."""

        pass

class DataGenerator:
    """generates synthetic data for the discriminator to train on"""
    
    def __init__(self, voicebank, crfAi:TrAi, mode:str = "reclist") -> None:
        self.voicebank = voicebank
        self.crfAi = crfAi
        self.mode = mode
        self.pool = {}
        self.rebuildPool()

    def rebuildPool(self) -> None:
        if self.mode == "reclist (strict vowels)":
            self.pool["long"] = [key for key, value in self.voicebank.phonemeDict.keys() if value[0].isPlosive and value[0].isVoiced]
            self.pool["short"] = [key for key, value in self.voicebank.phonemeDict.items() if value[0].isPlosive or not value[0].isVoiced]
        elif self.mode == "dataset file":
            tkui = Tk()
            tkui.withdraw()
            path = filedialog.askopenfilename(title = "Select dataset file", filetypes = (("Dataset files", "*.dataset"), ("All files", "*.*")))
            tkui.destroy()
            data = torch.load(path)
            for i in range(len(data)):
                calculatePitch(data[i], False)
                calculateSpectra(data[i], False)
                data[i] = data[i].specharm + data[i].avgSpecharm
            self.pool["dataset"] = DataLoader(data, batch_size = 1, shuffle = True)
        else:
            self.pool["long"] = [key for key, value in self.voicebank.phonemeDict.items() if not value[0].isPlosive]
            self.pool["short"] = [key for key, value in self.voicebank.phonemeDict.items() if value[0].isPlosive]

    def makeSequence(self, noise:list, targetLength:int = None, phonemeLength:int = None, expression:str = "") -> torch.Tensor:
        if self.mode in ["reclist", "reclist (strict vowels)"]:
            if phonemeLength is None:
                raise ValueError("Length must be specified for reclist mode")
            longPhonemes = [random.choice(self.pool["long"]) for _ in range(ceil(phonemeLength / 2))]
            shortPhoneme = random.choice(self.pool["short"])
            phonemeSequence = []
            i = phonemeLength - 1
            while i > 0:
                if i % 2 == 0:
                    phonemeSequence.append(shortPhoneme)
                else:
                    phonemeSequence.append(longPhonemes[floor(i / 2)])
                i -= 1
        elif self.mode in ["dictionary", "dictionary(syllables)"]:
            phonemeSequence = random.choice(self.voicebank.wordDict[0].values())
        else:
            raise ValueError("Invalid mode for Data Generator")
        def stripPhoneme(phoneme:str) -> str:
            phoneme = phoneme.split("_")
            if len(phoneme) == 1:
                return phoneme[0]
            return "_".join(phoneme[:-1])
        phonemeSequence = [stripPhoneme(phoneme) + "_" + expression if stripPhoneme(phoneme) + "_" + expression in self.voicebank.phonemeDict else (stripPhoneme(phoneme) if stripPhoneme(phoneme) in self.voicebank.phonemeDict else phoneme) for phoneme in phonemeSequence]
        idx = [random.randint(0, len(self.voicebank.phonemeDict[i]) - 1) for i in phonemeSequence]
        embeddings = [self.voicebank.phonemeDict[phoneme][idx[i]].embedding for i, phoneme in enumerate(phonemeSequence)]
        effectiveLength = targetLength - sum([self.voicebank.phonemeDict[phoneme][idx[i]].specharm.size()[0] for i, phoneme in enumerate(phonemeSequence) if phoneme in self.pool["short"]])
        if effectiveLength >= 0:
            shortMultiplier = 1
            longMultiplier = effectiveLength / sum([self.voicebank.phonemeDict[phoneme][idx[i]].specharm.size()[0] for i, phoneme in enumerate(phonemeSequence) if phoneme in self.pool["long"]])
        else:
            shortMultiplier = longMultiplier = targetLength / sum([self.voicebank.phonemeDict[phoneme][idx[i]].specharm.size()[0] for i, phoneme in enumerate(phonemeSequence)])
        phonemes = [self.voicebank.phonemeDict[phoneme][idx[i]] for i, phoneme in enumerate(phonemeSequence)]
        borders = [0, 25]
        for i, phoneme in enumerate(phonemes):
            if phonemeSequence[i] in self.pool["short"]:
                length = phoneme.specharm.size()[0] * shortMultiplier
            else:
                length = phoneme.specharm.size()[0] * longMultiplier
            borders.append(borders[-1] + min(0.2 * length, 25))
            borders.append(borders[-2] + max(0.8 * length, length - 25))
            borders.append(borders[-3] + length)
        borders.append(borders[-1] + 25)
        borders = [i + random.normalvariate(0, noise[0] * 25) for i in borders]
        borders = [int(i * targetLength / borders[-1]) for i in borders]
        if borders[0] < 0:
            borders[0] = 0
        for i in range(1, len(borders)):
            if borders[i] <= borders[i - 1]:
                borders[i] = borders[i - 1] + 5
        initialBorder = copy(borders[0])
        for i in range(len(borders)):
            borders[i] -= initialBorder
        borderLength = borders[-1] - borders[0]
        sequence = VocalSequence(borderLength,
                                 borders,
                                 phonemeSequence,
                                 torch.tensor([random.uniform(0, 1) * noise[1] for _ in range(len(phonemeSequence))], device = torch.device("cpu")),
                                 torch.tensor([random.uniform(0, 1) * noise[1] + 0.5 for _ in range(len(phonemeSequence))], device = torch.device("cpu")),
                                 torch.full((borderLength,), 300.5, device = torch.device("cpu")),#pitch
                                 torch.full((borderLength,), random.uniform(-1, 1) * noise[2], device = torch.device("cpu")),#steadiness
                                 torch.full((borderLength,), random.uniform(-1, 1) * noise[3], device = torch.device("cpu")),#breathiness
                                 torch.zeros((borderLength,), device = torch.device("cpu")),#AI balance
                                 torch.zeros((borderLength,), device = torch.device("cpu")),#vibrato speed
                                 torch.full((borderLength,), -1, device = torch.device("cpu")),#vibrato strength
                                 True,
                                 True,
                                 False,
                                 False,
                                 True,
                                 [],
                                 None
        )
        return sequence, embeddings

    def synthesize(self, noise:list, length:int, phonemeLength:int = None, expression:str = "") -> torch.Tensor:
        if self.mode == "dataset file":
            return next(iter(self.pool["dataset"]))
        """noise mappings: [borders, offsets/spacing, steadiness, breathiness]"""
        sequence, embeddings = self.makeSequence(noise, length, phonemeLength, expression)
        output = torch.zeros([sequence.length, global_consts.halfTripleBatchSize + global_consts.nHarmonics + 3], device = torch.device("cpu"))
        output[sequence.borders[0]:sequence.borders[3]] = getSpecharm(VocalSegment(sequence, self.voicebank, 0, torch.device("cpu")), torch.device("cpu"))
        for i in range(1, sequence.phonemeLength - 1):
            output[sequence.borders[3*i+2]:sequence.borders[3*i+3]] = getSpecharm(VocalSegment(sequence, self.voicebank, i, torch.device("cpu")), torch.device("cpu"))
        output[sequence.borders[-4]:sequence.borders[-1]] = getSpecharm(VocalSegment(sequence, self.voicebank, sequence.phonemeLength - 1, torch.device("cpu")), torch.device("cpu"))
        output = output.to(self.crfAi.device)
        for i in range(1, sequence.phonemeLength):
            output[sequence.borders[3*i]:sequence.borders[3*i+2]] = self.crfWrapper(output[sequence.borders[3*i] - 1],
                                                                               output[sequence.borders[3*i]],
                                                                               output[sequence.borders[3*i+2]],
                                                                               output[sequence.borders[3*i+2]+1],
                                                                               embeddings[i - 1],
                                                                               embeddings[i],
                                                                               sequence.borders[3*i+2] - sequence.borders[3*i],
                                                                               sequence.pitch[sequence.borders[3*i]:sequence.borders[3*i+2]],
                                                                               (sequence.borders[3*i+1] - sequence.borders[3*i])/(sequence.borders[3*i+2] - sequence.borders[3*i]))
        return output
    
    def crfWrapper(self, specharm1:torch.Tensor, specharm2:torch.Tensor, specharm3:torch.Tensor, specharm4:torch.Tensor, embedding1:torch.Tensor, embedding2:torch.Tensor, outputSize:int, pitchCurve:torch.Tensor, slopeFactor:int):
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
        factor = log(0.5, slopeFactor / outputSize)
        factor = torch.pow(torch.linspace(0, 1, outputSize, device = self.crfAi.device), factor)
        embedding1 = dec2bin(torch.tensor(embedding1, device = self.crfAi.device), 32)
        embedding2 = dec2bin(torch.tensor(embedding2, device = self.crfAi.device), 32)
        spectrum = torch.squeeze(self.crfAi(spectrum1, spectrum2, spectrum3, spectrum4, embedding1, embedding2, factor)).transpose(0, 1)
        #for i in self.defectiveCrfBins:
        #    spectrum[:, i] = torch.mean(torch.cat((spectrum[:, i - 1].unsqueeze(1), spectrum[:, i + 1].unsqueeze(1)), 1), 1)
        borderRange = torch.zeros((outputSize,), device = self.crfAi.device)
        borderLimit = min(global_consts.crfBorderAbs, ceil(outputSize * global_consts.crfBorderRel))
        borderRange[:borderLimit] = torch.linspace(1, 0, borderLimit, device = self.crfAi.device)
        spectrum *= (1. - borderRange.unsqueeze(1))
        spectrum += torch.matmul(borderRange.unsqueeze(1), ((spectrum1 + spectrum2) / 2).unsqueeze(0))
        borderRange = torch.flip(borderRange, (0,))
        spectrum *= (1. - borderRange.unsqueeze(1))
        spectrum += torch.matmul(borderRange.unsqueeze(1), ((spectrum3 + spectrum4) / 2).unsqueeze(0))
        phases = torch.empty(outputSize, phase1.size()[0], device = self.crfAi.device)
        nativePitch = ceil(global_consts.tripleBatchSize / pitchCurve[0])
        originSpace = torch.min(torch.linspace(nativePitch, int(global_consts.nHarmonics / 2) * nativePitch, int(global_consts.nHarmonics / 2) + 1, device = self.crfAi.device), torch.tensor([global_consts.halfTripleBatchSize,], device = self.crfAi.device))
        factors = interp(torch.linspace(0, global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = self.crfAi.device), (spectrum1 + spectrum2) / 2, originSpace)
        harmsStart = (harm1 + harm2) * 0.5 / factors
        nativePitch = ceil(global_consts.tripleBatchSize / pitchCurve[-1])
        originSpace = torch.min(torch.linspace(nativePitch, int(global_consts.nHarmonics / 2) * nativePitch, int(global_consts.nHarmonics / 2) + 1, device = self.crfAi.device), torch.tensor([global_consts.halfTripleBatchSize,], device = self.crfAi.device))
        factors = interp(torch.linspace(0, global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = self.crfAi.device), (spectrum3 + spectrum4) / 2, originSpace)
        harmsEnd = (harm3 + harm4) * 0.5 / factors
        harms = torch.empty((outputSize, halfHarms), device = self.crfAi.device)
        harmLimit = torch.max(torch.cat((harm1.unsqueeze(1), harm2.unsqueeze(1), harm3.unsqueeze(1), harm4.unsqueeze(1)), 1))
        for i in range(outputSize):
            phases[i] = phaseInterp(phaseInterp(phase1, phase2, 0.5), phaseInterp(phase3, phase4, 0.5), i / (outputSize - 1))
            nativePitch = ceil(global_consts.tripleBatchSize / pitchCurve[i])
            originSpace = torch.min(torch.linspace(nativePitch, int(global_consts.nHarmonics / 2) * nativePitch, int(global_consts.nHarmonics / 2) + 1, device = self.crfAi.device), torch.tensor([global_consts.halfTripleBatchSize,], device = self.crfAi.device))
            factors = interp(torch.linspace(0, global_consts.halfTripleBatchSize, global_consts.halfTripleBatchSize + 1, device = self.crfAi.device), spectrum[i], originSpace)
            harms[i] = ((1. - (i + 1) / (outputSize + 1)) * harmsStart + (i + 1) / (outputSize + 1) * harmsEnd) * factors
            harms[i] = torch.min(harms[i], harmLimit)
            harms[i] = torch.max(harms[i], torch.tensor([0.,], device = self.crfAi.device))
        output = torch.cat((harms, phases, spectrum), 1)
        return output