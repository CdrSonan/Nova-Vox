#Copyright 2022 - 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import random
from math import floor, ceil, log
from copy import copy
import ctypes
from tkinter import Tk, filedialog
from tqdm.auto import tqdm
import h5py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import global_consts
from Backend.VB_Components.Ai.TrAi import TrAi
from Backend.DataHandler.VocalSequence import VocalSequence
from Backend.DataHandler.VocalSegment import VocalSegment
from Backend.DataHandler.AudioSample import LiteSampleCollection
from Backend.DataHandler.HDF5 import SampleStorage
from Backend.Resampler.Resamplers import getSpecharm
from Backend.ESPER.SpectralCalculator import asyncProcess
from C_Bridge import esper
from Util import dec2bin

halfHarms = int(global_consts.nHarmonics / 2) + 1
input_dim = global_consts.frameSize

def dataLoader_collate(data):
    return data[0]
    """This is exactly as dumb as it looks."""

class DataGenerator(IterableDataset):
    """generates synthetic data for the discriminator to train on"""
    
    def __init__(self, voicebank, crfAi:TrAi, mode:str = "reclist") -> None:
        self.voicebank = voicebank
        self.crfAi = crfAi
        self.mode = mode
        self.pool = {}
        self.rebuildPool()
        self.savedParams = None
    
    def __iter__(self):
        if self.savedParams is not None:
            raise ValueError("Data Generation parameters have not been configured. Use setParams() method to set parameters, or use synthesize() method to generate a single sample.")
        return self
    
    def __next__(self):
        return self.synthesize(*self.savedParams)
    
    def setParams(self, noise:list, length:int, phonemeLength:int = None, expression:str = "") -> None:
        self.savedParams = [noise, length, phonemeLength, expression]

    def rebuildPool(self) -> None:
        if self.mode == "dataset file":
            tkui = Tk()
            tkui.withdraw()
            path = filedialog.askopenfilename(title = "Select dataset file", filetypes = (("HDF5 dataset files", "*.hdf5"), ("All files", "*.*")))
            tkui.destroy()
            with h5py.File(path, "r") as f:
                storage = SampleStorage(f, [], False)
                data = storage.toCollection("AI")
            dataset = LiteSampleCollection(None, False)
            inputQueue, outputQueue, processes = asyncProcess(False, True)
            expectedSamples = 0
            for i in tqdm(range(len(data)), desc = "preprocessing", unit = "samples"):
                samples = data[i].convert(True)
                for j in samples:
                    inputQueue.put(j)
                    expectedSamples += 1
            del data
            for _ in tqdm(range(expectedSamples), desc = "processing", unit = "samples"):
                dataset.append(outputQueue.get())
            for _ in processes:
                inputQueue.put(None)
            for process in processes:
                process.join()
            self.pool["dataset"] = DataLoader(dataset, batch_size = 1, shuffle = True, collate_fn = dataLoader_collate)
        elif self.mode == ".nvx file":
            self.pool["sequences"] = []
            tkui = Tk()
            tkui.withdraw()
            path = filedialog.askopenfilename(title = "Select .nvx file", filetypes = (("Nova-Vox project files", "*.nvx"), ("All files", "*.*")))
            tkui.destroy()
            file = h5py.File(path, "r")
            for group in file.values():
                phonemes = " ".join([phoneme.decode("utf-8") for phoneme in group["phonemes"]]).split(" ")
                if len(phonemes) < 2:
                    continue
                self.pool["sequences"].append(VocalSequence(len(phonemes),
                                                            torch.tensor(group["borders"]),
                                                            phonemes,
                                                            torch.tensor(group["loopOffset"]),
                                                            torch.tensor(group["loopOverlap"]),
                                                            torch.tensor(group["pitch"]),
                                                            torch.tensor(group["steadiness"]),
                                                            torch.tensor(group["breathiness"]),
                                                            torch.tensor(group["aiBalance"]),
                                                            torch.tensor(group["genderFactor"]),
                                                            torch.tensor(group["vibratoSpeed"]),
                                                            torch.tensor(group["vibratoStrength"]),
                                                            bool(group.attrs["useBreathiness"]),
                                                            bool(group.attrs["useSteadiness"]),
                                                            bool(group.attrs["useAIBalance"]),
                                                            bool(group.attrs["useGenderFactor"]),
                                                            bool(group.attrs["useVibratoSpeed"]),
                                                            bool(group.attrs["useVibratoStrength"]),
                                                            float(group.attrs["unvoicedShift"]),
                                                            [],
                                                            None))   
        elif self.mode == "reclist (strict vowels)":
            self.pool["long"] = [key for key, value in self.voicebank.phonemeDict.items() if not value[0].isPlosive and value[0].isVoiced]
            self.pool["short"] = [key for key, value in self.voicebank.phonemeDict.items() if value[0].isPlosive or not value[0].isVoiced]
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
        elif self.mode == ".nvx file":
            sequence = random.choice(self.pool["sequences"])
            start = random.randint(0, len(sequence.phonemes) - 2)
            end = start
            while end < len(sequence.phonemes) and sequence.borders[3 * end + 1] - sequence.borders[3 * start + 1] < targetLength:
                end += 1
            sequence = VocalSequence(sequence.borders[3 * end + 2] - sequence.borders[3 * start],
                                 [i - sequence.borders[3 * start] for i in sequence.borders[3 * start:3 * end + 3]],
                                 sequence.phonemes[start:end],
                                 sequence.offsets[start:end],
                                 sequence.repetititionSpacing[start:end],
                                 sequence.pitch[sequence.borders[3 * start]:sequence.borders[3 * end + 2]],
                                 sequence.steadiness[sequence.borders[3 * start]:sequence.borders[3 * end + 2]],
                                 sequence.breathiness[sequence.borders[3 * start]:sequence.borders[3 * end + 2]],
                                 sequence.aiBalance[sequence.borders[3 * start]:sequence.borders[3 * end + 2]],
                                 sequence.genderFactor[sequence.borders[3 * start]:sequence.borders[3 * end + 2]],
                                 sequence.vibratoSpeed[sequence.borders[3 * start]:sequence.borders[3 * end + 2]],
                                 sequence.vibratoStrength[sequence.borders[3 * start]:sequence.borders[3 * end + 2]],
                                 sequence.useBreathiness,
                                 sequence.useSteadiness,
                                 sequence.useAIBalance,
                                 sequence.useGenderFactor,
                                 sequence.useVibratoSpeed,
                                 sequence.useVibratoStrength,
                                 sequence.unvoicedShift,
                                 sequence.customCurves,
                                 sequence.nodeGraphFunction)
            idx = [random.randint(0, len(self.voicebank.phonemeDict[i]) - 1) for i in sequence.phonemes]
            embeddings = [self.voicebank.phonemeDict[phoneme][idx[i]].embedding for i, phoneme in enumerate(sequence.phonemes)]
            return sequence, embeddings
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
        borders = [0, 10]
        for i, phoneme in enumerate(phonemes):
            if phonemeSequence[i] in self.pool["short"]:
                length = phoneme.specharm.size()[0] * shortMultiplier
            else:
                length = phoneme.specharm.size()[0] * longMultiplier
            borders.append(borders[-1] + min(0.1 * length, 10))
            borders.append(borders[-2] + max(0.9 * length, length - 10))
            borders.append(borders[-3] + length)
        borders.append(borders[-1] + 25)
        borders = [i + random.normalvariate(0, noise[0] * 5) for i in borders]
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
                                 torch.zeros((borderLength,), device = torch.device("cpu")),#gender factor
                                 torch.zeros((borderLength,), device = torch.device("cpu")),#vibrato speed
                                 torch.full((borderLength,), -1, device = torch.device("cpu")),#vibrato strength
                                 True,
                                 True,
                                 False,
                                 True,
                                 True,
                                 True,
                                 0.,
                                 [],
                                 None
        )
        return sequence, embeddings

    def synthesize(self, noise:list, length:int, phonemeLength:int = None, expression:str = "") -> torch.Tensor:
        if self.mode == "dataset file":
            sample = next(iter(self.pool["dataset"]))
            sample = sample.specharm + torch.cat((sample.avgSpecharm[:global_consts.halfHarms], torch.zeros((global_consts.halfHarms,), device = sample.avgSpecharm.device), sample.avgSpecharm[global_consts.halfHarms:]), 0)
            return sample
        """noise mappings: [borders, offsets/spacing, steadiness, breathiness]"""
        sequence, embeddings = self.makeSequence(noise, length, phonemeLength, expression)
        output = torch.zeros([sequence.length, global_consts.halfTripleBatchSize + global_consts.nHarmonics + 3], device = torch.device("cpu"))
        output[sequence.borders[0]:sequence.borders[3]] = getSpecharm(VocalSegment(sequence, self.voicebank, 0, torch.device("cpu")), torch.device("cpu"))
        for i in range(1, sequence.phonemeLength - 1):
            output[sequence.borders[3*i+2]:sequence.borders[3*i+3]] = getSpecharm(VocalSegment(sequence, self.voicebank, i, torch.device("cpu")), torch.device("cpu"))
        output[sequence.borders[-4]:sequence.borders[-1]] = getSpecharm(VocalSegment(sequence, self.voicebank, sequence.phonemeLength - 1, torch.device("cpu")), torch.device("cpu"))
        for i in range(1, sequence.phonemeLength):
            output[sequence.borders[3*i]-1:sequence.borders[3*i+2]] = self.crfWrapper(output[sequence.borders[3*i] - 3],
                                                                               output[sequence.borders[3*i] - 2],
                                                                               output[sequence.borders[3*i+2]],
                                                                               output[sequence.borders[3*i+2] + 1],
                                                                               embeddings[i - 1],
                                                                               embeddings[i],
                                                                               sequence.borders[3*i+2] - sequence.borders[3*i] + 1,
                                                                               sequence.pitch[sequence.borders[3*i]-1:sequence.borders[3*i+2]],
                                                                               (sequence.borders[3*i+1] - sequence.borders[3*i]))
        return self.augment(output)
    
    def crfWrapper(self, specharm1:torch.Tensor, specharm2:torch.Tensor, specharm3:torch.Tensor, specharm4:torch.Tensor, embedding1:torch.Tensor, embedding2:torch.Tensor, outputSize:int, pitchCurve:torch.Tensor, slopeFactor:int):
        factor = log(0.5, slopeFactor / outputSize)
        factor = torch.pow(torch.linspace(0, 1, outputSize), factor)
        embedding1 = dec2bin(torch.tensor(embedding1), 32)
        embedding2 = dec2bin(torch.tensor(embedding2), 32)
        specharm = torch.squeeze(self.crfAi(specharm1, specharm2, specharm3, specharm4, embedding1, embedding2, factor))
        return specharm
    
    def augment(self, input:torch.Tensor) -> torch.Tensor:
        device = input.device
        input = input.to(torch.device("cpu"))
        input[:, :global_consts.halfHarms] = torch.clamp(input[:, :global_consts.halfHarms], 0.001, 1)
        input[:, 2*global_consts.halfHarms:] = torch.clamp(input[:, 2*global_consts.halfHarms:], 0.001, 1)
        input = input.contiguous()
        input_ptr = ctypes.cast(input.data_ptr(), ctypes.POINTER(ctypes.c_float))
        length = input.size()[0]
        c_length = ctypes.c_int(length)
        pitch = torch.full((length,), 300.5, dtype = torch.float32).contiguous()
        pitch_ptr = ctypes.cast(pitch.data_ptr(), ctypes.POINTER(ctypes.c_float))
        
        dynamics = torch.full((length,), random.uniform(-0.8, 0.8), dtype = torch.float32).contiguous()
        dynamics_ptr = ctypes.cast(dynamics.data_ptr(), ctypes.POINTER(ctypes.c_float))
        esper.applyDynamics(input_ptr, dynamics_ptr, pitch_ptr, c_length, global_consts.config)
        breathiness = torch.full((length,), random.uniform(-0.8, 0.8), dtype = torch.float32).contiguous()
        breathiness_ptr = ctypes.cast(breathiness.data_ptr(), ctypes.POINTER(ctypes.c_float))
        esper.applyBreathiness(input_ptr, breathiness_ptr, c_length, global_consts.config)
        
        pitchTgt = torch.full((length,), 300.5 + random.uniform(-80., 80.), dtype = torch.float32).contiguous()
        pitchTgt_ptr = ctypes.cast(pitchTgt.data_ptr(), ctypes.POINTER(ctypes.c_float))
        formantShift = torch.full((length,), random.uniform(0., 0.5), dtype = torch.float32).contiguous()
        formantShift_ptr = ctypes.cast(formantShift.data_ptr(), ctypes.POINTER(ctypes.c_float))
        esper.pitchShift(input_ptr, pitch_ptr, pitchTgt_ptr, formantShift_ptr, breathiness_ptr, c_length, global_consts.config)
        
        input = input.to(device)
        return input


class InceptionModule1d(nn.Module):
    
    def __init__(self, inDim:int, outDim:int, kernelSizes:tuple, dilations:tuple, poolSize:int, device:torch.device = None) -> None:
        super().__init__()
        effectiveOutDim = outDim // (len(kernelSizes) + 1)
        self.device = device
        self.paths = nn.ModuleList([nn.Sequential(
            nn.Conv1d(inDim, effectiveOutDim, 1, device = device),
            nn.LeakyReLU(0.1),
            nn.Conv1d(effectiveOutDim, effectiveOutDim, kernelSize, padding = dilation * (kernelSize - 1) // 2, dilation = dilation, device = device),
        ) for kernelSize, dilation in zip(kernelSizes, dilations)])
        self.poolPath = nn.Sequential(
            nn.MaxPool1d(poolSize, 1, poolSize // 2),
            nn.Conv1d(inDim, effectiveOutDim, 1, device = device),
        )
    
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        x = torch.cat([path(input) for path in self.paths], 0)
        pooled = self.poolPath(input)
        x = torch.cat([x, pooled], 0)
        return x

class InceptionBlock(nn.Module):
    
    def __init__(self, dim:int, kernelSizes:tuple, dilations:tuple, poolSize:int, device:torch.device = None) -> None:
        super().__init__()
        self.inception = InceptionModule1d(dim, dim, kernelSizes, dilations, poolSize, device = device)
        self.nla = nn.LeakyReLU(0.1)
        self.norm = nn.LayerNorm(dim, bias = False, device = device)
    
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        x = torch.adjoint(input)
        x = self.inception(x)
        x = torch.adjoint(x)
        x = self.nla(x)
        x = self.norm(x)
        x = x + input
        return x
    

class MainAi(nn.Module):
    
    def __init__(self, dim:int, embedDim:int, blockA:list, blockB:list, blockC:list, device:torch.device = None, learningRate:float=5e-5, regularization:float=1e-5, dropout:float=0.05, compile:bool = True) -> None:
        super().__init__()
        self.device = device
        self.learningRate = learningRate
        self.dim = dim
        self.blockAHParams = blockA
        self.blockBHParams = blockB
        self.blockCHParams = blockC
        self.regularization = regularization
        self.epoch = 0
        self.sampleCount = 0
        self.inputHead = nn.Sequential(
            nn.Linear(input_dim + embedDim, dim, device = device),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(dim, device = device),
        )
        self.blocks = nn.Sequential(*[InceptionBlock(dim, (3, 5, 7), (1, 1, 1), 3, device = device) for _ in range(24)])
        self.outputHead = nn.Sequential(
            nn.Unflatten(1, (dim, 1)),
            nn.Dropout1d(dropout),
            nn.Flatten(1, 2),
            nn.Linear(dim, input_dim, device = device),
            nn.Tanh(),
        )
    
    def forward(self, input:torch.Tensor, level:int, embedding:torch.Tensor) -> torch.Tensor:
        x = torch.cat((input, embedding.unsqueeze(0).tile((input.size()[0], 1))), 1)
        x = self.inputHead(x)
        x = self.blocks(x)
        x = self.outputHead(x)
        if True:
            x = torch.cat((x[:, :, None], x.roll(1, 0)[:, :, None], x.roll(-1, 0)[:, :, None], x.roll(2, 0)[:, :, None], x.roll(-2, 0)[:, :, None]), dim = 2)
            x = torch.median(x, 2)[0]
            x[:, halfHarms:2 * halfHarms] *= 0.
        return 3 * x + input
    
    def resetState(self) -> None:
        pass

class MainCritic(nn.Module):
    
    def __init__(self, dim:int, embedDim:int, blockA:list, blockB:list, blockC:list, outputWeight:int = 0.9, device:torch.device = None, learningRate:float=5e-5, regularization:float=1e-5, dropout:float=0.05, compile:bool = True) -> None:
        super().__init__()
        self.device = device
        self.learningRate = learningRate
        self.dim = dim
        self.blockAHParams = blockA
        self.blockBHParams = blockB
        self.blockCHParams = blockC
        self.regularization = regularization
        self.epoch = 0
        self.sampleCount = 0
        self.inputHead = nn.Sequential(
            nn.Unflatten(1, (input_dim + embedDim, 1)),
            nn.Dropout1d(dropout),
            nn.Flatten(1, 2),
            nn.Linear(input_dim + embedDim, dim, device = device),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(dim, device = device),
        )
        self.blocks = nn.Sequential(*[InceptionBlock(dim, (3, 5, 7), (1, 1, 1), 3, device = device) for _ in range(24)])
        self.outputHead = nn.Sequential(
            nn.Linear(dim, 1, device = device),
        )
    
    def forward(self, input:torch.Tensor, level:int, embedding:torch.Tensor) -> torch.Tensor:
        """perturbation = torch.normal(0, 0.2, input.size(), device = self.device)
        crosstalk = (
            torch.normal(0, 0.1, input.size(), device = self.device) * torch.roll(input, (0, 1), (0, 1)) +
            torch.normal(0, 0.1, input.size(), device = self.device) * torch.roll(input, (0, -1), (0, 1)) +
            torch.normal(0, 0.1, input.size(), device = self.device) * torch.roll(input, (1, 0), (1, 0)) +
            torch.normal(0, 0.1, input.size(), device = self.device) * torch.roll(input, (-1, 0), (1, 0))
        )
        input = input + perturbation + crosstalk"""
        x = torch.cat((input, embedding.unsqueeze(0).tile((input.size()[0], 1))), 1)
        x = self.inputHead(x)
        x = self.blocks(x)
        x = self.outputHead(x)
        return torch.mean(x)
    
    def resetState(self) -> None:
        pass
