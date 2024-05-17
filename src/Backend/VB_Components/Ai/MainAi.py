#Copyright 2022 - 2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import random
from math import floor, ceil, log
from copy import copy
from tkinter import Tk, filedialog
from tqdm.auto import tqdm
import h5py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import global_consts
from Backend.VB_Components.Ai.TrAi import TrAi
from Backend.VB_Components.Ai.Util import SReLU
from Backend.DataHandler.VocalSequence import VocalSequence
from Backend.DataHandler.VocalSegment import VocalSegment
from Backend.DataHandler.AudioSample import LiteSampleCollection
from Backend.DataHandler.HDF5 import SampleStorage
from Backend.Resampler.Resamplers import getSpecharm
from Backend.Resampler.CubicSplineInter import interp
from Backend.Resampler.PhaseShift import phaseInterp
from Backend.ESPER.SpectralCalculator import asyncProcess
from Util import dec2bin

halfHarms = int(global_consts.nHarmonics / 2) + 1
input_dim = global_consts.halfTripleBatchSize + halfHarms + 1

def dataLoader_collate(data):
    return data[0]
    """This is exactly as dumb as it looks. This function is only here to work around DataLoader default collation being active even when there is nothing to collate, with no way to turn it off.
    Also, it needs to be defined here because the Pickle pipes used by DataLoader can't send it to a worker process when it is defined in the scope it would normally belong in."""

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
            inputQueue, outputQueue, processes = asyncProcess(False, False, True)
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
                                                            torch.tensor(group["vibratoSpeed"]),
                                                            torch.tensor(group["vibratoStrength"]),
                                                            bool(group.attrs["useBreathiness"]),
                                                            bool(group.attrs["useSteadiness"]),
                                                            bool(group.attrs["useAIBalance"]),
                                                            bool(group.attrs["useVibratoSpeed"]),
                                                            bool(group.attrs["useVibratoStrength"]),
                                                            [],
                                                            None))   
        elif self.mode == "reclist (strict vowels)":
            self.pool["long"] = [key for key, value in self.voicebank.phonemeDict.items() if value[0].isPlosive and value[0].isVoiced]
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
                                 sequence.vibratoSpeed[sequence.borders[3 * start]:sequence.borders[3 * end + 2]],
                                 sequence.vibratoStrength[sequence.borders[3 * start]:sequence.borders[3 * end + 2]],
                                 sequence.useBreathiness,
                                 sequence.useSteadiness,
                                 sequence.useAIBalance,
                                 sequence.useVibratoSpeed,
                                 sequence.useVibratoStrength,
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
            sample = next(iter(self.pool["dataset"]))
            sample = sample.specharm + torch.cat((sample.avgSpecharm[:global_consts.halfHarms], torch.zeros((global_consts.halfHarms,), device = sample.avgSpecharm.device), sample.avgSpecharm[global_consts.halfHarms:]), 0)
            return sample.to(torch.device(self.crfAi.device))
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
    
    
    
    
    
    
    
    
    
    
class PseudoSSM(nn.Module):
    def __init__(self, srcDim:int, tgtDim, stateDim:int, device:torch.device = None, specnorm:bool = False) -> None:
        super().__init__()
        self.srcDim = srcDim
        self.tgtDim = tgtDim
        self.stateDim = stateDim
        self.device = device
        self.recurrent = nn.RNN(srcDim, stateDim, batch_first = True, bias = False, device = device)
        self.howb = nn.Linear(stateDim, tgtDim, device = device)
        self.iowb = nn.Linear(srcDim, tgtDim, device = device)
        self.ib = nn.Parameter(torch.zeros((srcDim,), device = device))
        self.initialState = nn.Parameter(torch.zeros((stateDim,), device = device))
        nn.init.orthogonal_(self.recurrent.weight_ih_l0)
        nn.init.orthogonal_(self.recurrent.weight_hh_l0)
        nn.init.orthogonal_(self.howb.weight)
        nn.init.orthogonal_(self.iowb.weight)
        nn.init.zeros_(self.howb.bias)
        nn.init.zeros_(self.iowb.bias)
        if specnorm:
            self.recurrent = nn.utils.parametrizations.spectral_norm(self.recurrent, name = "weight_hh_l0")
            self.recurrent = nn.utils.parametrizations.spectral_norm(self.recurrent, name = "weight_ih_l0")
            self.howb = nn.utils.parametrizations.spectral_norm(self.howb)
            self.iowb = nn.utils.parametrizations.spectral_norm(self.iowb)
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        skip = self.iowb(input)
        x = input + self.ib
        x = self.recurrent(x, self.initialState.unsqueeze(0))[0]
        x = self.howb(x)
        return x + skip
    
class PseudoSSMChain(nn.Module):
    """Chain of two PseudoSSMs, for use in an Inception-style block."""
    
    def __init__(self, srcDim:int, tgtDim:int, stateDim:int, timeStep:int, specnorm:bool = False, device:torch.device = torch.device("cpu")) -> None:
        super().__init__()
        self.timeStep = timeStep
        intermediateDim = ceil((srcDim + tgtDim) * 1 / 3)
        self.pool = nn.MaxPool1d(timeStep, timeStep, ceil_mode = True)
        self.ssm1 = PseudoSSM(srcDim, 2 * intermediateDim, stateDim, device = device, specnorm = specnorm)
        self.nla = nn.GLU()
        self.nla2 = nn.Tanh()
        self.ssm2 = PseudoSSM(intermediateDim, tgtDim, stateDim, device = device, specnorm = specnorm)
        
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        x = self.pool(input.transpose(-1, -2)).transpose(-1, -2)
        x = self.ssm1(x)
        x = self.nla(x)
        x = self.ssm2(x)
        x = self.nla2(x)
        x = torch.tile(x, (self.timeStep, 1))[:input.size()[-2], :]
        return x

class PseudoSSMInception(nn.Module):
    """Inception-style block consisting of multiple PseudoSSMs with different time steps, akin to the different convolutional kernel sizes in a traditional Inception block."""
    
    def __init__(self, dim:int, tgtDims:int, stateDims:tuple, timeSteps:tuple, specnorm:bool = False, device:torch.device = torch.device("cpu")) -> None:
        super().__init__()
        self.device = device
        self.specnorm = specnorm
        if len(tgtDims) != len(timeSteps) or len(stateDims) != len(timeSteps):
            raise ValueError("Number of target dimensions, state dimensions and time steps must match")
        if sum(tgtDims) != dim:
            raise ValueError("Sum of target dimensions must equal input dimension")
        self.ssms = nn.ModuleList([PseudoSSMChain(dim, tgtDim, stateDim, timeStep, specnorm = False, device = device) for timeStep, tgtDim, stateDim in zip(timeSteps, tgtDims, stateDims)])
        self.nla = SReLU()
    
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        output = torch.cat([ssm(input) for ssm in self.ssms], -1)
        output += input
        return output

class EncoderBlock(nn.Module):
    def __init__(self, srcDim:int, tgtDim, stateDim:int, device:torch.device = None, specnorm:bool = False) -> None:
        super().__init__()
        self.srcDim = srcDim
        self.tgtDim = tgtDim
        self.stateDim = stateDim
        self.device = device
        self.specnorm = specnorm
        self.ssm = PseudoSSM(tgtDim, 2 * tgtDim, stateDim, device = device, specnorm = specnorm)
        self.projection = nn.Conv1d(srcDim, 2 * tgtDim, 4, 2, 3, device = device)
        self.norm = nn.LayerNorm(tgtDim, device = device)
        self.nl1 = nn.GLU()
        self.nl2 = nn.GLU()
        nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        if specnorm:
            self.projection = nn.utils.parametrizations.spectral_norm(self.projection)
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        x = self.projection(input.transpose(-1, -2)).transpose(-1, -2)
        x = self.nl1(x)
        x = self.ssm(x)
        x = self.nl2(x)
        x = self.norm(x)
        return x, input.size()[-2]

class DecoderBlock(nn.Module):
    def __init__(self, srcDim:int, tgtDim, stateDim:int, device:torch.device = None, specnorm:bool = False) -> None:
        super().__init__()
        self.srcDim = srcDim
        self.tgtDim = tgtDim
        self.stateDim = stateDim
        self.device = device
        self.specnorm = specnorm
        self.ssm = PseudoSSM(tgtDim, 2 * tgtDim, stateDim, device = device, specnorm = specnorm)
        self.projection = nn.ConvTranspose1d(srcDim, 2 * tgtDim, 4, 2, device = device)
        self.norm = nn.LayerNorm(tgtDim, device = device)
        self.nl1 = nn.GLU()
        self.nl2 = nn.GLU()
        nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        if specnorm:
            self.projection = nn.utils.parametrizations.spectral_norm(self.projection)
    def forward(self, input:torch.Tensor, targetLength:int) -> torch.Tensor:
        x = self.projection(input.transpose(-1, -2)).transpose(-1, -2)
        x = self.nl1(x)
        x = self.ssm(x)
        x = self.nl2(x)
        x = self.norm(x)
        return x[3:targetLength + 3]

class MainAi(nn.Module):
    def __init__(self, dim:int, embedDim:int, blockA:list, blockB:list, blockC:list, device:torch.device = None, learningRate:float=5e-5, regularization:float=1e-5, dropout:float=0.05, compile:bool = True) -> None:
        super().__init__()
        self.preNet = nn.Sequential(
            nn.Linear(input_dim + embedDim, dim, device = device),
            nn.Tanh(),
        )
        self.encoderA = EncoderBlock(dim, blockA[1], blockA[0], device = device)
        self.encoderB = EncoderBlock(blockA[1], blockB[1], blockB[0], device = device)
        self.encoderC = EncoderBlock(blockB[1], blockC[1], blockC[0], device = device)
        self.decoderC = DecoderBlock(blockC[1], blockB[1], blockC[0], device = device)
        self.decoderB = DecoderBlock(blockB[1], blockA[1], blockB[0], device = device)
        self.decoderA = DecoderBlock(blockA[1], dim, blockA[0], device = device)
        self.postNet = nn.Sequential(
            nn.Linear(dim, input_dim, device = device),
            nn.Softplus(),
        )
        self.device = device
        self.learningRate = learningRate
        self.dim = dim
        self.blockAHParams = blockA
        self.blockBHParams = blockB
        self.blockCHParams = blockC
        self.regularization = regularization
        self.epoch = 0
        self.sampleCount = 0
    def forward(self, input:torch.Tensor, level:int, embedding:torch.Tensor) -> torch.Tensor:
        if level == 0:
            return input
        latent = torch.cat((input, embedding.unsqueeze(0).tile((input.size()[0], 1))), 1)
        x = self.preNet(latent)
        a, lengthA = self.encoderA(x)
        b, lengthB = self.encoderB(a)
        c, lengthC = self.encoderC(b)
        y = self.decoderC(c, lengthC)
        y = self.decoderB(y + b, lengthB)
        y = self.decoderA(y + a, lengthA)
        x = self.postNet(x + y)
        return x
    def resetState(self) -> None:
        pass
    def __new__(cls, dim:int, embedDim:int, blockA:list, blockB:list, blockC:list, outputWeight:int = 0.9, device:torch.device = None, learningRate:float=5e-5, regularization:float=1e-5, dropout:float=0.05, compile:bool = False):
        instance = super().__new__(cls)
        if compile:
            instance = torch.compile(instance, dynamic = True, mode = "reduce-overhead")
        return instance

class MainCritic(nn.Module):
    
    def __init__(self, dim:int, embedDim:int, blockA:list, blockB:list, blockC:list, outputWeight:int = 0.9, device:torch.device = None, learningRate:float=5e-5, regularization:float=1e-5, dropout:float=0.05, compile:bool = True) -> None:
        super().__init__()
        self.preNet = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Linear(input_dim + embedDim, dim, device = device)),
            nn.Tanh(),
        )
        self.encoderA = EncoderBlock(dim, blockA[1], blockA[0], device = device, specnorm = True)
        self.encoderB = EncoderBlock(blockA[1], blockB[1], blockB[0], device = device, specnorm = True)
        self.encoderC = EncoderBlock(blockB[1], blockC[1], blockC[0], device = device, specnorm = True)
        self.decoderC = DecoderBlock(blockC[1], blockB[1], blockC[0], device = device, specnorm = True)
        self.decoderB = DecoderBlock(blockB[1], blockA[1], blockB[0], device = device, specnorm = True)
        self.decoderA = DecoderBlock(blockA[1], dim, blockA[0], device = device, specnorm = True)
        self.postNet = nn.utils.parametrizations.spectral_norm(nn.Linear(dim, 1, device = device))
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
    def forward(self, input:torch.Tensor, level:int, embedding:torch.Tensor) -> torch.Tensor:
        latent = torch.cat((input, embedding.unsqueeze(0).tile((input.size()[0], 1))), 1)
        x = self.preNet(latent)
        a, lengthA = self.encoderA(x)
        b, lengthB = self.encoderB(a)
        c, lengthC = self.encoderC(b)
        y = self.decoderC(c, lengthC)
        y = self.decoderB(y + b, lengthB)
        y = self.decoderA(y + a, lengthA)
        x = self.postNet(x + y)
        return self.outputWeight * x[-1] + (1 - self.outputWeight) * torch.mean(x)
    def resetState(self) -> None:
        pass
    def __new__(cls, dim:int, embedDim:int, blockA:list, blockB:list, blockC:list, outputWeight:int = 0.9, device:torch.device = None, learningRate:float=5e-5, regularization:float=1e-5, dropout:float=0.05, compile:bool = False):
        instance = super().__new__(cls)
        if compile:
            instance = torch.compile(instance, dynamic = True, mode = "reduce-overhead")
        return instance

class MainAi(nn.Module):
    def __init__(self, dim:int, embedDim:int, blockA:list, blockB:list, blockC:list, device:torch.device = None, learningRate:float=5e-5, regularization:float=1e-5, dropout:float=0.05, compile:bool = True) -> None:
        super().__init__()
        self.preNet = nn.Sequential(
            nn.Linear(input_dim + embedDim, dim, device = device),
            nn.Softplus(),
        )
        self.mainNet = nn.Sequential(*[PseudoSSMInception(dim, (dim//4, dim//4, dim//4, dim//4), (dim//2, dim//2, dim//2, dim//2), (1, 4, 16, 64), False, device) for _ in range(1)])
        self.postNet = nn.Sequential(
            nn.Linear(dim, input_dim, device = device),
            nn.Softplus(),
        )
        self.device = device
        self.learningRate = learningRate
        self.dim = dim
        self.blockAHParams = blockA
        self.blockBHParams = blockB
        self.blockCHParams = blockC
        self.regularization = regularization
        self.epoch = 0
        self.sampleCount = 0
    def forward(self, input:torch.Tensor, level:int, embedding:torch.Tensor) -> torch.Tensor:
        if level == 0:
            return input
        latent = torch.cat((input, embedding.unsqueeze(0).tile((input.size()[0], 1))), 1)
        x = self.preNet(latent)
        x = self.mainNet(x)
        x = self.postNet(x)
        return x * input
    def resetState(self) -> None:
        pass
    def __new__(cls, dim:int, embedDim:int, blockA:list, blockB:list, blockC:list, outputWeight:int = 0.9, device:torch.device = None, learningRate:float=5e-5, regularization:float=1e-5, dropout:float=0.05, compile:bool = False):
        instance = super().__new__(cls)
        if compile:
            instance = torch.compile(instance, dynamic = True, mode = "reduce-overhead")
        return instance

class MainCritic(nn.Module):
    
    def __init__(self, dim:int, embedDim:int, blockA:list, blockB:list, blockC:list, outputWeight:int = 0.9, device:torch.device = None, learningRate:float=5e-5, regularization:float=1e-5, dropout:float=0.05, compile:bool = True) -> None:
        super().__init__()
        self.preNet = nn.Sequential(
            nn.Linear(input_dim + embedDim, dim, device = device),
            nn.Softplus(),
        )
        self.mainNet = nn.Sequential(*[PseudoSSMInception(dim, (dim//4, dim//4, dim//4, dim//4), (dim//2, dim//2, dim//2, dim//2), (1, 4, 16, 64), True, device) for _ in range(1)])
        self.postNet = nn.Linear(dim, 1, device = device)
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
    def forward(self, input:torch.Tensor, level:int, embedding:torch.Tensor) -> torch.Tensor:
        latent = torch.cat((input, embedding.unsqueeze(0).tile((input.size()[0], 1))), 1)
        x = self.preNet(latent)
        x = self.mainNet(x)
        x = self.postNet(x)
        x = self.outputWeight * x[-1] + (1 - self.outputWeight) * torch.mean(x)
        return x
    def resetState(self) -> None:
        pass
    def __new__(cls, dim:int, embedDim:int, blockA:list, blockB:list, blockC:list, outputWeight:int = 0.9, device:torch.device = None, learningRate:float=5e-5, regularization:float=1e-5, dropout:float=0.05, compile:bool = False):
        instance = super().__new__(cls)
        if compile:
            instance = torch.compile(instance, dynamic = True, mode = "reduce-overhead")
        return instance
