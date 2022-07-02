from kivy.core.image import Image as CoreImage
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, OptionProperty

from io import BytesIO
from copy import copy, deepcopy

import torch
import math

import sounddevice

from Backend.Param_Components.AiParams import AiParamStack
import global_consts

from MiddleLayer.IniParser import readSettings
import MiddleLayer.DataHandlers as dh

from UI.code.editor.AdaptiveSpace import ParamCurve, TimingOptns, PitchOptns
from UI.code.editor.Headers import SingerPanel, ParamPanel

class MiddleLayer(Widget):
    def __init__(self, ids, **kwargs):
        super().__init__(**kwargs)
        self.sequenceList = []
        self.voicebankList = []
        self.aiParamStackList = []
        from Backend.NV_Multiprocessing.Manager import RenderManager
        self.manager = RenderManager(self.sequenceList, self.voicebankList, self.aiParamStackList)
        self.ids = ids
        self.trackList = []
        self.activeTrack = None
        self.activeParam = "steadiness"
        self.mode = OptionProperty("notes", options = ["notes", "timing", "pitch"])
        self.mode = "notes"
        self.tool = OptionProperty("draw", options = ["draw", "line", "arch", "reset"])
        self.tool = "draw"
        self.shift = BooleanProperty()
        self.shift = False
        self.scrollValue = 0.
        self.audioBuffer = []
        self.mainAudioBufferPos = 0
        self.deletions = []
        self.playing = False
        settings = readSettings()
        device = None
        devices = sounddevice.query_devices()
        for i in devices:
            if i["name"] == settings["audioDevice"]:
                device = i["name"] + ", " + settings["audioApi"]
        self.audioStream = sounddevice.OutputStream(global_consts.sampleRate, global_consts.audioBufferSize, device, callback = self.playCallback)
        self.scriptCache = ""
    def importVoicebank(self, path, name, inImage):
        track = dh.Track(path)
        self.trackList.append(track)
        canvas_img = inImage
        data = BytesIO()
        canvas_img.save(data, format='png')
        data.seek(0)
        im = CoreImage(BytesIO(data.read()), ext='png')
        image = im.texture
        self.ids["singerList"].add_widget(SingerPanel(name = name, image = image, index = len(self.trackList) - 1))
        self.audioBuffer.append(torch.zeros([5000 * global_consts.batchSize,]))
        self.aiParamStackList.append(AiParamStack([]))
        self.submitAddTrack(track)
    def importParam(self, path, name):
        self.trackList[self.activeTrack].paramStack.append(dh.Parameter(path))
        self.ids["paramList"].add_widget(ParamPanel(name = name, switchable = True, sortable = True, deletable = True, index = len(self.trackList[self.activeTrack].paramStack) - 1))
        self.submitAddParam(path)
    def changeTrack(self, index):
        self.activeTrack = index
        self.updateParamPanel()
        self.updatePianoRoll()
    def copyTrack(self, index, name, inImage):
        reference = self.trackList[index]
        self.trackList.append(dh.Track(reference.vbPath))
        self.trackList[-1].volume = copy(reference.volume)
        for i in reference.notes:
            self.trackList[-1].notes.append(dh.Note(i.xPos, i.yPos, i.phonemeStart, i.phonemeEnd))
            self.trackList[-1].notes[-1].length = copy(i.length)
            self.trackList[-1].notes[-1].phonemeMode = copy(i.phonemeMode)
            self.trackList[-1].notes[-1].content = copy(i.content)
        self.trackList[-1].phonemes = deepcopy(reference.phonemes)
        self.trackList[-1].pitch = reference.pitch.clone()
        self.trackList[-1].basePitch = reference.basePitch.clone()
        self.trackList[-1].breathiness = reference.breathiness.clone()
        self.trackList[-1].steadiness = reference.steadiness.clone()
        self.trackList[-1].loopOverlap = reference.loopOverlap.clone()
        self.trackList[-1].loopOffset = reference.loopOffset.clone()
        self.trackList[-1].vibratoSpeed = reference.vibratoSpeed.clone()
        self.trackList[-1].vibratoStrength = reference.vibratoStrength.clone()
        self.trackList[-1].usePitch = copy(reference.usePitch)
        self.trackList[-1].useBreathiness = copy(reference.useBreathiness)
        self.trackList[-1].useSteadiness = copy(reference.useSteadiness)
        self.trackList[-1].useVibratoSpeed = copy(reference.useVibratoSpeed)
        self.trackList[-1].useVibratoStrength = copy(reference.useVibratoStrength)
        self.trackList[-1].paramStack = []#replace once paramStack is fully implemented
        self.trackList[-1].borders = deepcopy(reference.borders)
        self.trackList[-1].length = copy(reference.length)
        self.trackList[-1].mixinVB = copy(reference.mixinVB)
        self.trackList[-1].pauseThreshold = copy(reference.pauseThreshold)
        image = inImage
        self.ids["singerList"].add_widget(SingerPanel(name = name, image = image, index = len(self.trackList) - 1))
        self.audioBuffer.append(deepcopy(self.audioBuffer[index]))
        self.aiParamStackList.append(AiParamStack([]))
        self.submitDuplicateTrack(index)
    def deleteTrack(self, index):
        self.trackList.pop(index)
        self.audioBuffer.pop(index)
        self.aiParamStackList.pop(index)
        if index <= self.activeTrack and index > 0:
            self.changeTrack(self.activeTrack - 1)
        for i in self.ids["singerList"].children:
            if i.index == index:
                i.parent.remove_widget(i)
            if i.index > index:
                i.index = i.index - 1
        self.deletions.append(index)
        self.submitRemoveTrack(index)
        if len(self.trackList) == 0:
            self.ids["paramList"].clear_widgets()
            self.ids["adaptiveSpace"].clear_widgets()
            self.activeTrack = None
            self.updatePianoRoll()
    def deleteParam(self, index):
        self.trackList[self.activeTrack].paramStack.pop(index)
        if index <= self.activeParam:
            self.changeParam(self.activeParam - 1)
        for i in self.ids["paramList"].children:
            if i.index == index:
                i.parent.remove_widget(i)
            if i.index > index:
                i.index = i.index - 1
        self.submitRemoveParam(index)
    def enableParam(self, index, name):
        if index == -1:
            if name == "steadiness":
                self.trackList[self.activeTrack].useSteadiness = True
            if name == "breathiness":
                self.trackList[self.activeTrack].useBreathiness = True
            if name == "vibrato speed":
                self.trackList[self.activeTrack].useVibratoSpeed = True
            if name == "vibrato strength":
                self.trackList[self.activeTrack].useVibratoStrength = True
        else:
            self.trackList[self.activeTrack].paramStack[index].enabled = True
        self.submitEnableParam(index, name)
    def disableParam(self, index, name):
        if index == -1:
            if name == "steadiness":
                self.trackList[self.activeTrack].useSteadiness = False
            if name == "breathiness":
                self.trackList[self.activeTrack].useBreathiness = False
            if name == "vibrato speed":
                self.trackList[self.activeTrack].useVibratoSpeed = False
            if name == "vibrato strength":
                self.trackList[self.activeTrack].useVibratoStrength = False
        else:
            self.trackList[self.activeTrack].paramStack[index].enabled = False
        self.submitDisableParam(index, name)
    def moveParam(self, name, switchable, sortable, deletable, index, delta, switchState = True):
        param = self.trackList[self.activeTrack].paramStack[index]
        if delta > 0:
            for i in range(delta):
                self.trackList[self.activeTrack].paramStack[index + i] = self.trackList[self.activeTrack].paramStack[index + i + 1]
            self.trackList[self.activeTrack].paramStack[index + delta] = param
        if delta < 0:
            for i in range(-delta):
                self.trackList[self.activeTrack].paramStack[index - i] = self.trackList[self.activeTrack].paramStack[index - i - 1]
            self.trackList[self.activeTrack].paramStack[index - delta] = param
        for i in self.ids["paramList"].children:
            if i.index == index:
                i.parent.remove_widget(i)
                break
        self.ids["paramList"].add_widget(ParamPanel(name = name, switchable = switchable, sortable = sortable, deletable = deletable, index = index), index = index + delta, switchState = switchState)
        self.changeParam(index + delta)
    def updateParamPanel(self):
        self.ids["paramList"].clear_widgets()
        self.ids["adaptiveSpace"].clear_widgets()
        if self.mode == "notes":
            self.ids["paramList"].add_widget(ParamPanel(name = "steadiness", switchable = True, sortable = False, deletable = False, index = -1, switchState = self.trackList[self.activeTrack].useSteadiness, state = "down"))
            self.ids["paramList"].add_widget(ParamPanel(name = "breathiness", switchable = True, sortable = False, deletable = False, index = -1, switchState = self.trackList[self.activeTrack].useBreathiness))
            self.ids["adaptiveSpace"].add_widget(ParamCurve())
            counter = 0
            for i in self.trackList[self.activeTrack].paramStack:
                self.ids["paramList"].add_widget(ParamPanel(name = i.name, index = counter))
                counter += 1
            self.changeParam(-1, "steadiness")
        if self.mode == "timing":
            self.ids["paramList"].add_widget(ParamPanel(name = "loop overlap", switchable = False, sortable = False, deletable = False, index = -1, state = "down"))
            self.ids["paramList"].add_widget(ParamPanel(name = "loop offset", switchable = False, sortable = False, deletable = False, index = -1))
            self.ids["adaptiveSpace"].add_widget(TimingOptns())
            self.changeParam(-1, "loop overlap")
        if self.mode == "pitch":
            self.ids["paramList"].add_widget(ParamPanel(name = "vibrato speed", switchable = True, sortable = False, deletable = False, index = -1, switchState = self.trackList[self.activeTrack].useVibratoSpeed, state = "down"))
            self.ids["paramList"].add_widget(ParamPanel(name = "vibrato strength", switchable = True, sortable = False, deletable = False, index = -1, switchState = self.trackList[self.activeTrack].useVibratoStrength))
            self.ids["adaptiveSpace"].add_widget(PitchOptns())
            self.changeParam(-1, "vibrato speed")
    def updatePianoRoll(self):
        self.ids["pianoRoll"].updateTrack()
    def changeParam(self, index, name):
        if index == -1:
            if name == "steadiness":
                self.activeParam = "steadiness"
            if name == "breathiness":
                self.activeParam = "breathiness"
            if name == "loop overlap" or name == "loop offset":
                self.activeParam = "loop"
            if name == "vibrato speed" or name == "vibrato strength":
                self.activeParam = "vibrato"
        else:
            self.activeParam = index
        self.ids["adaptiveSpace"].children[0].redraw()
    def applyParamChanges(self, data, start, section = False):
        if self.activeParam == "steadiness":
            self.trackList[self.activeTrack].steadiness[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
            self.submitNamedParamChange(True, "steadiness", start, torch.tensor(data, dtype = torch.half))
        elif self.activeParam == "breathiness":
            self.trackList[self.activeTrack].breathiness[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
            self.submitNamedParamChange(True, "breathiness", start, torch.tensor(data, dtype = torch.half))
        elif self.activeParam == "loop":
            if section:
                self.trackList[self.activeTrack].loopOverlap[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
                self.submitNamedPhonParamChange(True, "repetititionSpacing", start, torch.tensor(data, dtype = torch.half))
            else:
                self.trackList[self.activeTrack].loopOffset[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
                self.submitNamedPhonParamChange(True, "offsets", start, torch.tensor(data, dtype = torch.half))
        elif self.activeParam == "vibrato":
            if section:
                self.trackList[self.activeTrack].vibratoSpeed[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
                self.submitNamedParamChange(True, "vibratoSpeed", start, torch.tensor(data, dtype = torch.half))
            else:
                self.trackList[self.activeTrack].vibratoStrength[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
                self.submitNamedParamChange(True, "vibratoStrength", start, torch.tensor(data, dtype = torch.half))
        else:
            self.trackList[self.activeTrack].paramStack[self.activeParam].curve[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
            self.submitParamChange(True, self.activeParam, start, torch.tensor(data, dtype = torch.half))
    def applyPitchChanges(self, data, start):
        self.trackList[self.activeTrack].pitch[start:start + len(data)] = torch.tensor(data, dtype = torch.float32)
        data = self.noteToPitch(torch.tensor(data, dtype = torch.float32))
        self.submitNamedPhonParamChange(True, "pitch", start, torch.tensor(data, dtype = torch.half))
    def changePianoRollMode(self):
        self.ids["pianoRoll"].changeMode()
    def applyScroll(self):
        self.ids["pianoRoll"].applyScroll(self.scrollValue)
        self.ids["adaptiveSpace"].applyScroll(self.scrollValue)
    def offsetPhonemes(self, index, offset, pause = False, futurePhonemes = None):
        phonIndex = self.trackList[self.activeTrack].notes[index].phonemeStart
        if offset > 0:
            if len(self.trackList[self.activeTrack].phonemes) > phonIndex and self.trackList[self.activeTrack].notes[index].phonemeEnd > phonIndex:
                if self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart] == "_autopause":
                    phonIndex += 1
            for i in range(offset):
                self.trackList[self.activeTrack].phonemes.insert(phonIndex + i, "_X")
                self.trackList[self.activeTrack].loopOverlap = torch.cat([self.trackList[self.activeTrack].loopOverlap[0:phonIndex], torch.tensor([0.5], dtype = torch.half), self.trackList[self.activeTrack].loopOverlap[phonIndex:]], dim = 0)
                self.trackList[self.activeTrack].loopOffset = torch.cat([self.trackList[self.activeTrack].loopOffset[0:phonIndex], torch.tensor([0.5], dtype = torch.half), self.trackList[self.activeTrack].loopOffset[phonIndex:]], dim = 0)
                for j in range(3):
                    self.trackList[self.activeTrack].borders.insert(phonIndex * 3, 0)
        elif offset < 0:
            for i in range(-offset):
                self.trackList[self.activeTrack].phonemes.pop(phonIndex)
                self.trackList[self.activeTrack].loopOverlap = torch.cat([self.trackList[self.activeTrack].loopOverlap[0:phonIndex], self.trackList[self.activeTrack].loopOverlap[phonIndex + 1:]], dim = 0)
                self.trackList[self.activeTrack].loopOffset = torch.cat([self.trackList[self.activeTrack].loopOffset[0:phonIndex], self.trackList[self.activeTrack].loopOffset[phonIndex + 1:]], dim = 0)
                for j in range(3):
                    self.trackList[self.activeTrack].borders.pop(phonIndex * 3)
        for i in self.trackList[self.activeTrack].notes[index + 1:]:
            i.phonemeStart += offset
            i.phonemeEnd += offset
        self.trackList[self.activeTrack].notes[index].phonemeEnd += offset
        self.submitOffset(False, phonIndex, offset)
        autopause = False
        if len(self.trackList[self.activeTrack].phonemes) == 0:
            return
        if self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart] == "_autopause":
            autopause = True
        start = self.trackList[self.activeTrack].notes[index].xPos
        end = self.trackList[self.activeTrack].notes[index].xPos + self.trackList[self.activeTrack].notes[index].length
        if index < len(self.trackList[self.activeTrack].notes) - 1:
            end = min(end, self.trackList[self.activeTrack].notes[index + 1].xPos)
        divisor = (self.trackList[self.activeTrack].notes[index].phonemeEnd - self.trackList[self.activeTrack].notes[index].phonemeStart + 1) * 3
        iterationEnd = self.trackList[self.activeTrack].notes[index].phonemeEnd
        iterationEndBorder = iterationEnd
        if index + 1 == len(self.trackList[self.activeTrack].notes):
            iterationEndBorder += 1
        iterationStart = self.trackList[self.activeTrack].notes[index].phonemeStart
        iterationOffset = 0
        if (pause == False) and (autopause == False):
            pass
        elif (pause == False) and (autopause == True):
            iterationStart += 1
            iterationOffset -= 1
        elif (pause == True) and (autopause == False):
            if offset < 0:
                pass
            elif offset >= 0:
                iterationStart += 1
                iterationOffset -= 1
                phonemeStart = self.trackList[self.activeTrack].notes[index].phonemeStart
                preStart = self.trackList[self.activeTrack].notes[index - 1].xPos + self.trackList[self.activeTrack].notes[index - 1].length
                self.trackList[self.activeTrack].borders[3 * phonemeStart] = preStart
                self.trackList[self.activeTrack].borders[3 * phonemeStart + 1] = preStart + self.trackList[self.activeTrack].pauseThreshold * 0.1
                self.trackList[self.activeTrack].borders[3 * phonemeStart + 2] = preStart + self.trackList[self.activeTrack].pauseThreshold * 0.2
        elif (pause == True) and (autopause == True):
            if offset < 0:
                pass
            elif offset >= 0:
                iterationStart += 1
                iterationOffset -= 1
        divisor += 3 * iterationOffset
        lengthDeltas = []
        if futurePhonemes != None:
            for i in range(iterationStart, iterationEnd):
                self.trackList[self.activeTrack].phonemes[i] = futurePhonemes[i - iterationStart]
        for i in range(3 * iterationStart, 3 * iterationEndBorder - 1):
            lengthDeltas.append(None)
        preuttrActive = True
        preuttr = 0.
        for i in range(iterationEnd - iterationStart):
            if self.trackList[self.activeTrack].phonemes[i + iterationStart] == "_autopause":
                continue
            if self.trackList[self.activeTrack].phonemeLengths[self.trackList[self.activeTrack].phonemes[i + iterationStart]] != None:
                lengthDeltas[3 * i + 2] = self.trackList[self.activeTrack].phonemeLengths[self.trackList[self.activeTrack].phonemes[i + iterationStart]] / 5
                lengthDeltas[3 * i + 3] = self.trackList[self.activeTrack].phonemeLengths[self.trackList[self.activeTrack].phonemes[i + iterationStart]] / 5
                lengthDeltas[3 * i + 4] = self.trackList[self.activeTrack].phonemeLengths[self.trackList[self.activeTrack].phonemes[i + iterationStart]] / 5
                if preuttrActive:
                    preuttr -= self.trackList[self.activeTrack].phonemeLengths[self.trackList[self.activeTrack].phonemes[i + iterationStart]] * 3 / 5
                if lengthDeltas[3 * i] == None:
                    lengthDeltas[3 * i] = self.trackList[self.activeTrack].phonemeLengths[self.trackList[self.activeTrack].phonemes[i + iterationStart]] / 5
                    lengthDeltas[3 * i + 1] = self.trackList[self.activeTrack].phonemeLengths[self.trackList[self.activeTrack].phonemes[i + iterationStart]] / 5
                else:
                    lengthDeltas[3 * i] = min(lengthDeltas[3 * i], self.trackList[self.activeTrack].phonemeLengths[self.trackList[self.activeTrack].phonemes[i + iterationStart]] / 5)
                    lengthDeltas[3 * i + 1] = min(lengthDeltas[3 * i + 1], self.trackList[self.activeTrack].phonemeLengths[self.trackList[self.activeTrack].phonemes[i + iterationStart]] / 5)
            else:
                preuttrActive = False
        if preuttrActive:
            preuttr = 0.
        if index > 0:
            preuttrLim = 0.
            preuttrLim = self.trackList[self.activeTrack].notes[index - 1].phonemeEnd - self.trackList[self.activeTrack].notes[index - 1].phonemeStart
            preuttrLim = 2 * preuttrLim + 6
            if self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index - 1].phonemeStart] == "_autopause":
                preuttrLim -= 6
            preuttrLim += self.trackList[self.activeTrack].notes[index - 1].xPos - self.trackList[self.activeTrack].notes[index].xPos
            preuttr = max(preuttr, preuttrLim)
        preuttrNext = 0.
        if index + 1 < len(self.trackList[self.activeTrack].notes):
            iterationStartNext = self.trackList[self.activeTrack].notes[index + 1].phonemeStart
            iterationEndNext = self.trackList[self.activeTrack].notes[index + 1].phonemeEnd
            for i in range(iterationStartNext, iterationEndNext):
                if self.trackList[self.activeTrack].phonemes[i] == "_autopause":
                    continue
                elif self.trackList[self.activeTrack].phonemeLengths[self.trackList[self.activeTrack].phonemes[i]] != None:
                    preuttrNext -= self.trackList[self.activeTrack].phonemeLengths[self.trackList[self.activeTrack].phonemes[i]] * 3 / 5
                else:
                    break

        start += preuttr
        end += preuttrNext
        availableLength = end - start
        for i in lengthDeltas:
            if i != None:
                divisor -= 1
                availableLength -= i
        if divisor == 1:
            divisor += 1
        availableLength = int(availableLength / (divisor - 1))
        for i in range(len(lengthDeltas)):
            if lengthDeltas[i] == None:
                lengthDeltas[i] = availableLength
        if sum(lengthDeltas) <= end - start:
            counter = 0
            for i in range(3 * iterationStart, 3 * iterationEndBorder):
                self.trackList[self.activeTrack].borders[i] = start + counter
                if i + 1 < 3 * iterationEndBorder:
                    counter += lengthDeltas[i - 3 * iterationStart]
        else:
            for i in range(iterationStart, iterationEndBorder):
                j = i - self.trackList[self.activeTrack].notes[index].phonemeStart + iterationOffset
                self.trackList[self.activeTrack].borders[3 * i] = start + int((end - start) * (3 * j) / divisor)
                self.trackList[self.activeTrack].borders[3 * i + 1] = start + int((end - start) * (3 * j + 1) / divisor)
                self.trackList[self.activeTrack].borders[3 * i + 2] = start + int((end - start) * (3 * j + 2) / divisor)

        for i in range(3 * (iterationStart + iterationOffset), 3 * iterationEndBorder):
            self.repairBorders(i)
        self.submitNamedPhonParamChange(False, "borders", 3 * (iterationStart + iterationOffset), self.trackList[self.activeTrack].borders[3 * (iterationStart + iterationOffset):3 * iterationEndBorder])

    def makeAutoPauses(self, index):
        if index > 0:
            offset = 0
            if self.trackList[self.activeTrack].notes[index].phonemeStart < self.trackList[self.activeTrack].notes[index].phonemeEnd:
                if self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart] == "_autopause":
                    offset -= 1
            if self.trackList[self.activeTrack].notes[index].xPos - self.trackList[self.activeTrack].notes[index - 1].xPos - self.trackList[self.activeTrack].notes[index - 1].length > self.trackList[self.activeTrack].pauseThreshold:
                offset += 1
            if offset != 0:
                self.offsetPhonemes(index, offset, True)
            if offset == 1:
                self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart] = "_autopause"
                self.submitNamedPhonParamChange(False, "phonemes", self.trackList[self.activeTrack].notes[index].phonemeStart, self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart:self.trackList[self.activeTrack].notes[index].phonemeEnd])
        if index < len(self.trackList[self.activeTrack].notes) - 1 and self.trackList[self.activeTrack].notes[index].phonemeEnd < len(self.trackList[self.activeTrack].phonemes):
            offset = 0
            if self.trackList[self.activeTrack].notes[index + 1].phonemeStart < self.trackList[self.activeTrack].notes[index + 1].phonemeEnd:
                if self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index + 1].phonemeStart] == "_autopause":
                    offset -= 1
            if self.trackList[self.activeTrack].notes[index + 1].xPos - self.trackList[self.activeTrack].notes[index].xPos - self.trackList[self.activeTrack].notes[index].length > self.trackList[self.activeTrack].pauseThreshold:
                offset += 1
            if offset != 0:
                self.offsetPhonemes(index + 1, offset, True)
            if offset == 1:
                self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index + 1].phonemeStart] = "_autopause"
                self.submitNamedPhonParamChange(False, "phonemes", self.trackList[self.activeTrack].notes[index + 1].phonemeStart, self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index + 1].phonemeStart:self.trackList[self.activeTrack].notes[index + 1].phonemeEnd])

    def recalculatePauses(self, index):
        for i in range(len(self.trackList[index].notes)):
            self.makeAutoPauses(i)
    
    def switchNote(self, index):
        note = self.trackList[self.activeTrack].notes.pop(index + 1)
        seq1 = self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart:self.trackList[self.activeTrack].notes[index].phonemeEnd]
        seq2 = self.trackList[self.activeTrack].phonemes[note.phonemeStart:note.phonemeEnd]
        brd1 = self.trackList[self.activeTrack].borders[3 * self.trackList[self.activeTrack].notes[index].phonemeStart:3 * self.trackList[self.activeTrack].notes[index].phonemeEnd]
        brd2 = self.trackList[self.activeTrack].borders[3 * note.phonemeStart:3 * note.phonemeEnd]
        if index == len(self.trackList[self.activeTrack].notes) - 1:#notes are 1 element shorter because of pop
            scalingFactor = (self.trackList[self.activeTrack].notes[index].phonemeEnd - self.trackList[self.activeTrack].notes[index].phonemeStart + 1) / (note.phonemeEnd - note.phonemeStart + 1)
            for i in range(len(self.trackList[self.activeTrack].borders) - 3, len(self.trackList[self.activeTrack].borders)):
                self.trackList[self.activeTrack].borders[i] = (self.trackList[self.activeTrack].borders[i] - note.xPos) * scalingFactor / note.length * (self.trackList[self.activeTrack].notes[index].xPos - note.xPos) + self.trackList[self.activeTrack].notes[index].xPos
            for i in range(3 * note.phonemeEnd - 3 * note.phonemeStart):
                brd2[i] = (brd2[i] - note.xPos) * scalingFactor + note.xPos
        self.trackList[self.activeTrack].notes.insert(index, note)
        phonemeMid = note.phonemeEnd - note.phonemeStart
        self.trackList[self.activeTrack].notes[index].phonemeStart = self.trackList[self.activeTrack].notes[index + 1].phonemeStart
        self.trackList[self.activeTrack].notes[index + 1].phonemeEnd = self.trackList[self.activeTrack].notes[index].phonemeEnd
        self.trackList[self.activeTrack].notes[index].phonemeEnd = self.trackList[self.activeTrack].notes[index].phonemeStart + phonemeMid
        self.trackList[self.activeTrack].notes[index + 1].phonemeStart = self.trackList[self.activeTrack].notes[index].phonemeStart + phonemeMid
        self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart:self.trackList[self.activeTrack].notes[index].phonemeEnd] = seq2
        self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index + 1].phonemeStart:self.trackList[self.activeTrack].notes[index + 1].phonemeEnd] = seq1
        self.trackList[self.activeTrack].borders[3 * self.trackList[self.activeTrack].notes[index].phonemeStart:3 * self.trackList[self.activeTrack].notes[index].phonemeEnd] = brd2
        self.trackList[self.activeTrack].borders[3 * self.trackList[self.activeTrack].notes[index + 1].phonemeStart:3 * self.trackList[self.activeTrack].notes[index + 1].phonemeEnd] = brd1
        self.makeAutoPauses(index + 1)
        self.submitNamedPhonParamChange(False, "phonemes", self.trackList[self.activeTrack].notes[index].phonemeStart, seq2)
        self.submitNamedPhonParamChange(False, "phonemes", self.trackList[self.activeTrack].notes[index + 1].phonemeStart, seq1)
        self.repairBorders(index + 1)#does not check entire changed border interval
        self.repairBorders(index)#does not check entire changed border interval
        self.submitNamedPhonParamChange(False, "borders", 3 * self.trackList[self.activeTrack].notes[index].phonemeStart, brd2)
        self.submitNamedPhonParamChange(False, "borders", 3 * self.trackList[self.activeTrack].notes[index + 1].phonemeStart, brd1)

    def scaleNote(self, index, oldLength):
        length = self.trackList[self.activeTrack].notes[index].length
        iterationEnd = self.trackList[self.activeTrack].notes[index].phonemeEnd
        iterationEndBorder = iterationEnd
        if index + 1 < len(self.trackList[self.activeTrack].notes):
            length = max(min(length, self.trackList[self.activeTrack].notes[index + 1].xPos - self.trackList[self.activeTrack].notes[index].xPos), 1)
        else:
            iterationEndBorder += 1
        iterationStart = self.trackList[self.activeTrack].notes[index].phonemeStart
        lengthDeltas = []
        if iterationEnd > iterationStart:
            if self.trackList[self.activeTrack].phonemes[iterationStart] == "_autopause":
                iterationStart += 1
        for i in range(3 * iterationStart, 3 * iterationEndBorder - 1):
            lengthDeltas.append(None)
        for i in range(iterationStart, iterationEnd):
            if self.trackList[self.activeTrack].phonemeLengths[self.trackList[self.activeTrack].phonemes[i]] != None:
                lengthDeltas[3 * (i - iterationStart) + 2] = self.trackList[self.activeTrack].borders[3 * i + 3] - self.trackList[self.activeTrack].borders[3 * i + 2]
                lengthDeltas[3 * (i - iterationStart) + 3] = self.trackList[self.activeTrack].borders[3 * i + 4] - self.trackList[self.activeTrack].borders[3 * i + 3]
                lengthDeltas[3 * (i - iterationStart) + 4] = self.trackList[self.activeTrack].borders[3 * i + 5] - self.trackList[self.activeTrack].borders[3 * i + 4]
                if lengthDeltas[3 * (i - iterationStart)] == None:
                    lengthDeltas[3 * (i - iterationStart)] = self.trackList[self.activeTrack].borders[3 * i + 1] - self.trackList[self.activeTrack].borders[3 * i]
                    lengthDeltas[3 * (i - iterationStart) + 1] = self.trackList[self.activeTrack].borders[3 * i + 2] - self.trackList[self.activeTrack].borders[3 * i + 1]
                else:
                    lengthDeltas[3 * (i - iterationStart)] = min(lengthDeltas[3 * i], self.trackList[self.activeTrack].borders[3 * i + 1] - self.trackList[self.activeTrack].borders[3 * i])
                    lengthDeltas[3 * (i - iterationStart) + 1] = min(lengthDeltas[3 * i + 1], self.trackList[self.activeTrack].borders[3 * i + 2] - self.trackList[self.activeTrack].borders[3 * i + 1])
        staticLength = 0.
        for i in lengthDeltas:
            if i != None:
                staticLength += i
        for i in range(len(lengthDeltas)):
            if lengthDeltas[i] == None:
                lengthDeltas[i] = (self.trackList[self.activeTrack].borders[3 * iterationStart + i + 1] - self.trackList[self.activeTrack].borders[3 * iterationStart + i]) * ((length - staticLength) / (oldLength - staticLength))
        counter = 0
        if len(lengthDeltas) > 0:
            correction = min(length / sum(lengthDeltas), 1.)
        else:
            correction = 1.
        for i in range(3 * iterationStart, 3 * iterationEndBorder):
            self.trackList[self.activeTrack].borders[i] = self.trackList[self.activeTrack].notes[index].xPos + counter
            if i + 1 < 3 * iterationEndBorder:
                counter += lengthDeltas[i - 3 * iterationStart] * correction
        for i in range(3 * iterationStart, 3 * iterationEndBorder):
            self.repairBorders(i)
        self.submitNamedPhonParamChange(False, "borders", 3 * iterationStart, self.trackList[self.activeTrack].borders[3 * iterationStart:3 * iterationEndBorder])
    
    def adjustNote(self, index, oldLength, oldPos):
        result = None
        nextLength = oldLength
        if index + 1 < len(self.trackList[self.activeTrack].notes):
            if self.trackList[self.activeTrack].notes[index].xPos > self.trackList[self.activeTrack].notes[index + 1].xPos:
                nextLength = self.trackList[self.activeTrack].notes[index + 1].length
                if index + 2 < len(self.trackList[self.activeTrack].notes):
                    nextLength = max(min(nextLength, self.trackList[self.activeTrack].notes[index + 2].xPos - self.trackList[self.activeTrack].notes[index + 1].xPos), 1)
                self.switchNote(index)
                result = True
                self.scaleNote(index + 1, oldLength)
        self.scaleNote(index, nextLength)
        if index > 0:
            if self.trackList[self.activeTrack].notes[index - 1].xPos > self.trackList[self.activeTrack].notes[index].xPos:
                self.switchNote(index - 1)
                result = False
                self.scaleNote(index, max(min(oldPos - self.trackList[self.activeTrack].notes[index - 1].xPos, self.trackList[self.activeTrack].notes[index - 1].length), 1))
                self.scaleNote(index - 1, oldLength)
            else:
                self.scaleNote(index - 1, max(min(oldPos - self.trackList[self.activeTrack].notes[index - 1].xPos, self.trackList[self.activeTrack].notes[index - 1].length), 1))
        self.repairNotes(index)
        self.makeAutoPauses(index)
        if index > 0:
            self.recalculateBasePitch(index - 1, self.trackList[self.activeTrack].notes[index - 1].xPos, max(min(oldPos, self.trackList[self.activeTrack].notes[index - 1].xPos + self.trackList[self.activeTrack].notes[index - 1].length), 1))
        self.recalculateBasePitch(index, oldPos, oldPos + oldLength)
        if index + 1 < len(self.trackList[self.activeTrack].notes):
            self.recalculateBasePitch(index + 1, oldPos + oldLength, oldPos + oldLength + nextLength)
        return result

    def addNote(self, index, x, y, reference):
        if self.activeTrack == None:
            return
        if index == 0:
            self.trackList[self.activeTrack].notes.insert(index, dh.Note(x, y, 0, 0, reference))
            if len(self.trackList[self.activeTrack].notes) == 1:
                self.trackList[self.activeTrack].borders[0] = x + 33
                self.trackList[self.activeTrack].borders[1] = x + 66
                self.trackList[self.activeTrack].borders[2] = x + 100
                self.submitNamedPhonParamChange(False, "borders", 0, [33, 66, 100])
        elif index == len(self.trackList[self.activeTrack].notes):
            self.trackList[self.activeTrack].notes.append(dh.Note(x, y, self.trackList[self.activeTrack].notes[index - 1].phonemeEnd, self.trackList[self.activeTrack].notes[index - 1].phonemeEnd, reference))
            self.trackList[self.activeTrack].borders[3 * len(self.trackList[self.activeTrack].phonemes)] = x + 33
            self.trackList[self.activeTrack].borders[3 * len(self.trackList[self.activeTrack].phonemes) + 1] = x + 66
            self.trackList[self.activeTrack].borders[3 * len(self.trackList[self.activeTrack].phonemes) + 2] = x + 100
            self.repairBorders(3 * len(self.trackList[self.activeTrack].phonemes) + 2)
            self.repairBorders(3 * len(self.trackList[self.activeTrack].phonemes) + 1)
            self.repairBorders(3 * len(self.trackList[self.activeTrack].phonemes))
        else:
            self.trackList[self.activeTrack].notes.insert(index, dh.Note(x, y, self.trackList[self.activeTrack].notes[index].phonemeStart, self.trackList[self.activeTrack].notes[index].phonemeStart, reference))
        self.adjustNote(index, 100, x)

    def removeNote(self, index):
        self.offsetPhonemes(index, self.trackList[self.activeTrack].notes[index].phonemeStart - self.trackList[self.activeTrack].notes[index].phonemeEnd)
        self.trackList[self.activeTrack].notes.pop(index)
        if index < len(self.trackList[self.activeTrack].notes):
            self.adjustNote(index, self.trackList[self.activeTrack].notes[index].length, self.trackList[self.activeTrack].notes[index].xPos)

    def changeNoteLength(self, index, x, length):
        iterationEnd = self.trackList[self.activeTrack].notes[index].phonemeEnd
        if index + 1 == len(self.trackList[self.activeTrack].notes):
            iterationEnd += 1
        for i in range(self.trackList[self.activeTrack].notes[index].phonemeStart, iterationEnd):
            self.trackList[self.activeTrack].borders[3 * i] += x - self.trackList[self.activeTrack].notes[index].xPos
            self.trackList[self.activeTrack].borders[3 * i + 1] += x - self.trackList[self.activeTrack].notes[index].xPos
            self.trackList[self.activeTrack].borders[3 * i + 2] += x - self.trackList[self.activeTrack].notes[index].xPos
            self.repairBorders(3 * i + 2)
            self.repairBorders(3 * i + 1)
            self.repairBorders(3 * i)
        oldLength = self.trackList[self.activeTrack].notes[index].length
        if index + 1 < len(self.trackList[self.activeTrack].notes):
            oldLength = min(oldLength, self.trackList[self.activeTrack].notes[index + 1].xPos - self.trackList[self.activeTrack].notes[index].xPos)
        oldLength = max(oldLength, 1)
        oldPos = self.trackList[self.activeTrack].notes[index].xPos
        self.trackList[self.activeTrack].notes[index].length = length
        self.trackList[self.activeTrack].notes[index].xPos = x
        return self.adjustNote(index, oldLength, oldPos)

    def moveNote(self, index, x, y):
        iterationEnd = self.trackList[self.activeTrack].notes[index].phonemeEnd
        iterationStart = self.trackList[self.activeTrack].notes[index].phonemeStart
        if iterationEnd > iterationStart:
            if self.trackList[self.activeTrack].phonemes[iterationStart] == "_autopause":
                iterationStart += 1
        if index + 1 == len(self.trackList[self.activeTrack].notes):
            iterationEnd += 1
        elif self.trackList[self.activeTrack].phonemes[iterationEnd] == "_autopause":
            iterationEnd += 1
        for i in range(iterationStart, iterationEnd):
            self.trackList[self.activeTrack].borders[3 * i] += x - self.trackList[self.activeTrack].notes[index].xPos
            self.trackList[self.activeTrack].borders[3 * i + 1] += x - self.trackList[self.activeTrack].notes[index].xPos
            self.trackList[self.activeTrack].borders[3 * i + 2] += x - self.trackList[self.activeTrack].notes[index].xPos
            self.repairBorders(3 * i + 2)
            self.repairBorders(3 * i + 1)
            self.repairBorders(3 * i)
        self.submitNamedPhonParamChange(False, "borders", 3 * iterationStart, self.trackList[self.activeTrack].borders[3 * iterationStart:3 * iterationEnd])
        oldLength = self.trackList[self.activeTrack].notes[index].length
        if index + 1 < len(self.trackList[self.activeTrack].notes):
            oldLength = min(oldLength, self.trackList[self.activeTrack].notes[index + 1].xPos - self.trackList[self.activeTrack].notes[index].xPos)
        oldLength = max(oldLength, 1)
        oldPos = self.trackList[self.activeTrack].notes[index].xPos
        self.trackList[self.activeTrack].notes[index].xPos = x
        self.trackList[self.activeTrack].notes[index].yPos = y
        return self.adjustNote(index, oldLength, oldPos)

    def changeLyrics(self, index, text):
        self.trackList[self.activeTrack].notes[index].content = text
        if self.trackList[self.activeTrack].notes[index].phonemeMode:
            text = text.split(" ")
        else:
            text = []#TO DO: Dictionary lookup here
        phonemes = []
        for i in text:
            if i in self.trackList[self.activeTrack].phonemeLengths:
                phonemes.append(i)
        if phonemes == [""]:
            phonemes = []
        if (len(self.trackList[self.activeTrack].phonemes) > 0) and (self.trackList[self.activeTrack].notes[index].phonemeStart < self.trackList[self.activeTrack].notes[index].phonemeEnd):
            if self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart] == "_autopause":
                phonemes.insert(0, "_autopause")
        offset = len(phonemes) - self.trackList[self.activeTrack].notes[index].phonemeEnd + self.trackList[self.activeTrack].notes[index].phonemeStart
        print("offset", offset)
        self.offsetPhonemes(index, offset, futurePhonemes = phonemes)
        self.trackList[self.activeTrack].phonemes[self.trackList[self.activeTrack].notes[index].phonemeStart:self.trackList[self.activeTrack].notes[index].phonemeEnd] = phonemes
        self.submitNamedPhonParamChange(False, "phonemes", self.trackList[self.activeTrack].notes[index].phonemeStart, phonemes)
        self.makeAutoPauses(index)
        self.submitFinalize()
    def changeBorder(self, border, pos):
        self.trackList[self.activeTrack].borders[border] = pos
        self.repairBorders(border)
        self.submitBorderChange(True, border, [pos,])
    def repairNotes(self, index):
        if index == 0 or index == len(self.trackList[self.activeTrack].notes):
            return
        if self.trackList[self.activeTrack].notes[index].xPos == self.trackList[self.activeTrack].notes[index - 1].xPos:
            self.trackList[self.activeTrack].notes[index].xPos += 1
            self.repairNotes(index + 1)
    def repairBorders(self, index):
        if index == 0 or index == len(self.trackList[self.activeTrack].borders):
            return None
        if self.trackList[self.activeTrack].borders[index] < self.trackList[self.activeTrack].borders[index - 1] + 1.:
            self.trackList[self.activeTrack].borders[index] = self.trackList[self.activeTrack].borders[index - 1] + 1.
            self.submitNamedPhonParamChange(False, "borders", index, [self.trackList[self.activeTrack].borders[index],])
            self.repairBorders(index + 1)
    def submitTerminate(self):
        self.manager.sendChange("terminate", True)
    def submitAddTrack(self, track):
        self.manager.sendChange("addTrack", True, track.vbPath, track.to_sequence(), self.aiParamStackList[-1])
    def submitRemoveTrack(self, index):
        self.manager.sendChange("removeTrack", True, index)
    def submitDuplicateTrack(self, index):
        self.manager.sendChange("duplicateTrack", True, index)
    def submitChangeVB(self, index, path):
        self.manager.sendChange("changeVB", True, index, path)
    def submitAddParam(self, path):
        self.manager.sendChange("addParam", True, self.activeTrack, path)
    def submitRemoveParam(self, index):
        self.manager.sendChange("removeParam", True, self.activeTrack, index)
    def submitEnableParam(self, index, name):
        if index == -1:
            self.manager.sendChange("enableParam", True, self.activeTrack, name)
        else:
            self.manager.sendChange("enableParam", True, self.activeTrack, index)
    def submitDisableParam(self, index, name):
        if index == -1:
            self.manager.sendChange("disableParam", True, self.activeTrack, name)
        else:
            self.manager.sendChange("disableParam", True, self.activeTrack, index)
    def submitBorderChange(self, final, index, data):
        self.manager.sendChange("changeInput", final, self.activeTrack, "borders", index, data)
    def submitNamedParamChange(self, final, param, index, data):
        self.manager.sendChange("changeInput", final, self.activeTrack, param, index, data)
    def submitNamedPhonParamChange(self, final, param, index, data):
        self.manager.sendChange("changeInput", final, self.activeTrack, param, index, data)
    def submitParamChange(self, final, param, index, data):
        self.manager.sendChange("changeInput", final, self.activeTrack, param, index, data)
    def submitOffset(self, final, index, offset):
        self.manager.sendChange("offset", final, self.activeTrack, index, offset)
    def submitFinalize(self):
        self.manager.sendChange("finalize", True)
    def updateRenderStatus(self, track, index, value):
        for i in self.deletions:
            if i == track:
                return None
            elif i < track:
                track -= 1
    def updateVolume(self, index, volume):
        self.trackList[index].volume = volume
    def updateAudioBuffer(self, track, index, data):
        for i in self.deletions:
            if i == track:
                return None
            elif i < track:
                track -= 1
        self.audioBuffer[track][index:index + len(data)] = data
    def movePlayhead(self, position):
        self.ids["pianoRoll"].changePlaybackPos(position)
    def play(self, state = None):
        if state == None:
            state = not(self.playing)
        if state == True:
            self.ids["playButton"].state = "down"
            self.audioStream.start()
        if state == False:
            self.ids["playButton"].state = "normal"
            self.audioStream.stop()
        self.playing = state
    def playCallback(self, outdata, frames, time, status):
        if self.playing:
            newBufferPos = self.mainAudioBufferPos + global_consts.audioBufferSize
            mainAudioBuffer = torch.zeros([newBufferPos - self.mainAudioBufferPos],)
            volumes = []
            for i in range(len(self.audioBuffer)):
                buffer = self.audioBuffer[i][self.mainAudioBufferPos:newBufferPos] * self.trackList[i].volume
                mainAudioBuffer += buffer
                volumes.append(buffer.abs().max())
            for i in self.ids["singerList"].children:
                i.children[0].children[0].children[0].children[0].value = volumes[i.index]
            buffer = mainAudioBuffer.expand(2, -1).transpose(0, 1).numpy()
            self.mainAudioBufferPos = newBufferPos
            self.movePlayhead(int(self.mainAudioBufferPos / global_consts.batchSize))
        else:
            buffer = torch.zeros([global_consts.audioBufferSize, 2], dtype = torch.float32).expand(-1, 2).numpy()
        outdata[:] = buffer.copy()
    def noteToPitch(self, data):
        #return torch.full_like(data, global_consts.sampleRate) / (torch.pow(2, (data - torch.full_like(data, 69)) / torch.full_like(data, 12)) * 440)
        return torch.full_like(data, global_consts.sampleRate) / (torch.pow(2, (data - torch.full_like(data, 69 - 36)) / torch.full_like(data, 12)) * 440)
    def recalculateBasePitch(self, index, oldStart, oldEnd):
        dipWidth = global_consts.pitchDipWidth
        dipHeight = global_consts.pitchDipHeight
        transitionLength1 = min(global_consts.pitchTransitionLength, int(self.trackList[self.activeTrack].notes[index].length))
        transitionLength2 = min(global_consts.pitchTransitionLength, int(self.trackList[self.activeTrack].notes[index].length))
        currentHeight = self.trackList[self.activeTrack].notes[index].yPos
        transitionPoint1 = self.trackList[self.activeTrack].notes[index].xPos
        transitionPoint2 = self.trackList[self.activeTrack].notes[index].xPos + self.trackList[self.activeTrack].notes[index].length
        if index == 0:
            previousHeight = None
        else:
            previousHeight = self.trackList[self.activeTrack].notes[index - 1].yPos
            transitionLength1 = min(transitionLength1, self.trackList[self.activeTrack].notes[index - 1].length)
            if self.trackList[self.activeTrack].notes[index - 1].xPos + self.trackList[self.activeTrack].notes[index - 1].length < transitionPoint1:
                transitionLength1 += transitionPoint1 - self.trackList[self.activeTrack].notes[index - 1].xPos - self.trackList[self.activeTrack].notes[index - 1].length
                transitionPoint1 = (transitionPoint1 + self.trackList[self.activeTrack].notes[index - 1].xPos + self.trackList[self.activeTrack].notes[index - 1].length) / 2  
        if index + 1 == len(self.trackList[self.activeTrack].notes):
            nextHeight = None
        else:
            transitionPoint2 = self.trackList[self.activeTrack].notes[index + 1].xPos
            nextHeight = self.trackList[self.activeTrack].notes[index + 1].yPos
            transitionLength2 = min(transitionLength2, self.trackList[self.activeTrack].notes[index + 1].length)
            if self.trackList[self.activeTrack].notes[index].xPos + self.trackList[self.activeTrack].notes[index].length < transitionPoint1:
                transitionLength2 += transitionPoint2 - self.trackList[self.activeTrack].notes[index].xPos - self.trackList[self.activeTrack].notes[index].length
                transitionPoint2 = (transitionPoint2 + self.trackList[self.activeTrack].notes[index].xPos + self.trackList[self.activeTrack].notes[index].length) / 2
        scalingFactor = min(self.trackList[self.activeTrack].notes[index].length / 2 / dipWidth, 1.)
        dipWidth *= scalingFactor
        dipHeight *= scalingFactor
        start = int(transitionPoint1 - math.ceil(transitionLength1 / 2))
        end = int(transitionPoint2 + math.ceil(transitionLength2 / 2))
        transitionPoint1 = int(transitionPoint1)
        transitionPoint2 = int(transitionPoint2)
        if previousHeight == None:
            start =  self.trackList[self.activeTrack].notes[index].xPos
        if nextHeight == None:
            end = self.trackList[self.activeTrack].notes[index].xPos + self.trackList[self.activeTrack].notes[index].length
        pitchDelta = (self.trackList[self.activeTrack].pitch[start:end] - self.trackList[self.activeTrack].basePitch[start:end]) * torch.heaviside(self.trackList[self.activeTrack].basePitch[start:end], torch.ones_like(self.trackList[self.activeTrack].basePitch[start:end]))
        self.trackList[self.activeTrack].basePitch[oldStart:oldEnd] = torch.full_like(self.trackList[self.activeTrack].basePitch[oldStart:oldEnd], -1.)
        self.trackList[self.activeTrack].pitch[oldStart:oldEnd] = torch.full_like(self.trackList[self.activeTrack].basePitch[oldStart:oldEnd], -1.)
        self.trackList[self.activeTrack].basePitch[start:end] = torch.full_like(self.trackList[self.activeTrack].basePitch[start:end], currentHeight)
        if previousHeight != None:
            self.trackList[self.activeTrack].basePitch[transitionPoint1 - math.ceil(transitionLength1 / 2):transitionPoint1 + math.ceil(transitionLength1 / 2)] = torch.pow(torch.cos(torch.linspace(0, math.pi / 2, 2 * math.ceil(transitionLength1 / 2))), 2) * (previousHeight - currentHeight) + torch.full([2 * math.ceil(transitionLength1 / 2),], currentHeight)
        if nextHeight != None:
            self.trackList[self.activeTrack].basePitch[transitionPoint2 - math.ceil(transitionLength2 / 2):transitionPoint2 + math.ceil(transitionLength2 / 2)] = torch.pow(torch.cos(torch.linspace(0, math.pi / 2, 2 * math.ceil(transitionLength2 / 2))), 2) * (currentHeight - nextHeight) + torch.full([2 * math.ceil(transitionLength2 / 2),], nextHeight)
        self.trackList[self.activeTrack].basePitch[self.trackList[self.activeTrack].notes[index].xPos:self.trackList[self.activeTrack].notes[index].xPos + int(dipWidth)] -= torch.pow(torch.sin(torch.linspace(0, math.pi, int(dipWidth))), 2) * dipHeight
        self.trackList[self.activeTrack].pitch[start:end] = self.trackList[self.activeTrack].basePitch[start:end] + torch.heaviside(self.trackList[self.activeTrack].basePitch[start:end], torch.ones_like(self.trackList[self.activeTrack].basePitch[start:end])) * pitchDelta
        self.applyPitchChanges(self.trackList[self.activeTrack].pitch[start:end], start)
