from logging import root
from turtle import pos
from typing import Text
from kivy.core.image import Image as CoreImage
from PIL import Image as PilImage, ImageDraw, ImageFont

from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.behaviors import ButtonBehavior, ToggleButtonBehavior
from kivy.uix.image import Image
from kivy.properties import StringProperty, ObjectProperty, BooleanProperty, NumericProperty, ListProperty, OptionProperty
from kivy.graphics import Color, Line, Rectangle, InstructionGroup
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.modalview import ModalView
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.bubble import Bubble, BubbleButton

from io import BytesIO

from kivy.clock import mainthread

import os
import torch
import subprocess
import math

import MiddleLayer.DataHandlers as dh
from Backend.NV_Multiprocessing.Manager import RenderManager

class MiddleLayer(Widget):
    def __init__(self, ids, **kwargs):
        super().__init__(**kwargs)
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
        self.seqLength = 0
    def importVoicebank(self, path, name, inImage):
        self.trackList.append(dh.Track(path))
        canvas_img = inImage
        data = BytesIO()
        canvas_img.save(data, format='png')
        data.seek(0)
        im = CoreImage(BytesIO(data.read()), ext='png')
        image = im.texture
        self.ids["singerList"].add_widget(SingerPanel(name = name, image = image, index = len(self.trackList) - 1))
    def importParam(self, path, name):
        self.trackList[self.activeTrack].paramStack.append(dh.Parameter(path))
        self.ids["paramList"].add_widget(ParamPanel(name = name, switchable = True, sortable = True, deletable = True, index = len(self.trackList[self.activeTrack].paramStack) - 1))
    def changeTrack(self, index):
        self.activeTrack = index
        self.updateParamPanel()
    def copyTrack(self, index, name, inImage):
        self.trackList.append(self.trackList[index])
        image = inImage
        self.ids["singerList"].add_widget(SingerPanel(name = name, image = image, index = len(self.trackList) - 1))
    def deleteTrack(self, index):
        self.trackList.pop(index)
        if index <= self.activeTrack:
            self.changeTrack(self.activeTrack - 1)
        for i in self.ids["singerList"].children:
            if i.index == index:
                i.parent.remove_widget(i)
            if i.index > index:
                i.index = i.index - 1
    def deleteParam(self, index):
        self.trackList[self.activeTrack].paramStack.pop(index)
        if index <= self.activeParam:
            self.changeParam(self.activeParam - 1)
        for i in self.ids["paramList"].children:
            if i.index == index:
                i.parent.remove_widget(i)
            if i.index > index:
                i.index = i.index - 1
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
    def moveParam(self, name, switchable, sortable, deletable, index, delta):
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
        self.ids["paramList"].add_widget(ParamPanel(name = name, switchable = switchable, sortable = sortable, deletable = deletable, index = index), index = index + delta)
        self.changeParam(index + delta)
    def updateParamPanel(self):
        self.ids["paramList"].clear_widgets()
        self.ids["adaptiveSpace"].clear_widgets()
        if self.mode == "notes":
            self.ids["paramList"].add_widget(ParamPanel(name = "steadiness", switchable = True, sortable = False, deletable = False, index = -1, state = "down"))
            self.ids["paramList"].add_widget(ParamPanel(name = "breathiness", switchable = True, sortable = False, deletable = False, index = -1))
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
            self.ids["paramList"].add_widget(ParamPanel(name = "vibrato speed", switchable = True, sortable = False, deletable = False, index = -1, state = "down"))
            self.ids["paramList"].add_widget(ParamPanel(name = "vibrato strength", switchable = True, sortable = False, deletable = False, index = -1))
            self.ids["adaptiveSpace"].add_widget(PitchOptns())
            self.changeParam(-1, "vibrato speed")
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
        elif self.activeParam == "breathiness":
            self.trackList[self.activeTrack].breathiness[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
        elif self.activeParam == "loop":
            if section:
                self.trackList[self.activeTrack].loopOverlap[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
            else:
                self.trackList[self.activeTrack].loopOffset[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
        elif self.activeParam == "vibrato":
            if section:
                self.trackList[self.activeTrack].vibratoSpeed[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
            else:
                self.trackList[self.activeTrack].vibratoStrength[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
        else:
            self.trackList[self.activeTrack].paramStack[self.activeParam].curve[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
    def applyPitchChanges(self, data, start):
        self.trackList[self.activeTrack].pitch[start:start + len(data)] = torch.tensor(data, dtype = torch.float32)
    def changePianoRollMode(self):
        self.ids["pianoRoll"].changeMode()
    def applyScroll(self):
        self.ids["pianoRoll"].applyScroll(self.scrollValue)
        self.ids["adaptiveSpace"].applyScroll(self.scrollValue)
    def offsetPhonemes(self, index, offset):
        phonIndex = self.trackList[self.activeTrack].notes[index].phonemeStart
        if offset > 0:
            for i in range(offset):
                self.trackList[self.activeTrack].sequence.insert(phonIndex, "X")
                self.trackList[self.activeTrack].loopOverlap = torch.cat([self.trackList[self.activeTrack].loopOverlap[0:phonIndex], torch.tensor([0.5], dtype = torch.half), self.trackList[self.activeTrack].loopOverlap[phonIndex:]], dim = 0)
                self.trackList[self.activeTrack].loopOffset = torch.cat([self.trackList[self.activeTrack].loopOffset[0:phonIndex], torch.tensor([0.5], dtype = torch.half), self.trackList[self.activeTrack].loopOffset[phonIndex:]], dim = 0)
                for j in range(3):
                    self.trackList[self.activeTrack].borders.insert(phonIndex * 3, 0)
        elif offset < 0:
            for i in range(-offset):
                self.trackList[self.activeTrack].sequence.pop(phonIndex)
                self.trackList[self.activeTrack].loopOverlap = torch.cat([self.trackList[self.activeTrack].loopOverlap[0:phonIndex], self.trackList[self.activeTrack].loopOverlap[phonIndex + 1:]], dim = 0)
                self.trackList[self.activeTrack].loopOffset = torch.cat([self.trackList[self.activeTrack].loopOffset[0:phonIndex], self.trackList[self.activeTrack].loopOffset[phonIndex + 1:]], dim = 0)
                for j in range(3):
                    self.trackList[self.activeTrack].borders.pop(phonIndex * 3)
        for i in self.trackList[self.activeTrack].notes[index + 1:]:
            i.phonemeStart += offset
            i.phonemeEnd += offset
        self.trackList[self.activeTrack].notes[index].phonemeEnd += offset
    def makeAutoPauses(self, index):
        if index > 0:
            if self.trackList[self.activeTrack].sequence[self.trackList[self.activeTrack].notes[index].phonemeStart] == "_autopause":
                self.trackList[self.activeTrack].sequence.pop(self.trackList[self.activeTrack].notes[index].phonemeStart)
                for i in self.trackList[self.activeTrack].notes[index:]:
                    i.phonemeStart -= 1
                    i.phonemeEnd -= 1
            if self.trackList[self.activeTrack].notes[index].xPos - self.trackList[self.activeTrack].notes[index - 1].xPos - self.trackList[self.activeTrack].notes[index - 1].length > self.trackList[self.activeTrack].pauseThreshold:
                self.trackList[self.activeTrack].sequence.insert(self.trackList[self.activeTrack].notes[index].phonemeStart, "_autopause")
                for i in self.trackList[self.activeTrack].notes[index:]:
                    i.phonemeStart += 1
                    i.phonemeEnd += 1
        if index < len(self.trackList[self.activeTrack].notes - 1):
            if self.trackList[self.activeTrack].sequence[self.trackList[self.activeTrack].notes[index].phonemeEnd] == "_autopause":
                self.trackList[self.activeTrack].sequence.pop(self.trackList[self.activeTrack].notes[index].phonemeEnd)
                for i in self.trackList[self.activeTrack].notes[index + 1:]:
                    i.phonemeStart -= 1
                    i.phonemeEnd -= 1
            if self.trackList[self.activeTrack].notes[index + 1].xPos - self.trackList[self.activeTrack].notes[index].xPos - self.trackList[self.activeTrack].notes[index].length > self.trackList[self.activeTrack].pauseThreshold:
                self.trackList[self.activeTrack].sequence.insert(self.trackList[self.activeTrack].notes[index + 1].phonemeStart, "_autopause")
                for i in self.trackList[self.activeTrack].notes[index + 1:]:
                    i.phonemeStart += 1
                    i.phonemeEnd += 1
    def switchNote(self, index):
        note = self.trackList[self.activeTrack].notes.pop(index + 1)
        seq1 = self.trackList[self.activeTrack].sequence[self.trackList[self.activeTrack].notes[index].phonemeStart:self.trackList[self.activeTrack].notes[index].phonemeEnd]
        seq2 = self.trackList[self.activeTrack].sequence[note.phonemeStart:note.phonemeEnd]
        brd1 = self.trackList[self.activeTrack].borders[3 * self.trackList[self.activeTrack].notes[index].phonemeStart:3 * self.trackList[self.activeTrack].notes[index].phonemeEnd]
        brd2 = self.trackList[self.activeTrack].borders[3 * note.phonemeStart:3 * note.phonemeEnd]
        if index == len(self.trackList[self.activeTrack].notes) - 1:#notes are 1 element shorter because of pop
            scalingFactor = (self.trackList[self.activeTrack].notes[index].phonemeEnd - self.trackList[self.activeTrack].notes[index].phonemeStart + 1) / (note.phonemeEnd - note.phonemeStart + 1)
            print("last", scalingFactor)
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
        self.trackList[self.activeTrack].sequence[self.trackList[self.activeTrack].notes[index].phonemeStart:self.trackList[self.activeTrack].notes[index].phonemeEnd] = seq2
        self.trackList[self.activeTrack].sequence[self.trackList[self.activeTrack].notes[index + 1].phonemeStart:self.trackList[self.activeTrack].notes[index + 1].phonemeEnd] = seq1
        self.trackList[self.activeTrack].borders[3 * self.trackList[self.activeTrack].notes[index].phonemeStart:3 * self.trackList[self.activeTrack].notes[index].phonemeEnd] = brd2
        self.trackList[self.activeTrack].borders[3 * self.trackList[self.activeTrack].notes[index + 1].phonemeStart:3 * self.trackList[self.activeTrack].notes[index + 1].phonemeEnd] = brd1
    def scaleNote(self, index, oldLength):
        length = self.trackList[self.activeTrack].notes[index].length
        iterationEnd = self.trackList[self.activeTrack].notes[index].phonemeEnd
        if index + 1 < len(self.trackList[self.activeTrack].notes):
            length = max(min(length, self.trackList[self.activeTrack].notes[index + 1].xPos - self.trackList[self.activeTrack].notes[index].xPos), 1)
        else:
            iterationEnd += 1
        for i in range(self.trackList[self.activeTrack].notes[index].phonemeStart, iterationEnd):
            self.trackList[self.activeTrack].borders[3 * i] = (self.trackList[self.activeTrack].borders[3 * i] - self.trackList[self.activeTrack].notes[index].xPos) * length / oldLength + self.trackList[self.activeTrack].notes[index].xPos
            self.trackList[self.activeTrack].borders[3 * i + 1] = (self.trackList[self.activeTrack].borders[3 * i + 1] - self.trackList[self.activeTrack].notes[index].xPos) * length / oldLength + self.trackList[self.activeTrack].notes[index].xPos
            self.trackList[self.activeTrack].borders[3 * i + 2] = (self.trackList[self.activeTrack].borders[3 * i + 2] - self.trackList[self.activeTrack].notes[index].xPos) * length / oldLength + self.trackList[self.activeTrack].notes[index].xPos
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
        return result
    def addNote(self, index, x, y, reference):
        if index == 0:
            self.trackList[self.activeTrack].notes.insert(index, dh.Note(x, y, 0, 0, reference))
            if len(self.trackList[self.activeTrack].notes) == 1:
                self.trackList[self.activeTrack].borders[0] = x + 33
                self.trackList[self.activeTrack].borders[1] = x + 66
                self.trackList[self.activeTrack].borders[2] = x + 100
        elif index == len(self.trackList[self.activeTrack].notes):
            self.trackList[self.activeTrack].notes.append(dh.Note(x, y, self.trackList[self.activeTrack].notes[index - 1].phonemeEnd, self.trackList[self.activeTrack].notes[index - 1].phonemeEnd, reference))
            self.trackList[self.activeTrack].borders[3 * len(self.trackList[self.activeTrack].sequence)] = x + 33
            self.trackList[self.activeTrack].borders[3 * len(self.trackList[self.activeTrack].sequence) + 1] = x + 66
            self.trackList[self.activeTrack].borders[3 * len(self.trackList[self.activeTrack].sequence) + 2] = x + 100
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
        if index + 1 == len(self.trackList[self.activeTrack].notes):
            iterationEnd += 1
        for i in range(self.trackList[self.activeTrack].notes[index].phonemeStart, iterationEnd):
            self.trackList[self.activeTrack].borders[3 * i] += x - self.trackList[self.activeTrack].notes[index].xPos
            self.trackList[self.activeTrack].borders[3 * i + 1] += x - self.trackList[self.activeTrack].notes[index].xPos
            self.trackList[self.activeTrack].borders[3 * i + 2] += x - self.trackList[self.activeTrack].notes[index].xPos
        oldLength = self.trackList[self.activeTrack].notes[index].length
        if index + 1 < len(self.trackList[self.activeTrack].notes):
            oldLength = min(oldLength, self.trackList[self.activeTrack].notes[index + 1].xPos - self.trackList[self.activeTrack].notes[index].xPos)
        oldLength = max(oldLength, 1)
        oldPos = self.trackList[self.activeTrack].notes[index].xPos
        self.trackList[self.activeTrack].notes[index].xPos = x
        self.trackList[self.activeTrack].notes[index].yPos = y
        return self.adjustNote(index, oldLength, oldPos)
    def changeLyrics(self, index, text, mode):
        self.trackList[self.activeTrack].notes[index].content = text
        if mode:
            text = text.split(" ")
        else:
            text = ""#TO DO: Dictionary lookup here
        phonemes = []
        """
        for i in text:
            if i in self.trackList[self.activeTrack].phonemeDict:
                phonemes.append(i)
        """
        phonemes = text#TO DO: Input validation, phoneme type awareness
        offset = len(phonemes) - self.trackList[self.activeTrack].notes[index].phonemeEnd + self.trackList[self.activeTrack].notes[index].phonemeStart
        self.offsetPhonemes(index, offset)
        self.trackList[self.activeTrack].sequence[self.trackList[self.activeTrack].notes[index].phonemeStart:self.trackList[self.activeTrack].notes[index].phonemeEnd] = phonemes
        start = self.trackList[self.activeTrack].notes[index].xPos
        end = self.trackList[self.activeTrack].notes[index].xPos + self.trackList[self.activeTrack].notes[index].length
        if index < len(self.trackList[self.activeTrack].notes) - 1:
            end = min(end, self.trackList[self.activeTrack].notes[index + 1].xPos)
        divisor = (len(phonemes) + 1) * 3
        iterationEnd = self.trackList[self.activeTrack].notes[index].phonemeEnd
        if index + 1 == len(self.trackList[self.activeTrack].notes):
            iterationEnd += 1
        for i in range(self.trackList[self.activeTrack].notes[index].phonemeStart, iterationEnd):
            j = i - self.trackList[self.activeTrack].notes[index].phonemeStart
            self.trackList[self.activeTrack].borders[3 * i] = start + int((end - start) * (3 * j) / divisor)
            self.trackList[self.activeTrack].borders[3 * i + 1] = start + int((end - start) * (3 * j + 1) / divisor)
            self.trackList[self.activeTrack].borders[3 * i + 2] = start + int((end - start) * (3 * j + 2) / divisor)
        #if self.trackList[self.activeTrack].notes[index].phonemeEnd != "_autopause":
            #self.trackList[self.activeTrack].borders[3 * self.trackList[self.activeTrack].notes[index].phonemeEnd] = end - 2
            #self.trackList[self.activeTrack].borders[3 * self.trackList[self.activeTrack].notes[index].phonemeEnd + 1] = end - 1
            #self.trackList[self.activeTrack].borders[3 * self.trackList[self.activeTrack].notes[index].phonemeEnd + 2] = end
    def changeBorder(self, border, pos):
        self.trackList[self.activeTrack].borders[border] = pos
        self.submitSequenceChanges(math.floor(border / 3), 1)
    def aAaAaAa(self, index):#deprecated
        def repairBorders(self, index):
            if index == len(self.trackList[self.activeTrack].notes):
                return None
            if self.trackList[self.activeTrack].notes[index].xPos == self.trackList[self.activeTrack].notes[index - 1].xPos:
                self.trackList[self.activeTrack].notes[index].xPos += 1
                repairBorders(self, index + 1)
        iterationEnd = self.trackList[self.activeTrack].notes[index].phonemeEnd
        if index + 1 == len(self.trackList[self.activeTrack].notes):
            iterationEnd += 1
        for i in range(3 * self.trackList[self.activeTrack].notes[index].phonemeStart, 3 * iterationEnd):
            self.trackList[self.activeTrack].borders[i] = int(self.trackList[self.activeTrack].borders[i])
        for i in range(3 * self.trackList[self.activeTrack].notes[index].phonemeStart, 3 * iterationEnd):
            repairBorders(self, i)
        return None
    def submitSequenceChanges(self, index, length):
        pass
    def submitParamChanges(self, param):
        pass
    def updateRenderStatus(self, track, index, value):
        pass
    def updateAudioBuffer(self, track, index, value):
        pass
        
class ImageButton(ButtonBehavior, Image):
    imageNormal = StringProperty()
    imagePressed = StringProperty()
    function = ObjectProperty(None)
    def on_press(self):
        self.source = self.imagePressed
        if self.function != None:
            self.function()
        else:
            print("NONE function callback")
    def on_release(self):
        self.source = self.imageNormal

class ImageToggleButton(ToggleButtonBehavior, Image):
    imageNormal = StringProperty()
    imagePressed = StringProperty()
    function = ObjectProperty(None)
    def on_press(self):
        if self.function != None:
            self.function()
        else:
            print("NONE function callback")
    def on_release(self):
        pass

    def on_state(self, widget, value):
        if value == 'down':
            self.source = self.imagePressed
        else:
            self.source = self.imageNormal

class SingerPanel(AnchorLayout):
    name = StringProperty()
    image = ObjectProperty()
    index = NumericProperty()
    def changeTrack(self):
        global middleLayer
        middleLayer.changeTrack(self.index)
    def copyTrack(self):
        global middleLayer
        middleLayer.copyTrack(self.index, self.name, self.image)
    def deleteTrack(self):
        global middleLayer
        middleLayer.deleteTrack(self.index)

class ParamPanel(ToggleButton):
    def __init__(self, name, switchable, sortable, deletable, index, **kwargs):
        super().__init__(**kwargs)
        self.name = StringProperty()
        self.switchable = BooleanProperty()
        self.sortable = BooleanProperty()
        self.index = NumericProperty()
        self.deletable = BooleanProperty()
        self.name = name
        self.switchable = switchable
        self.sortable = sortable
        self.deletable = deletable
        self.index = index
        self.background_color = (1, 1, 1, 0.3)
        self.makeWidgets()
    @mainthread
    def makeWidgets(self):
        self.add_widget(Label(size_hint = (None, None), size = (self.width - 106, 30), pos = (self.x + 103, self.y + 3), text = self.name))
        if self.switchable:
            self.add_widget(ImageToggleButton(size_hint = (None, None), size = (30, 30), pos = (self.x + 3, self.y + 3), imageNormal = "UI/assets/ParamList/Adaptive02.png", imagePressed = "UI/assets/ParamList/Adaptive01.png", on_state = self.enableParam))
        if self.sortable:
            self.add_widget(ImageButton(size_hint = (None, None), size = (40, 30), pos = (self.x + 33, self.y + 3), imageNormal = "UI/assets/ParamList/Adaptive03.png", imagePressed = "UI/assets/ParamList/Adaptive03_clicked.png", on_release = self.moveParam))
        if self.deletable:
            self.add_widget(ImageButton(size_hint = (None, None), size = (30, 30), pos = (self.x + 73, self.y + 3), imageNormal = "UI/assets/TrackList/SingerGrey03.png", imagePressed = "UI/assets/TrackList/SingerGrey03_clicked.png", on_press = self.deleteParam))
    def enableParam(self):
        global middleLayer
        if self.state == "down":
            middleLayer.enableParam(self.index, self.name)
        else:
            middleLayer.disableParam(self.index, self.name)
    def moveParam(self):
        global middleLayer
        delta = 0
        middleLayer.moveParam(self.name, self.switchable, self.sortable, self.index, delta)
    def deleteParam(self):
        global middleLayer
        middleLayer.deleteParam(self.index)
    def changeParam(self):
        global middleLayer
        middleLayer.changeParam(self.index, self.name)

class AdaptiveSpace(AnchorLayout):
    def redraw(self):
        self.children[0].redraw()
    def applyScroll(self, scrollValue):
        self.children[0].scroll_x = scrollValue
    def triggerScroll(self):
        global middleLayer
        middleLayer.scrollValue = self.children[0].scroll_x
        middleLayer.applyScroll()

class ParamCurve(ScrollView):
    xScale = NumericProperty(10)
    seqLength = NumericProperty(1000)
    line = ObjectProperty()
    line = Line()
    def redraw(self):
        if middleLayer.activeParam == "steadiness":
            data = middleLayer.trackList[middleLayer.activeTrack].steadiness
        elif middleLayer.activeParam == "breathiness":
            data = middleLayer.trackList[middleLayer.activeTrack].breathiness
        else:
            data = middleLayer.trackList[middleLayer.activeTrack].paramStack[middleLayer.activeParam].curve
        points = []
        c = 0
        for i in data:
            points.append(c * self.xScale)
            points.append((i.item() + 1) * self.height / 2)
            c += 1
        self.children[0].canvas.remove(self.line)
        with self.children[0].canvas:
            Color(1, 0, 0, 1)
            self.line = Line(points = points)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if touch.is_mouse_scrolling:
                if touch.button == 'scrollup':
                    newvalue = self.scroll_x + self.convert_distance_to_scroll(self.scroll_wheel_distance, 0)[0]
                    if newvalue < 1:
                        self.scroll_x = newvalue
                    else:
                        self.scroll_x = 1.
                elif touch.button == 'scrolldown':
                    newvalue = self.scroll_x - self.convert_distance_to_scroll(self.scroll_wheel_distance, 0)[0]
                    if newvalue > 0:
                        self.scroll_x = newvalue
                    else:
                        self.scroll_x = 0.
            else:
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points=[touch.x, touch.y])
                    touch.ud['startPoint'] = self.to_local(touch.x, touch.y)
                    touch.ud['startPoint'] = [int(touch.ud['startPoint'][0] / self.xScale), min(max(touch.ud['startPoint'][1], 0.), self.height)]
                    touch.ud['startPointOffset'] = 0
            return True
        else:
            return False
    def on_touch_move(self, touch):
        if 'startPoint' in touch.ud:
            global middleLayer
            coord = self.to_local(touch.x, touch.y)
            x = int(coord[0] / self.xScale)
            y = min(max(coord[1], 0.), self.height)
            p = x - touch.ud['startPoint'][0]
            if middleLayer.tool == "draw":
                if p < 0:
                    for i in range(-p):
                        touch.ud['line'].points = [touch.ud['startPoint'][0] - i * self.xScale, y] + touch.ud['line'].points
                        touch.ud['startPoint'][0] -= 1
                elif p < int(len(touch.ud['line'].points) / 2):
                    points = touch.ud['line'].points
                    points[2 * p] = x * self.xScale
                    points[2 * p + 1] = y
                    touch.ud['line'].points = points
                else:
                    diff = p - int(len(touch.ud['line'].points) / 2)
                    for i in range(diff):
                        touch.ud['line'].points += [(touch.ud['startPoint'][0] + int(len(touch.ud['line'].points) / 2)) * self.xScale, y]
            elif middleLayer.tool == "line":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * -i / p]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * i / p]
            elif middleLayer.tool == "arch":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * (-i / p) * (-i / p)]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * (i / p) * (i / p)]
            elif middleLayer.tool == "reset":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, self.height / 2]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, self.height / 2]
        else:
            return super(ParamCurve, self).on_touch_move(touch)
    def on_touch_up(self, touch):
        global middleLayer
        if 'startPoint' in touch.ud:
            data = []
            if touch.ud['startPointOffset'] == 0:
                for i in range(int(len(touch.ud['line'].points) / 2)):
                    data.append((touch.ud['line'].points[2 * i + 1] * 2 / self.height) - 1)
            else:
                for i in range(int(len(touch.ud['line'].points) / 2)):
                    data.append((touch.ud['line'].points[2 * (int(len(touch.ud['line'].points) / 2) - i) - 1] * 2 / self.height) - 1)
            middleLayer.applyParamChanges(data, touch.ud['startPoint'][0] - touch.ud['startPointOffset'])
            self.children[0].canvas.remove(touch.ud['line'])
            self.redraw()
        else:
            return super(ParamCurve, self).on_touch_up(touch)

"""class ParamBars(ScrollView):
    xScale = NumericProperty(1)
    seqLength = NumericProperty(1000)
    points = ListProperty()
    rectangles = ListProperty([])
    def redraw(self):
        data = []
        self.points = []
        for i in self.rectangles:
            self.children[0].canvas.remove(i)
        c = 0
        for i in data:
            self.points.append((self.parent.xScale * i[0], i[1] * self.height))
            c += 1
        with self.children[0].canvas:
            Color(1, 0, 0, 1)
            for i in self.points:
                self.rectangles.append(Rectangle(pos = (i[0], self.y), size = (10, i[1])))"""

class TimingOptns(ScrollView):
    xScale = NumericProperty(1)
    seqLength = NumericProperty(1000)
    points1 = ListProperty()
    points2 = ListProperty()
    rectangles1 = ListProperty()
    rectangles2 = ListProperty()
    def redraw(self):
        for i in self.rectangles1:
            self.children[0].canvas.remove(i)
        del self.rectangles1[:]
        for i in self.rectangles2:
            self.children[0].canvas.remove(i)
        del self.rectangles2[:]
        data1 = middleLayer.trackList[middleLayer.activeTrack].loopOverlap
        data2 = middleLayer.trackList[middleLayer.activeTrack].loopOffset
        self.points1 = []
        self.points2 = []
        for i in range(data1.size()[0]):
            self.points1.append((self.parent.xScale * middleLayer.trackList[middleLayer.activeTrack].borders[3 * i + 1], data1[i].item() * self.height / 2))
            self.points2.append((self.parent.xScale * middleLayer.trackList[middleLayer.activeTrack].borders[3 * i + 1],  (data2[i].item() * self.height) / 2))
        with self.children[0].canvas:
            Color(1, 0, 0, 1)
            for i in self.points1:
                self.rectangles1.append(ObjectProperty())
                self.rectangles1[-1] = Rectangle(pos = (i[0], self.y + 0.5 * self.height), size = (10, i[1]))
            for i in self.points2:
                self.rectangles2.append(ObjectProperty())
                self.rectangles2[-1] = Rectangle(pos = (i[0], self.y), size = (10, i[1]))
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if touch.is_mouse_scrolling:
                if touch.button == 'scrollup':
                    newvalue = self.scroll_x + self.convert_distance_to_scroll(self.scroll_wheel_distance, 0)[0]
                    if newvalue < 1:
                        self.scroll_x = newvalue
                    else:
                        self.scroll_x = 1.
                elif touch.button == 'scrolldown':
                    newvalue = self.scroll_x - self.convert_distance_to_scroll(self.scroll_wheel_distance, 0)[0]
                    if newvalue > 0:
                        self.scroll_x = newvalue
                    else:
                        self.scroll_x = 0.
            else:
                with self.children[0].canvas:
                    touch.ud['startPoint'] = self.to_local(touch.x, touch.y)
                    touch.ud['startPoint'] = [int(touch.ud['startPoint'][0] / self.xScale), min(max(touch.ud['startPoint'][1], 0.), self.height)]
                    touch.ud['section'] = touch.ud['startPoint'][1] > self.height / 2
                    if middleLayer.tool != "draw":
                        Color(0, 0, 1)
                        touch.ud['line'] = Line(points=[touch.x, touch.y])
                        touch.ud['startPointOffset'] = 0
                        touch.ud['lastPoint'] = touch.ud['startPoint'][0]
            return True
        else:
            return False
    def on_touch_move(self, touch):
        def getPreviousBar(self, x):
            if touch.ud['section']:
                for i in range(len(self.points1)):
                    j = len(self.points1) - i - 1
                    if x > self.points1[j][0]:
                        return j
            else:
                for i in range(len(self.points2)):
                    j = len(self.points2) - i - 1
                    if x > self.points2[j][0]:
                        return j
            return None
        def barToPos(self, bar, x):
            if bar == None:
                return x
            if touch.ud['section']:
                return self.points1[bar][0]
            else:
                return self.points2[bar][0]
        if 'startPoint' in touch.ud:
            global middleLayer
            coord = self.to_local(touch.x, touch.y)
            x = int(coord[0] / self.xScale)
            if touch.ud['section']:
                y = min(max(coord[1], self.height / 2), self.height)
            else:
                y = min(max(coord[1], 0.), self.height / 2)
            p = x - touch.ud['startPoint'][0]
            if middleLayer.tool == "draw":
                bar = getPreviousBar(self, x)
                if bar == None:
                    return True
                if touch.ud["section"]:
                    middleLayer.applyParamChanges([y * 2 / self.height - 1], bar, section = touch.ud['section'])
                    self.points1[bar] = (self.points1[bar][0], y)
                    self.children[0].canvas.remove(self.rectangles1[bar])
                    del self.rectangles1[bar]
                    with self.children[0].canvas:
                        Color(1, 0, 0, 1)
                        self.rectangles1.insert(bar, ObjectProperty())
                        self.rectangles1[bar] = Rectangle(pos = (barToPos(self, bar, x), self.y + 0.5 * self.height), size = (10, self.points1[bar][1] - 0.5 * self.height))
                else:
                    middleLayer.applyParamChanges([y * 2 / self.height], bar, section = touch.ud['section'])
                    self.points2[bar] = (self.points2[bar][0], y)
                    self.children[0].canvas.remove(self.rectangles2[bar])
                    del self.rectangles2[bar]
                    with self.children[0].canvas:
                        Color(1, 0, 0, 1)
                        self.rectangles2.insert(bar, ObjectProperty)
                        self.rectangles2[bar] = Rectangle(pos = (barToPos(self, bar, x), self.y), size = (10, self.points2[bar][1]))
            elif middleLayer.tool == "line":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * -i / p]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * i / p]
            elif middleLayer.tool == "arch":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * (-i / p) * (-i / p)]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * (i / p) * (i / p)]
            elif middleLayer.tool == "reset":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, self.height / 2]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, self.height / 2]
        else:
            return super(TimingOptns, self).on_touch_move(touch)
    def on_touch_up(self, touch):
        def getCurrentBar(self, x):
            if touch.ud['section']:
                for i in range(len(self.points1)):
                    j = len(self.points1) - i - 1
                    if x == self.points1[j][0]:
                        return j
                    if x > self.points1[j][0]:
                        break
            else:
                for i in range(len(self.points2)):
                    j = len(self.points2) - i - 1
                    if x == self.points2[j][0]:
                        return j
                    if x > self.points2[j][0]:
                        break
            return None
        global middleLayer
        if middleLayer.tool == "draw":
            pass #submit data to backend
        elif 'startPoint' in touch.ud:
            data = []
            if touch.ud['section']:
                if touch.ud['startPointOffset'] == 0:
                    for i in range(int(len(touch.ud['line'].points) / 2)):
                        data.append((touch.ud['line'].points[2 * i + 1] * 2 / self.height) - 1)
                else:
                    for i in range(int(len(touch.ud['line'].points) / 2)):
                        data.append((touch.ud['line'].points[2 * (int(len(touch.ud['line'].points) / 2) - i) - 1] * 2 / self.height) - 1)
            else:
                if touch.ud['startPointOffset'] == 0:
                    for i in range(int(len(touch.ud['line'].points) / 2)):
                        data.append(touch.ud['line'].points[2 * i + 1] * 2 / self.height)
                else:
                    for i in range(int(len(touch.ud['line'].points) / 2)):
                        data.append(touch.ud['line'].points[2 * (int(len(touch.ud['line'].points) / 2) - i) - 1] * 2 / self.height)
            finalData = []
            firstBar = None
            for i in range(len(data)):
                bar = getCurrentBar(self, touch.ud["startPoint"][0] - touch.ud["startPointOffset"] + i)
                if bar != None:
                    finalData.append(data[i])
                    if firstBar == None:
                        firstBar = bar
            if firstBar != None:
                middleLayer.applyParamChanges(finalData, firstBar, section = touch.ud['section'])
            self.children[0].canvas.remove(touch.ud['line'])
            self.redraw()
        else:
            return super(TimingOptns, self).on_touch_up(touch)
class PitchOptns(ScrollView):
    xScale = NumericProperty(1)
    seqLength = NumericProperty(1000)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.line1 = ObjectProperty()
        self.line2 = ObjectProperty()
        with self.children[0].canvas:
            self.line1 = Line()
            self.line2 = Line()
    def redraw(self):
        self.children[0].canvas.remove(self.line1)
        self.children[0].canvas.remove(self.line2)
        data1 = middleLayer.trackList[middleLayer.activeTrack].vibratoStrength
        data2 = middleLayer.trackList[middleLayer.activeTrack].vibratoSpeed
        points1 = []
        points2 = []
        c = 0
        for i in data1:
            points1.append((self.parent.xScale * c, int((1 + i) * self.height / 4)))
            c += 1
        c = 0
        for i in data2:
            points2.append((self.parent.xScale * c, int((3 + i) * self.height / 4)))
            c += 1
        with self.children[0].canvas:
            Color(1, 0, 0, 1)
            self.line1 = Line(points = points1)
            self.line2 = Line(points = points2)
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if touch.is_mouse_scrolling:
                if touch.button == 'scrollup':
                    newvalue = self.scroll_x + self.convert_distance_to_scroll(self.scroll_wheel_distance, 0)[0]
                    if newvalue < 1:
                        self.scroll_x = newvalue
                    else:
                        self.scroll_x = 1.
                elif touch.button == 'scrolldown':
                    newvalue = self.scroll_x - self.convert_distance_to_scroll(self.scroll_wheel_distance, 0)[0]
                    if newvalue > 0:
                        self.scroll_x = newvalue
                    else:
                        self.scroll_x = 0.
            else:
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points=[touch.x, touch.y])
                    touch.ud['startPoint'] = self.to_local(touch.x, touch.y)
                    touch.ud['startPoint'] = [int(touch.ud['startPoint'][0] / self.xScale), min(max(touch.ud['startPoint'][1], 0.), self.height)]
                    touch.ud['startPointOffset'] = 0
                    touch.ud['section'] = touch.ud['startPoint'][1] > self.height / 2
                    touch.ud['lastPoint'] = touch.ud['startPoint'][0]
                    touch.ud['param'] = True
            return True
        else:
            return False#super(PitchOptns, self).on_touch_down(touch)
    def on_touch_move(self, touch):
        if 'startPoint' in touch.ud and touch.ud['param']:
            global middleLayer
            coord = self.to_local(touch.x, touch.y)
            x = int(coord[0] / self.xScale)
            if touch.ud['section']:
                y = min(max(coord[1], self.height / 2), self.height)
            else:
                y = min(max(coord[1], 0.), self.height / 2)
            p = x - touch.ud['startPoint'][0]
            if middleLayer.tool == "draw":
                if p < 0:
                    for i in range(-p):
                        touch.ud['line'].points = [touch.ud['startPoint'][0] - i * self.xScale, y] + touch.ud['line'].points
                        touch.ud['startPoint'][0] -= 1
                        touch.ud['lastPoint'] = touch.ud['line'].points[0] / self.xScale
                elif p < int(len(touch.ud['line'].points) / 2):
                    points = touch.ud['line'].points
                    if x >= touch.ud['lastPoint']:
                        domain = range(int(touch.ud['lastPoint']) - int(touch.ud['startPoint'][0]), p)
                    else:
                        domain = range(p, int(touch.ud['lastPoint']) - int(touch.ud['startPoint'][0]))
                    for i in domain:
                        points[2 * i + 1] = y
                    touch.ud['line'].points = points
                    touch.ud['lastPoint'] = points[2 * p] / self.xScale
                else:
                    diff = p - int(len(touch.ud['line'].points) / 2)
                    for i in range(diff):
                        touch.ud['line'].points += [(touch.ud['startPoint'][0] + int(len(touch.ud['line'].points) / 2)) * self.xScale, y]
                    touch.ud['lastPoint'] = touch.ud['line'].points[len(touch.ud['line'].points) - 2] / self.xScale
            elif middleLayer.tool == "line":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * -i / p]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * i / p]
            elif middleLayer.tool == "arch":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * (-i / p) * (-i / p)]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (y - touch.ud['startPoint'][1]) * (i / p) * (i / p)]
            elif middleLayer.tool == "reset":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, self.height / 2]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, self.height / 2]
        else:
            return super(PitchOptns, self).on_touch_move(touch)
    def on_touch_up(self, touch):
        global middleLayer
        if 'startPoint' in touch.ud and touch.ud['param']:
            data = []
            if touch.ud['section']:
                if touch.ud['startPointOffset'] == 0:
                    for i in range(int(len(touch.ud['line'].points) / 2)):
                        data.append((touch.ud['line'].points[2 * i + 1] * 4 / self.height) - 3)
                else:
                    for i in range(int(len(touch.ud['line'].points) / 2)):
                        data.append((touch.ud['line'].points[2 * (int(len(touch.ud['line'].points) / 2) - i) - 1] * 4 / self.height) - 3)
            else:
                if touch.ud['startPointOffset'] == 0:
                    for i in range(int(len(touch.ud['line'].points) / 2)):
                        data.append((touch.ud['line'].points[2 * i + 1] * 4 / self.height) - 1)
                else:
                    for i in range(int(len(touch.ud['line'].points) / 2)):
                        data.append((touch.ud['line'].points[2 * (int(len(touch.ud['line'].points) / 2) - i) - 1] * 4 / self.height) - 1)
            middleLayer.applyParamChanges(data, touch.ud['startPoint'][0] - touch.ud['startPointOffset'], section = touch.ud['section'])
            self.children[0].canvas.remove(touch.ud['line'])
            self.redraw()
        else:
            return super(PitchOptns, self).on_touch_up(touch)

class NoteProperties(Bubble):
    reference = ObjectProperty()
    def on_parent(self, instance, value):
        for i in self.content.children:
            i.reference = self.reference
        return super().on_parent(instance, value)

class ReferencingButton(BubbleButton):
    reference = ObjectProperty()

class Note(ToggleButton):
    index = NumericProperty()
    xPos = NumericProperty()
    yPos = NumericProperty()
    length = NumericProperty()
    inputMode = BooleanProperty()
    def on_parent(self, screen, parent):
        if parent == None:
            return
        self.redraw()
    def redraw(self):
        self.pos = (self.xPos * self.parent.parent.xScale, self.yPos * self.parent.parent.yScale)
        self.width = self.length * self.parent.parent.xScale
        self.height = self.parent.parent.yScale
    def on_touch_down(self, touch):
        if middleLayer.mode != "notes":
            return False
        if self.collide_point(*touch.pos):
            coord = self.to_local(touch.x, touch.y)
            touch.ud["initialPos"] = coord
            touch.ud["noteIndex"] = self.index
            if coord[0] <= self.x + self.width and coord[0] > max(self.x, self.x + self.width - 10):
                touch.ud["grabMode"] = "end"
            elif coord[0] >= self.x and coord[0] < min(self.x + 10, self.x + self.width):
                touch.ud["grabMode"] = "start"
            else:
                touch.ud["grabMode"] = "mid"
                touch.ud["xOffset"] = (self.pos[0] - coord[0]) / self.parent.parent.xScale
                touch.ud["yOffset"] = (self.pos[1] - coord[1]) / self.parent.parent.yScale
            touch.ud['param'] = False
            return True
        return super().on_touch_down(touch)
    def on_touch_up(self, touch):
        if middleLayer.mode != "notes" or touch.is_mouse_scrolling or "initialPos" not in touch.ud.keys():
            return False
        coord = self.to_local(touch.x, touch.y)
        if (abs(touch.ud["initialPos"][0] - coord[0]) < 4) and (abs(touch.ud["initialPos"][1] - coord[1]) < 4) and self.collide_point(*coord):
            if self.state == "down":
                super().on_touch_down(touch)
                super().on_touch_up(touch)
            else:
                self.trigger_action()
            return True
        return False
    def on_state(self, screen, state):
        if state == "normal":
            self.remove_widget(self.children[0])
        else:
            self.add_widget(NoteProperties(reference = self))
    def changeInputMode(self):
        self.inputMode = not self.inputMode
    def delete(self):
        middleLayer.removeNote(self.index)
        for i in self.parent.children:
            if i.__class__.__name__ == "Note":
                if i.index > self.index:
                    i.index -= 1
        self.parent.remove_widget(self)
    def changeLyrics(self, text):
        middleLayer.changeLyrics(self.index, text, self.inputMode)

class PianoRollOctave(FloatLayout):
    pass

class PianoRoll(ScrollView):
    def __init__(self, **kwargs):
        super(PianoRoll, self).__init__(**kwargs)
        self.xScale = NumericProperty()
        self.yScale = NumericProperty()
        self.currentNote = ObjectProperty()
        self.timingMarkers = ListProperty()
        self.pitchLine = ObjectProperty()
        self.timingMarkers = []
        self.pitchLine = None
    def generate_notes(self):
        for d in self.data:
            self.children[0].add_widget(Note(**d))
    def redrawPitch(self):
        data = middleLayer.trackList[middleLayer.activeTrack].pitch
        points = []
        c = 0
        for i in data:
            points.append(c * self.xScale)
            points.append(i.item() * self.yScale)
            c += 1
        if self.pitchLine != None:
            self.children[0].canvas.remove(self.pitchLine)
        with self.children[0].canvas:
            Color(1, 0, 0, 1)
            self.pitchLine = Line(points = points)
    def changeMode(self):
        if middleLayer.mode == "notes":
            for i in self.timingMarkers:
                self.children[0].canvas.remove(i)
            del self.timingMarkers[:]
            if self.pitchLine != None:
                self.children[0].canvas.remove(self.pitchLine)
                self.pitchLine = None
        if middleLayer.mode == "timing":
            for i in self.children[0].children:
                i.state = "normal"
            if self.pitchLine != None:
                self.children[0].canvas.remove(self.pitchLine)
                self.pitchLine = None
            with self.children[0].canvas:
                Color(1, 0, 0)
                for i in middleLayer.trackList[middleLayer.activeTrack].borders:
                    self.timingMarkers.append(ObjectProperty())
                    self.timingMarkers[-1] = Line(points = [self.xScale * i, 0, self.xScale * i, self.children[0].height])
        if middleLayer.mode == "pitch":
            for i in self.children[0].children:
                i.state = "normal"
            for i in self.timingMarkers:
                self.children[0].canvas.remove(i)
            del self.timingMarkers[:]
            self.redrawPitch()
    def applyScroll(self, scrollValue):
        self.scroll_x = scrollValue
    def triggerScroll(self):
        global middleLayer
        middleLayer.scrollValue = self.scroll_x
        middleLayer.applyScroll()
    def on_touch_down(self, touch):
        global middleLayer
        if touch.is_mouse_scrolling == False:
            if super(PianoRoll, self).on_touch_down(touch):
                return True
        if self.collide_point(*touch.pos):
            if touch.is_mouse_scrolling:
                if middleLayer.shift:
                    if touch.button == 'scrollup':
                        newvalue = self.scroll_x + self.convert_distance_to_scroll(self.scroll_wheel_distance, 0)[0]
                        if newvalue < 1:
                            self.scroll_x = newvalue
                        else:
                            self.scroll_x = 1.
                    elif touch.button == 'scrolldown':
                        newvalue = self.scroll_x - self.convert_distance_to_scroll(self.scroll_wheel_distance, 0)[0]
                        if newvalue > 0:
                            self.scroll_x = newvalue
                        else:
                            self.scroll_x = 0.
                    return True
                return super(PianoRoll, self).on_touch_down(touch)
            else:
                coord = self.to_local(touch.x, touch.y)
                x = int(coord[0] / self.xScale)
                y = int(coord[1] / self.yScale)
                if middleLayer.mode == "notes":
                    index = 0
                    for i in self.children[0].children:
                        if i.__class__.__name__ == "Note":
                            if i.xPos < x:
                                index += 1
                            elif i.xPos > x:
                                i.index += 1
                            else:
                                index += 1
                                x += 1
                    newNote = Note(index = index, xPos = x, yPos = y, length = 100, height = self.yScale)
                    middleLayer.addNote(index, x, y, newNote)
                    self.children[0].add_widget(newNote)
                    touch.ud["noteIndex"] = index
                    touch.ud["grabMode"] = "end"
                    touch.ud["initialPos"] = coord
                elif middleLayer.mode == "timing":
                    def getNearestBorder(x):
                        nextBorder = 0
                        for i in range(len(middleLayer.trackList[middleLayer.activeTrack].borders) - 1):
                            previousBorder = nextBorder
                            nextBorder = (middleLayer.trackList[middleLayer.activeTrack].borders[i] + middleLayer.trackList[middleLayer.activeTrack].borders[i + 1]) / 2
                            if previousBorder < x and x <= nextBorder:
                                return i
                        return len(middleLayer.trackList[middleLayer.activeTrack].borders) - 1
                    border = getNearestBorder(x)
                    touch.ud["border"] = border
                    touch.ud["offset"] = x - middleLayer.trackList[middleLayer.activeTrack].borders[border]
                elif middleLayer.mode == "pitch":
                    with self.children[0].canvas:
                        Color(0, 0, 1)
                        touch.ud['line'] = Line(points=[touch.x, touch.y])
                        touch.ud['startPoint'] = self.to_local(touch.x, touch.y)
                        touch.ud['startPoint'] = [int(touch.ud['startPoint'][0] / self.xScale), min(max(touch.ud['startPoint'][1], 0.), self.height)]
                        touch.ud['startPointOffset'] = 0
                touch.ud['param'] = False
                return True
        return False
    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos) == False or touch.ud['param']:
            return False
        coord = self.to_local(touch.x, touch.y)
        x = int(coord[0] / self.xScale)
        y = int(coord[1] / self.yScale)
        yMod = coord[1]
        if middleLayer.mode == "notes":
            if "noteIndex" in touch.ud:
                note = middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"]].reference
                if abs(touch.ud["initialPos"][0] - coord[0]) < 4 and abs(touch.ud["initialPos"][1] - coord[1]) < 4:
                    return True
                if touch.ud["grabMode"] == "start":
                    length = max(note.xPos + note.length - x, 1)
                    note.length = length
                    note.xPos = x
                    switch = middleLayer.changeNoteLength(touch.ud["noteIndex"], x, length)
                    if switch == True:
                        middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"] + 1].reference.index += 1
                        middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"]].reference.index -= 1
                        touch.ud["noteIndex"] += 1
                    elif switch == False:
                        middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"] - 1].reference.index -= 1
                        middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"]].reference.index += 1
                        note.index -= 1
                        touch.ud["noteIndex"] -= 1
                    note.redraw()
                elif touch.ud["grabMode"] == "mid":
                    note.xPos = int(x + touch.ud["xOffset"] + 1)
                    note.yPos = int(y + touch.ud["yOffset"] + 1)
                    switch = middleLayer.moveNote(touch.ud["noteIndex"], int(x + touch.ud["xOffset"] + 1), int(x + touch.ud["yOffset"] + 1))
                    if switch == True:
                        middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"] + 1].reference.index += 1
                        middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"]].reference.index -= 1
                        touch.ud["noteIndex"] += 1
                    elif switch == False:
                        middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"] - 1].reference.index -= 1
                        middleLayer.trackList[middleLayer.activeTrack].notes[touch.ud["noteIndex"]].reference.index += 1
                        touch.ud["noteIndex"] -= 1
                    note.redraw()
                elif touch.ud["grabMode"] == "end":
                    length = max(x - note.xPos, 1)
                    note.length = length
                    middleLayer.changeNoteLength(touch.ud["noteIndex"], note.xPos, length)
                    note.redraw()
                return True
            else:
                return False
        elif middleLayer.mode == "timing":
            self.children[0].canvas.remove(self.timingMarkers[touch.ud["border"]])
            del self.timingMarkers[touch.ud["border"]]
            self.timingMarkers.insert(touch.ud["border"], ObjectProperty())
            newPos = x - touch.ud["offset"]
            with self.children[0].canvas:
                self.timingMarkers[touch.ud["border"]] = Line(points = [self.xScale * newPos, 0, self.xScale * newPos, self.children[0].height])
            middleLayer.changeBorder(touch.ud["border"], newPos)
        elif middleLayer.mode == "pitch":
            p = x - int(touch.ud['startPoint'][0] / self.xScale)
            if middleLayer.tool == "draw":
                if p < 0:
                    for i in range(-p):
                        touch.ud['line'].points = [touch.ud['startPoint'][0] - i * self.xScale, yMod] + touch.ud['line'].points
                        touch.ud['startPoint'][0] -= 1
                elif p < int(len(touch.ud['line'].points) / 2):
                    points = touch.ud['line'].points
                    points[2 * p] = x * self.xScale
                    points[2 * p + 1] = yMod
                    touch.ud['line'].points = points
                else:
                    diff = p - int(len(touch.ud['line'].points) / 2)
                    for i in range(diff):
                        touch.ud['line'].points += [(touch.ud['startPoint'][0] + int(len(touch.ud['line'].points) / 2)) * self.xScale, yMod]
            elif middleLayer.tool == "line":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (yMod - touch.ud['startPoint'][1]) * -i / p]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (yMod - touch.ud['startPoint'][1]) * i / p]
            elif middleLayer.tool == "arch":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (yMod - touch.ud['startPoint'][1]) * (-i / p) * (-i / p)]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, touch.ud['startPoint'][1] + (yMod - touch.ud['startPoint'][1]) * (i / p) * (i / p)]
            elif middleLayer.tool == "reset":
                self.children[0].canvas.remove(touch.ud['line'])
                with self.children[0].canvas:
                    Color(0, 0, 1)
                    touch.ud['line'] = Line(points = [])
                if p < 0:
                    touch.ud['startPointOffset'] = -p
                    for i in range(-p):
                        touch.ud['line'].points += [(-i + touch.ud['startPoint'][0]) * self.xScale, self.height / 2]
                if p >= 0:
                    for i in range(p):
                        touch.ud['line'].points += [(i + touch.ud['startPoint'][0]) * self.xScale, self.height / 2]
        else:
            return super().on_touch_move(touch)
    def on_touch_up(self, touch):
        global middleLayer
        if 'startPoint' in touch.ud and touch.ud['param'] == False:
            data = []
            if touch.ud['startPointOffset'] == 0:
                for i in range(int(len(touch.ud['line'].points) / 2)):
                    data.append(touch.ud['line'].points[2 * i + 1] / self.yScale)
            else:
                for i in range(int(len(touch.ud['line'].points) / 2)):
                    data.append(touch.ud['line'].points[2 * (int(len(touch.ud['line'].points) / 2) - i) - 1] / self.yScale)
            middleLayer.applyPitchChanges(data, touch.ud['startPoint'][0] - touch.ud['startPointOffset'])
            self.children[0].canvas.remove(touch.ud['line'])
            self.redrawPitch()
        else:
            return super(PianoRoll, self).on_touch_up(touch)

class ListElement(Button):
    index = NumericProperty()

class FileSidePanel(ModalView):
    pass

class SingerSidePanel(ModalView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.voicebanks = []
        self.filepaths = []
        self.selectedIndex = None
    def listVoicebanks(self):
        files = os.listdir("Voices/")
        for file in files:
            if file.endswith(".nvvb"):
                data = torch.load(os.path.join("Voices/", file))
                self.voicebanks.append(data["metadata"])
                self.filepaths.append(os.path.join("Voices/", file))
        j = 0
        for i in self.voicebanks:
            self.ids["singers_list"].add_widget(ListElement(text = i.name, index = j))
            j += 1
    def detailElement(self, index):
        self.ids["singer_name"].text = self.voicebanks[index].name
        #self.ids["singer_image"].source = self.voicebanks[index].image
        canvas_img = self.voicebanks[index].image
        data = BytesIO()
        canvas_img.save(data, format='png')
        data.seek(0)
        im = CoreImage(BytesIO(data.read()), ext='png')
        self.ids["singer_image"].texture = im.texture
        self.ids["singer_version"].text = self.voicebanks[index].version
        self.ids["singer_description"].text = self.voicebanks[index].description
        self.ids["singer_license"].text = self.voicebanks[index].license
        self.selectedIndex = index
    def importVoicebank(self, path, name, image):
        global middleLayer
        middleLayer.importVoicebank(path, name, image)

class ParamSidePanel(ModalView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameters = []
        self.filepaths = []
        self.selectedIndex = None
    def listParams(self):
        files = os.listdir("Params/")
        for file in files:
            if file.endswith(".nvpr"):
                data = torch.load(os.path.join("Params/", file))
                self.parameters.append(data["metadata"])
                self.filepaths.append(os.path.join("Params/", file))
        j = 0
        for i in self.parameters:
            self.ids["params_list"].add_widget(ListElement(text = i.name, index = j))
            j += 1
    def detailElement(self, index):
        self.ids["param_name"].text = self.voicebanks[index].name
        self.ids["param_type"].text = self.voicebanks[index]._type
        self.ids["param_capacity"].text = self.voicebanks[index].capacity
        self.ids["param_recurrency"].text = self.voicebanks[index].recurrency
        self.ids["param_version"].text = self.voicebanks[index].version
        self.ids["param_license"].text = self.voicebanks[index].license
        self.selectedIndex = index
    def importVoicebank(self, path, name):
        global middleLayer
        middleLayer.importParam(path, name)

class ScriptingSidePanel(ModalView):
    def openDevkit(self):
        subprocess.Popen("Devkit.exe")
    def runScript(self):
        exec(self.ids["scripting_editor"].text)

class SettingsSidePanel(ModalView):
    def readSettings(self):
        settings = {}
        with open("settings.ini", 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split(" ")
                settings[line[0]] = line[1]
        self.ids["settings_lang"].text = settings["language"]
        self.ids["settings_accel"].text = settings["accelerator"]
        self.ids["settings_tcores"].text = settings["tensorCores"]
        self.ids["settings_prerender"].text = settings["intermediateOutputs"]
        self.ids["settings_loglevel"].text = settings["loglevel"]
    def writeSettings(self):
        with open("settings.ini", 'w') as f:
            f.write("language " + self.ids["settings_lang"].text + "\n")
            f.write("accelerator " + self.ids["settings_accel"].text + "\n")
            f.write("tensorCores " + self.ids["settings_tcores"].text + "\n")
            f.write("intermediateOutputs " + self.ids["settings_prerender"].text + "\n")
            f.write("loglevel " + self.ids["settings_loglevel"].text + "\n")

class LicensePanel(Popup):
    pass

class NovaVoxUI(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        global middleLayer
        global manager
        global sequenceList
        global voicebankList
        global aiParamStackList
        middleLayer = MiddleLayer(self.ids)
        sequenceList = []
        voicebankList = []
        aiParamStackList = []
        manager = RenderManager(sequenceList, voicebankList, aiParamStackList)
        self._keyboard = Window.request_keyboard(None, self, 'text')
        if self._keyboard.widget:
            pass
        self._keyboard.bind(on_key_down = self.on_keyboard_down)
        self._keyboard.bind(on_key_up = self.on_keyboard_up)
    def update(self, deltatime):
        change = manager.receiveChange()
        if change.type == False:
            middleLayer.updateRenderStatus(change.track, change.index, change.value)
        elif change.type == True:
            middleLayer.updateAudioBuffer(change.track, change.index, change.value)
    def setMode(self, mode):
        global middleLayer
        middleLayer.mode = mode
        middleLayer.updateParamPanel()
        middleLayer.changePianoRollMode()
    def setTool(self, tool):
        global middleLayer
        middleLayer.tool = tool
    def on_keyboard_down(self, window, keycode, text, modifiers):
        if keycode[0] == 303 or keycode[0] == 304: 
            middleLayer.shift = True
        else:
            pass
        return True
    def on_keyboard_up(self, keyboard, keycode):
        if keycode[0] == 303 or keycode[0] == 304: 
            middleLayer.shift = False
        else:
            pass
        return True