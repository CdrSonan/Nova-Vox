#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from kivy.core.image import Image as CoreImage
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, OptionProperty
from kivy.clock import mainthread

from io import BytesIO
from copy import copy, deepcopy

import torch
import math

import sounddevice

import global_consts

from MiddleLayer.IniParser import readSettings
import MiddleLayer.DataHandlers as dh

from Backend.Util import ensureTensorLength

from UI.code.editor.AdaptiveSpace import ParamCurve, TimingOptns, PitchOptns
from UI.code.editor.Headers import SingerPanel, ParamPanel

class MiddleLayer(Widget):
    """Central class of the program. Contains data handlers for all data on the main process, and callbacks for modifying it, which are triggered from the UI.
    Through its manager attribute, it indirectly also handles communication between the main and rendering process.
    Contains some functions for updating the UI when mode changes occur, but these functions may be moved to a dedicated class in the future."""

    def __init__(self, ids, **kwargs) -> None:
        """Constructor called once during program startup. uses the id list of the main UI for referencing various UI elements and updating them. Functionality related to such UI updates may be moved to a dedicated class in the future, deprecating this argument."""
        
        super().__init__(**kwargs)
        self.trackList = []
        from Backend.NV_Multiprocessing.Manager import RenderManager
        self.manager = RenderManager(self.trackList)
        self.ids = ids
        self.activeTrack = None
        self.activeParam = "steadiness"
        self.mode = OptionProperty("notes", options = ["notes", "timing", "pitch"])
        self.mode = "notes"
        self.tool = OptionProperty("draw", options = ["draw", "line", "arch", "reset"])
        self.tool = "draw"
        self.shift = BooleanProperty()
        self.shift = False
        self.alt = BooleanProperty()
        self.alt = False
        self.ctrl = BooleanProperty()
        self.ctrl = False
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

    def importVoicebank(self, path:str, name:str, inImage) -> None:
        """Creates a new vocal track with a Voicebank loaded from disk.

        Arguments:
            path: filepath of the .nvvb Voicebank file used for the track

            name: display name of the Voicebank/track

            inImage: image displayed in the track header"""


        self.importVoicebankNoSubmit(path, name, inImage)
        self.submitAddTrack(self.trackList[-1])

    def importVoicebankNoSubmit(self, path:str, name:str, inImage) -> None:
        """Creates a new vocal track with a Voicebank loaded from disk, but does not submit it to the rendering process.
        The rendering process needs to be restarted, or submitAddTrack needs to be called separately for the new track to be recognized

        Arguments:
            path: filepath of the .nvvb Voicebank file used for the track

            name: display name of the Voicebank/track

            inImage: image displayed in the track header"""
        

        track = dh.Track(path)
        self.trackList.append(track)
        canvas_img = inImage
        data = BytesIO()
        canvas_img.save(data, format='png')
        data.seek(0)
        im = CoreImage(BytesIO(data.read()), ext='png')
        image = im.texture
        self.ids["singerList"].add_widget(SingerPanel(name = name, image = image, index = len(self.trackList) - 1))
        self.audioBuffer.append(torch.zeros([track.length * global_consts.batchSize,]))

    def importParam(self, path:str, name:str) -> None:
        """placeholder function for importing an Ai-driven parameter. Deprecated with the introduction of node-based processing."""

        self.trackList[self.activeTrack].nodeGraph.append(path)
        self.ids["paramList"].add_widget(ParamPanel(name = name, switchable = True, sortable = True, deletable = True, index = len(self.trackList[self.activeTrack].nodegraph.params) - 1))
        self.submitNodegraphUpdate()

    def changeTrack(self, index) -> None:
        """Helper function triggering the required UI updates when the user selects a different track"""

        self.activeTrack = index
        self.updateParamPanel()
        self.updatePianoRoll()

    def copyTrack(self, index:int, name:str, inImage) -> None:
        """Duplicates a vocal track and all of its associated data.

        Arguments:
            index: the index of the track to duplicate in self.trackList
            
            name: the display name of the duplicated track
            
            inImage: Image displayed in the header of the duplicated track"""


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
        self.trackList[-1].aiBalance = reference.aiBalance.clone()
        self.trackList[-1].loopOverlap = reference.loopOverlap.clone()
        self.trackList[-1].loopOffset = reference.loopOffset.clone()
        self.trackList[-1].vibratoSpeed = reference.vibratoSpeed.clone()
        self.trackList[-1].vibratoStrength = reference.vibratoStrength.clone()
        self.trackList[-1].usePitch = copy(reference.usePitch)
        self.trackList[-1].useBreathiness = copy(reference.useBreathiness)
        self.trackList[-1].useSteadiness = copy(reference.useSteadiness)
        self.trackList[-1].useAIBalance = copy(reference.useAIBalance)
        self.trackList[-1].useVibratoSpeed = copy(reference.useVibratoSpeed)
        self.trackList[-1].useVibratoStrength = copy(reference.useVibratoStrength)
        self.trackList[-1].nodegraph = copy(reference.nodegraph)
        self.trackList[-1].borders = deepcopy(reference.borders)
        self.trackList[-1].length = copy(reference.length)
        self.trackList[-1].mixinVB = copy(reference.mixinVB)
        self.trackList[-1].pauseThreshold = copy(reference.pauseThreshold)
        image = inImage
        self.ids["singerList"].add_widget(SingerPanel(name = name, image = image, index = len(self.trackList) - 1))
        self.audioBuffer.append(deepcopy(self.audioBuffer[index]))
        self.submitDuplicateTrack(index)

    def deleteTrack(self, index:int) -> None:
        """Deletes the track at index index in self.trackList, and all of its associated data"""

        self.trackList.pop(index)
        self.audioBuffer.pop(index)
        if self.activeTrack != None:
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

    def deleteParam(self, index:int) -> None:
        """Placeholder function for removing an Ai-driven parameter from a track's stack. Deprecated with the introduction of node-based processing."""

        self.trackList[self.activeTrack].nodegraph.delete(index)
        if index <= self.activeParam:
            self.changeParam(self.activeParam - 1)
        for i in self.ids["paramList"].children:
            if i.index == index:
                i.parent.remove_widget(i)
            if i.index > index:
                i.index = i.index - 1
        self.submitNodegraphUpdate

    def enableParam(self, name:str = None) -> None:
        """Enables a toggle-able parameter curve of a vocal track.

        Arguments:
            index: index of the parameter curve in the curve stack. -1 for named resampler parameters.
            
            name: when adressing a named resampler parameter, the name of the parameter. Otherwise ignored."""


        if name == "steadiness":
            self.trackList[self.activeTrack].useSteadiness = True
        elif name == "breathiness":
            self.trackList[self.activeTrack].useBreathiness = True
        elif name == "AI balance":
            self.trackList[self.activeTrack].useAIBalance = True
        elif name == "vibrato speed":
            self.trackList[self.activeTrack].useVibratoSpeed = True
        elif name == "vibrato strength":
            self.trackList[self.activeTrack].useVibratoStrength = True
        else:
            self.trackList[self.activeTrack].nodegraph.params[name].enabled = True
        self.submitEnableParam(name)
        
    def disableParam(self, name:str) -> None:
        """Disables a toggle-able parameter curve of a vocal track.

        Arguments:
            index: index of the parameter curve in the curve stack. -1 for named resampler parameters.
            
            name: When adressing a named resampler parameter, the name of the parameter. Otherwise ignored."""


        if name == "steadiness":
            self.trackList[self.activeTrack].useSteadiness = False
        elif name == "breathiness":
            self.trackList[self.activeTrack].useBreathiness = False
        elif name == "AI balance":
            self.trackList[self.activeTrack].useAIBalance = False
        elif name == "vibrato speed":
            self.trackList[self.activeTrack].useVibratoSpeed = False
        elif name == "vibrato strength":
            self.trackList[self.activeTrack].useVibratoStrength = False
        else:
            self.trackList[self.activeTrack].nodegraph.params[name].enabled = False
        self.submitDisableParam(name)

    def moveParam(self, name:str, switchable:bool, sortable:bool, deletable:bool, index:int, delta:int, switchState:bool = True) -> None:
        """Moves a sortable parameter curve at index index of the current track's param curve stack to a different position defined by delta.
        All other arguments specify information about the parameter being moved for re-applying its header widget."""

        for i in self.ids["paramList"].children:
            if i.index == index:
                i.parent.remove_widget(i)
                break
        self.ids["paramList"].add_widget(ParamPanel(name = name, switchable = switchable, sortable = sortable, deletable = deletable, index = index), index = index + delta, switchState = switchState)
        self.changeParam(index + delta)

    def updateParamPanel(self) -> None:
        """updates the adaptive space and parameter panel after a track or mode change"""

        self.ids["paramList"].clear_widgets()
        self.ids["adaptiveSpace"].clear_widgets()
        if self.mode == "notes":
            self.ids["paramList"].add_widget(ParamPanel(name = "steadiness", switchable = True, sortable = False, deletable = False, index = -1, switchState = self.trackList[self.activeTrack].useSteadiness, state = "down"))
            self.ids["paramList"].add_widget(ParamPanel(name = "breathiness", switchable = True, sortable = False, deletable = False, index = -1, switchState = self.trackList[self.activeTrack].useBreathiness))
            self.ids["paramList"].add_widget(ParamPanel(name = "AI balance", switchable = True, sortable = False, deletable = False, index = -1, switchState = self.trackList[self.activeTrack].useAIBalance))
            self.ids["adaptiveSpace"].add_widget(ParamCurve())
            counter = 0
            for i in self.trackList[self.activeTrack].nodegraph.params:
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

    def updatePianoRoll(self) -> None:
        """updates the piano roll UI after a track or mode change"""

        self.ids["pianoRoll"].updateTrack()

    def changeParam(self, index:int, name:str) -> None:
        """updates the adaptive space after an active parameter change"""

        if index == -1:
            if name == "steadiness":
                self.activeParam = "steadiness"
            elif name == "breathiness":
                self.activeParam = "breathiness"
            elif name == "AI balance":
                self.activeParam = "AI balance"
            elif name == "loop overlap" or name == "loop offset":
                self.activeParam = "loop"
            elif name == "vibrato speed" or name == "vibrato strength":
                self.activeParam = "vibrato"
        else:
            self.activeParam = index
        self.ids["adaptiveSpace"].children[0].redraw()

    def applyParamChanges(self, data:list, start:int, section:bool = False) -> None:
        """submits edits made to a parameter curve to the rendering process. For multicurve parameter views, section represents whether the upper or lower curve was edited."""

        if self.activeParam == "steadiness":
            self.trackList[self.activeTrack].steadiness[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
            self.submitNamedParamChange(True, "steadiness", start, torch.tensor(data, dtype = torch.half))
        elif self.activeParam == "breathiness":
            self.trackList[self.activeTrack].breathiness[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
            self.submitNamedParamChange(True, "breathiness", start, torch.tensor(data, dtype = torch.half))
        elif self.activeParam == "AI balance":
            self.trackList[self.activeTrack].aiBalance[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
            self.submitNamedParamChange(True, "aiBalance", start, torch.tensor(data, dtype = torch.half))
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
            self.trackList[self.activeTrack].nodegraph.params[self.activeParam].curve[start:start + len(data)] = torch.tensor(data, dtype = torch.half)
            self.submitParamChange(True, self.activeParam, start, torch.tensor(data, dtype = torch.half))

    def applyPitchChanges(self, data:list, start:int) -> None:
        """submits edits made to the pitch curve to the rendering process"""

        self.trackList[self.activeTrack].pitch[start:start + len(data)] = torch.tensor(data, dtype = torch.float32)
        data = self.noteToPitch(torch.tensor(data, dtype = torch.float32))
        self.submitNamedPhonParamChange(True, "pitch", start, torch.tensor(data, dtype = torch.half))

    def changePianoRollMode(self) -> None:
        """helper function for piano roll UI updates when changing modes"""

        self.ids["pianoRoll"].changeMode()

    def applyScroll(self) -> None:
        """helper function for synchromizing scrolling between the piano roll and adaptive space"""

        self.ids["pianoRoll"].applyScroll(self.scrollValue)
        self.ids["adaptiveSpace"].applyScroll(self.scrollValue)

    def offsetPhonemes(self, index:int, offset:int, pause:bool = False, futurePhonemes:list = None) -> None:
        """adds or deletes phonemes from the active track, and recalculates timing markers to fit the new sequence.

        Arguments:
            index: the index to add phonemes at, or remove them from

            offset: if positive, number of phonemes added. If negative, number of phonemes removed.
            
            pause: Flag indicating whether an _autopause control phoneme should be inserted
            
            futurePhonemes: when using positive offset, a "preview" of the phonemes that are to be inserted. Used for timing marker calculations.
            
        When using a positive offset, this function adds the placeholder phoneme _X (or several of them). These should be overwritten with normal
        phonemes before submitting the change to the rendering process with the final flag set."""

        #TODO: refactor timing calculations to dedicated function and add callback in switchNote
        phonIndex = self.trackList[self.activeTrack].notes[index].phonemeStart
        #update phoneme list and other phoneme-space lists
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
        #_autopause handling and iterator setup for timing marker calculations
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
        #update timing markers
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

    def makeAutoPauses(self, index:int) -> None:
        """helper function for calculating the _autopause phonemes required by the note at position index of the active track"""

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

    def recalculatePauses(self, index:int) -> None:
        """recalculates all _autopause phonemes for the track at position index of the track list. Used during startup and repair operations."""

        for i in range(len(self.trackList[index].notes)):
            self.makeAutoPauses(i)
    
    def switchNote(self, index:int) -> None:
        """Switches the places of the notes at positions index and index + 1 or the active track. Does currently not clean up timing markers afterwards, so doing so manually or by prompting a call of offsetPhonemes is currently required."""

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

    def scaleNote(self, index:int, oldLength:int) -> None:
        """Changes the length of the note at position index of the active track. THe new length is read from its UI representation, the old length must be given as an argument.
        It does not perform any checks of surrounding notes or other conditions. Therefore, it is recommended to call changeNoteLength instead whenever possible."""

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
    
    def adjustNote(self, index:int, oldLength:int, oldPos:int) -> None:
        """Adjusts a note's attributes after it has been moved or scaled with respect to its surrounding notes. The current position and length are read from its UI representation, the old one must be given as arguments."""

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

    def addNote(self, index:int, x:int, y:int, reference) -> None:
        """adds a new note at position (x, y) and index index to the active track. reference is an ObjectProperty pointing to its UI representation."""

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

    def removeNote(self, index:int) -> None:
        """removes the note at position index of the active track"""

        self.offsetPhonemes(index, self.trackList[self.activeTrack].notes[index].phonemeStart - self.trackList[self.activeTrack].notes[index].phonemeEnd)
        self.trackList[self.activeTrack].notes.pop(index)
        if index < len(self.trackList[self.activeTrack].notes):
            self.adjustNote(index, self.trackList[self.activeTrack].notes[index].length, self.trackList[self.activeTrack].notes[index].xPos)

    def changeNoteLength(self, index:int, x:int, length:int) -> None:
        """changes the length and start position of the note at position index of the active track to length and x. This makes this function useful for moving either the beginning or the end of a note."""

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

    def moveNote(self, index:int, x:int, y:int) -> None:
        """moves the note at position index of the active track to the position (x, y)."""

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

    def changeLyrics(self, index:int, text:str) -> None:
        """changes the lyrics of the note at position index of the active track to text. Performs dictionary lookup and phoneme sanitization, respecting the note's phoneme input mode."""

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
        offsets = torch.tensor([], dtype = torch.half)
        for i in phonemes:
            if self.trackList[self.activeTrack].phonemeLengths[i] == None:
                offsets = torch.cat((offsets, torch.tensor([0.5,], dtype = torch.half)), 0)
            else:
                offsets = torch.cat((offsets, torch.tensor([0.05,], dtype = torch.half)), 0)
        self.trackList[self.activeTrack].loopOffset[self.trackList[self.activeTrack].notes[index].phonemeStart:self.trackList[self.activeTrack].notes[index].phonemeEnd] = offsets
        self.submitNamedPhonParamChange(False, "offsets", self.trackList[self.activeTrack].notes[index].phonemeStart, offsets)
        self.makeAutoPauses(index)
        self.submitFinalize()

    def changeBorder(self, border:int, pos:int) -> None:
        """sets the position of the timing marker at index border of the active track to pos"""

        self.trackList[self.activeTrack].borders[border] = pos
        self.repairBorders(border)
        self.submitBorderChange(True, border, [pos,])

    def repairNotes(self, index:int) -> None:
        """checks if the note at position index of the active track has exactly the same position as the previous note. If so, it is moved forward by one tick, ensuring that no note gets assigned a length of 0."""
        if index == 0 or index == len(self.trackList[self.activeTrack].notes):
            return
        if self.trackList[self.activeTrack].notes[index].xPos == self.trackList[self.activeTrack].notes[index - 1].xPos:
            self.trackList[self.activeTrack].notes[index].xPos += 1
            self.repairNotes(index + 1)

    def repairBorders(self, index:int) -> None:
        """checks if the note at position index of the active track has exactly the same position as the previous note. If so, it is moved forward by one tick, ensuring that no note gets assigned a length of 0."""

        if index == 0 or index == len(self.trackList[self.activeTrack].borders):
            return
        if self.trackList[self.activeTrack].borders[index] < self.trackList[self.activeTrack].borders[index - 1] + 1.:
            self.trackList[self.activeTrack].borders[index] = self.trackList[self.activeTrack].borders[index - 1] + 1.
            self.submitNamedPhonParamChange(False, "borders", index, [self.trackList[self.activeTrack].borders[index],])
            self.repairBorders(index + 1)

    def changeLength(self, length:int) -> None:
        """changes the length of the active track, and submits the change to the renderer"""

        self.trackList[self.activeTrack].length = length
        self.trackList[self.activeTrack].pitch = ensureTensorLength(self.trackList[self.activeTrack].pitch, length, -1)
        self.trackList[self.activeTrack].basePitch = ensureTensorLength(self.trackList[self.activeTrack].basePitch, length, -1)
        self.trackList[self.activeTrack].breathiness = ensureTensorLength(self.trackList[self.activeTrack].breathiness, length, 0)
        self.trackList[self.activeTrack].steadiness = ensureTensorLength(self.trackList[self.activeTrack].steadiness, length, 0)
        self.trackList[self.activeTrack].aiBalance = ensureTensorLength(self.trackList[self.activeTrack].aiBalance, length, 0)
        self.trackList[self.activeTrack].vibratoSpeed = ensureTensorLength(self.trackList[self.activeTrack].vibratoSpeed, length, 0)
        self.trackList[self.activeTrack].vibratoStrength = ensureTensorLength(self.trackList[self.activeTrack].vibratoStrength, length, 0)
        self.audioBuffer[self.activeTrack] = ensureTensorLength(self.audioBuffer[self.activeTrack], length * global_consts.batchSize, 0)
        self.submitChangeLength(True, length)

    def validate(self) -> None:
        """validates the data held by the middle layer, and fixes any errors encountered"""

        for index, track in enumerate(self.trackList):
            track.validate()
            self.audioBuffer[index] = ensureTensorLength(self.audioBuffer[index], track.length * global_consts.batchSize, 0)
        del self.deletions[:]
        self.manager.restart(self.trackList)
        self.ids["pianoRoll"].updateTrack()

    def submitTerminate(self) -> None:
        self.manager.sendChange("terminate", True)

    def submitAddTrack(self, track:dh.Track) -> None:
        self.manager.sendChange("addTrack", True, *track.convert())

    def submitRemoveTrack(self, index:int) -> None:
        self.manager.sendChange("removeTrack", True, index)
    
    def submitDuplicateTrack(self, index:int) -> None:
        self.manager.sendChange("duplicateTrack", True, index)
    
    def submitChangeVB(self, index:int, path:str) -> None:
        self.manager.sendChange("changeVB", True, index, path)
    
    def submitNodegraphUpdate(self) -> None:
        #placeholder until NodeGraph is fully implemented
        self.manager.sendChange("nodeUpdate", True, self.activeTrack, None)
    
    def submitEnableParam(self, name:str) -> None:
        self.manager.sendChange("enableParam", True, self.activeTrack, name)
    
    def submitDisableParam(self, name:str) -> None:
        self.manager.sendChange("disableParam", True, self.activeTrack, name)
    
    def submitBorderChange(self, final:bool, index:int, data) -> None:
        self.manager.sendChange("changeInput", final, self.activeTrack, "borders", index, data)
    
    def submitNamedParamChange(self, final:bool, param, index:int, data) -> None:
        self.manager.sendChange("changeInput", final, self.activeTrack, param, index, data)
    
    def submitNamedPhonParamChange(self, final:bool, param, index:int, data) -> None:
        self.manager.sendChange("changeInput", final, self.activeTrack, param, index, data)
    
    def submitParamChange(self, final:bool, param, index:int, data) -> None:
        self.manager.sendChange("changeInput", final, self.activeTrack, param, index, data)
    
    def submitOffset(self, final:bool, index:int, offset:int) -> None:
        self.manager.sendChange("offset", final, self.activeTrack, index, offset)

    def submitChangeLength(self, final:bool, length:int) -> None:
        self.manager.sendChange("changeLength", final, self.activeTrack, length)
    
    def submitFinalize(self) -> None:
        self.manager.sendChange("finalize", True)
    
    def updateRenderStatus(self, track:int, index:int, value:int) -> None:
        """updates the visual representation of the rendering progress of the note at index index of track track"""

        for i in self.deletions:
            if i == track:
                return None
            elif i < track:
                track -= 1
            #update code here
    
    def updateVolume(self, index:int, volume:float) -> None:
        """updates the volume of the track at position index of the track list"""

        self.trackList[index].volume = volume
    
    def updateAudioBuffer(self, track:int, index:int, data:torch.Tensor) -> None:
        """updates the audio buffer. The data of the track at position track of the trackList is updated with the tensor Data, starting from position index"""

        for i in self.deletions:
            if i == track:
                return None
            elif i < track:
                track -= 1
        self.audioBuffer[track][index:index + len(data)] = data
    
    @mainthread
    def movePlayhead(self, position:int) -> None:
        """sets the position of the playback head in the piano roll UI"""

        self.ids["pianoRoll"].changePlaybackPos(position)
    
    def play(self, state:bool = None) -> None:
        """starts or stops audio playback.
        If state is true, starts playback, if state is false, stops playback. If state is None or not given, starts playback if it is not already in progress, and stops playback if it is."""

        if state == None:
            state = not(self.playing)
        if state == True:
            self.ids["playButton"].state = "down"
            self.audioStream.start()
        if state == False:
            self.ids["playButton"].state = "normal"
            self.audioStream.stop()
        self.playing = state
    
    def playCallback(self, outdata, frames, time, status) -> None:
        """callback function used for updating the driver audio stream with new data from the audio buffer during playback"""

        if self.playing:
            newBufferPos = self.mainAudioBufferPos + global_consts.audioBufferSize
            mainAudioBuffer = torch.zeros((int(newBufferPos - self.mainAudioBufferPos),))
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

    def noteToPitch(self, data:torch.Tensor) -> torch.Tensor:
        """Utility function for converting the y position of a note to its corresponding pitch, following the MIDI standard."""

        #return torch.full_like(data, global_consts.sampleRate) / (torch.pow(2, (data - torch.full_like(data, 69)) / torch.full_like(data, 12)) * 440)
        return torch.full_like(data, global_consts.sampleRate) / (torch.pow(2, (data - torch.full_like(data, 69 - 36)) / torch.full_like(data, 12)) * 440)

    def recalculateBasePitch(self, index:int, oldStart:float, oldEnd:float) -> None:
        """recalculates the base pitch curve after the note at position index of the active track has been modified. oldStart and oldEnd are the start and end of the note before the transformation leading to the function call."""

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
        vibratoSpeed = (self.trackList[self.activeTrack].vibratoSpeed[start:end] / 3. + 0.66) * global_consts.maxVibratoSpeed
        vibratoStrength = (self.trackList[self.activeTrack].vibratoStrength[start:end] / 2. + 0.5) * global_consts.maxVibratoStrength
        vibratoCurve = torch.cumsum(vibratoSpeed.to(torch.float32), 0)
        if previousHeight != None:
            vibratoCurve -= torch.sum(vibratoSpeed[:transitionPoint1 + math.ceil(transitionLength1 / 2) - start])
        vibratoCurve = vibratoStrength * torch.sin(vibratoCurve)
        vibratoCurve *= torch.pow(torch.sin(torch.linspace(0, math.pi, end - start + 1)[:-1]), torch.tensor([global_consts.vibratoEnvelopeExp,], device = self.trackList[self.activeTrack].basePitch.device))
        self.trackList[self.activeTrack].basePitch[start:end] += vibratoCurve
        self.trackList[self.activeTrack].pitch[start:end] = self.trackList[self.activeTrack].basePitch[start:end] + torch.heaviside(self.trackList[self.activeTrack].basePitch[start:end], torch.ones_like(self.trackList[self.activeTrack].basePitch[start:end])) * pitchDelta
        self.applyPitchChanges(self.trackList[self.activeTrack].pitch[start:end], start)
