#Copyright 2022-2024 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

import logging
import torch.multiprocessing as mp
from torch import Tensor
import Backend.NV_Multiprocessing.RenderProcess
from Backend.NV_Multiprocessing.Interface import SequenceStatusControl, InputChange, StatusChange
from Localization.editor_localization import getLanguage
from API.Addon import override

class RenderManager():
    """Class responsible for starting, terminating, and managing the connection to a rendering process. Uses two queues for managing data flow between processes and can create a rendering process in an arbitrary state.
    
    Methods:
        __init__: constructor for a rendering process containing the tracks, voicebanks and parameters already present on the main process

        receiveChange: receives a StatusChange object from the rendering process if available
        
        sendChange: sends an InputChange object to the rendering process
        
        restart: restarts the rendering process, and re-renders all audio
        
        stop: terminates the rendering process"""


    def __init__(self, trackList:list) -> None:
        """Initializes a RenderManager object and its associated rendering process based on the data alread present on the main process.
        
        Arguments:
            sequenceList: a list or other iterable of VocalSequence objects, and the main source of data for the rendering process.

            voicebankList: a list or other iterable of Voicebank or LiteVoicebank objects with the same length as sequenceList. Holds the voicebanks used for synthesis by the rendering process.

            NodeGraphList: placeholder for future list of serialized NodeGraph objects
        
        Returns:
            None"""


        statusControl = []
        voicebankList, NodeGraphList, sequenceList = self.batchConvert(trackList)
        self.connection = mp.Queue(0)
        self.remoteConnection = mp.Queue(0)
        for i in sequenceList:
            statusControl.append(SequenceStatusControl(i))
        self.renderProcess = mp.Process(target=Backend.NV_Multiprocessing.RenderProcess.renderProcess, args=(statusControl, voicebankList, NodeGraphList, sequenceList, self.connection, self.remoteConnection), name = getLanguage()["render_process_name"], daemon = True)
        self.renderProcess.start()

    def receiveChange(self) -> StatusChange:
        """non-blocking method for receiving a change from the rendering process.
        
        Arguments:
            None
            
        Returns:
            StatusChange if a new change is available, None otherwise"""


        try:
            return self.remoteConnection.get_nowait()
        except:
            return None
            
    def sendChange(self, type:str, final:bool = True, *data) -> None:
        """method for sending an InputChange object containing arbitrary data to the rendering process"""

        dataOut = []
        for i in data:
            if i.__class__ == Tensor:
                dataOut.append(i.clone())
            else:
                dataOut.append(i)
        dataOut = tuple(dataOut)
        self.connection.put(InputChange(type, final, *dataOut), True)
        logging.info(", ".join(("sent packet", type, final.__repr__(), data.__repr__())))

    @override
    def restart(self, trackList:list) -> None:
        """Method for restarting the rendering process. The required data is fetched again from the main process. This is to prevent any possible issues with the data held by the rendering process from persisting.
        
        Arguments:
            trackList: a list or other iterable of Track objects, and the main source of data for the rendering process.

        Returns:
            None"""


        global middleLayer
        from UI.editor.Main import middleLayer
        self.stop()
        print("rendering subprocess restarting...")
        self.connection.close()
        self.remoteConnection.close()
        del self.connection
        del self.remoteConnection
        for i in range(len(middleLayer.audioBuffer)):
            middleLayer.audioBuffer[i] *= 0
        self.__init__(trackList)

    @override
    def stop(self) -> None:
        """simple method for safely terminating the rendering process"""
        self.renderProcess.terminate()

    def batchConvert(self, trackList:list) -> tuple:
        """utility function for simultaneously converting multiple tracks to vocalSequence representation"""

        voicebankList = []
        NodeGraphList = []
        sequenceList = []
        for i in trackList:
            info = i.convert()
            voicebankList.append(info[0])
            NodeGraphList.append(info[1])
            sequenceList.append(info[2])
        return voicebankList, NodeGraphList, sequenceList