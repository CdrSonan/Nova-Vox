#Copyright 2022 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from PIL import Image

class VbMetadata():
    """Helper class for holding Voicebank metadata. To be expanded.
    
    Attributes:
        name: The name of the Voicebank
        
    Methods:
        __init__: basic class constructor"""
        
        
    def __init__(self):
        """basic class constructor.
        
        Arguments:
            None
            
        Returns:
            None"""
            
            
        self.name = ""
        self.sampleRate = 48000
        self.image = Image.open("UI/assets/TrackList/SingerGrey04.png").resize((200, 200), resample = 1)
        self.version = "1.0"
        self.description = ""
        self.license = ""
