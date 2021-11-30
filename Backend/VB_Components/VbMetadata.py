from PIL import Image
class VbMetadata:
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
        #self.image = Image.new('RGB', (200, 200), color=(55, 55, 55))
        self.image = Image.open("UI/assets/TrackList/SingerGrey04.png").resize((200, 200), resample = 1)
        self.version = "1.0"
        self.description = ""
        self.license = ""