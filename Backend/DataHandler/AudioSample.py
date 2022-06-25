import torchaudio
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")
import global_consts

class AudioSample():
    """class for holding a single recording audio sample and processing it for usage in a Voicebank.
    
    Attributes:
        filepath: original filepath of the audio sample. used for reloading files.
        
        waveform: original audio waveform
        
        pitchDeltas: duration of each vocal chord vibration in data points
        
        pitch: average pitch of the sample given as wavelength in data points
        
        spectra: deviations of the audio spectrum from the average for each stft batch
        
        spectrum: audio spectrum averaged across the entire sample
        
        excitation: unvoiced excitation signal
        
        voicedExcitation: voiced excitation signal
        
        phases: phase of the waveform in radians for each stft batch

        isVoiced: flag indicating whether the sample is voiced (voicedExcitation is muted for unvoiced samples during ESPER processing)

        isPlosive: flag indicating whether the sample is considered a plosive sound (if possible, plosives retain their original length after border creation during synthesis)
        
        expectedPitch: estimated pitch of the sample in Hz. Must be manually filled and is used during calculatePitch() calls.
        
        searchRange: pitch frequency search range relative to expectedPitch. Should be a value between 0 and 1. Must be manually filled and is used during calculatePitch() calls.
        
        voicedFilter: spectral filtering iterations used for voiced component. Must be manually filled with a positive Integer and is used during calculateSpectra() calls.
        
        unvoicedIterations: spectral filtering iterations used for unvoiced component. Must be manually filled with a positive Integer and is used during calculateSpectra() calls.
    
    Methods:
        __init__: Constructor for initialising an AudioSample based on an audio file
        
        calculatePitch(): Method for calculating pitch data based on previously set attributes
        
        calculateSpectra(): Method for calculating spectral and excitation data based on previously set attributes"""
        
        
    def __init__(self, filepath:str) -> None:
        """Constructor for initialising an AudioSample based on an audio file and desired sample rate.
        
        Arguments:
            filepath: Expects a String that can be interpreted as a filepath to a .wav audio file. Determines the audio file to be loaded into the object.
            
        Returns:
            None
        
        This method initializes the properties used for spectral and excitation calculation to the default values and initialises all attributes with empty objects.
        Loads the selected audio file (based on filepath string) into the waveform property and resamples it to the desired sample rate"""
        
        
        loadedData = torchaudio.load(filepath)
        self.filepath = filepath
        self.waveform = loadedData[0][0]
        sampleRate = loadedData[1]
        del loadedData
        transform = torchaudio.transforms.Resample(sampleRate, global_consts.sampleRate)
        self.waveform = transform(self.waveform)
        del transform
        del sampleRate
        self.pitchDeltas = torch.tensor([], dtype = int)
        self.pitch = torch.tensor([0], dtype = int)
        self.spectra = torch.tensor([[]], dtype = float)
        self.spectrum = torch.tensor([], dtype = float)
        self.excitation = torch.tensor([], dtype = float)
        self.voicedExcitation = torch.tensor([], dtype = float)
        self.phases = torch.tensor([], dtype = float)
        self.isVoiced = True
        self.isPlosive = False
        
        self.expectedPitch = global_consts.defaultExpectedPitch
        self.searchRange = global_consts.defaultSearchRange
        self.voicedFilter = global_consts.defaultVoicedFilter
        self.unvoicedIterations = global_consts.defaultUnvoicedIterations
        
class LiteAudioSample():
    """A stripped down version of AudioSample only holding the data required for synthesis.
    
    Attributes:
        pitchDeltas: duration of each vocal chord vibration in data points
        
        pitch: average pitch of the sample given as wavelength in data points
        
        spectra: deviations of the audio spectrum from the average for each stft batch
        
        spectrum: audio spectrum averaged across the entire sample
        
        excitation: unvoiced excitation signal
        
        voicedExcitation: voiced excitation signal

        phases: phase of the waveform in radians for each stft batch
        
    Methods:
        __init__: Constructor for initialising the class based on an AudioSample object, discarding all extraneous data."""
    
    
    def __init__(self, audioSample) -> None:
        """Constructor for initialising the class based on an AudioSample object, discarding all extraneous data.
        
        Arguments:
            audioSample: AudioSample base object
            
        Returns:
            None"""
            
            
        self.pitchDeltas = audioSample.pitchDeltas
        self.pitch = audioSample.pitch
        self.spectra = audioSample.spectra
        self.spectrum = audioSample.spectrum
        self.excitation = audioSample.excitation
        self.voicedExcitation = audioSample.voicedExcitation
        self.phases = audioSample.phases