class AudioSample:
    """class for holding a single recording audio sample and processing it for usage in a Voicebank.
    
    Attributes:
        filepath: original filepath of the audio sample. used for reloading files.
        
        waveform: original audio waveform
        
        pitchDeltas: duration of each vocal chord vibration in data points
        
        pitchBorders: borders between each vocal chord vibration in data points measured from sample start
        
        pitch: average pitch of the sample given as wavelength in data points
        
        spectra: deviations of the audio spectrum from the average for each stft batch
        
        spectrum: audio spectrum averaged across the entire sample
        
        excitation: unvoiced excitation signal
        
        voicedExcitation: voiced excitation signal
        
        _voicedExcitations: please do not access. Temp container used during excitation signal calculation
        
        expectedPitch: estimated pitch of the sample in Hz. Must be manually filled and is used during calculatePitch() calls.
        
        searchRange: pitch frequency search range relative to expectedPitch. Should be a value between 0 and 1. Must be manually filled and is used during calculatePitch() calls.
        
        filterWidth: filter width used for spectral calculation. Must be manually filled with a positive Integer and is used during calculateSpectra() calls.
        
        voicedIterations: spectral filtering iterations used for voiced component. Must be manually filled with a positive Integer and is used during calculateSpectra() calls.
        
        unvoicedIterations: spectral filtering iterations used for unvoiced component. Must be manually filled with a positive Integer and is used during calculateSpectra() calls.
    
    Methods:
        __init__: Constructor for initialising an AudioSample based on an audio file
        
        calculatePitch(): Method for calculating pitch data based on previously set attributes
        
        calculateSpectra(): Method for calculating spectral and excitation data based on previously set attributes"""
        
        
    def __init__(self, filepath):
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
        self.sampleRate = loadedData[1]
        del loadedData
        transform = torchaudio.transforms.Resample(self.sampleRate, global_consts.sampleRate)
        self.waveform = transform(self.waveform)
        del transform
        del self.sampleRate
        self.pitchDeltas = torch.tensor([], dtype = int)
        self.pitchBorders = torch.tensor([], dtype = int)
        self.pitch = torch.tensor([0], dtype = int)
        self.spectra = torch.tensor([[]], dtype = float)
        self.spectrum = torch.tensor([], dtype = float)
        self.excitation = torch.tensor([], dtype = float)
        self.voicedExcitation = torch.tensor([], dtype = float)
        self._voicedExcitations = torch.tensor([], dtype = float)
        self.breathinessCompensation = 1.
        
        self.expectedPitch = 249.
        self.searchRange = 0.2
        self.voicedIterations = 10
        self.unvoicedIterations = 20
        
    def calculatePitch(self):
        """Method for calculating pitch data based on previously set attributes expectedPitch and searchRange.
        
        Arguments:
            None
            
        Returns:
            None
            
        The pitch calculation uses 0-transitions to determine the borders between vocal chord vibrations. The algorithm searches for such transitions around expectedPitch (should be a value in Hz),
        with the range around it being defined by searchRange (should be a value between 0 and 1), which is interpreted as a percentage of the wavelength of expectedPitch.
        The function fills the pitchDeltas, pitchBorders and pitch properties."""
        
        
        batchSize = math.floor((1. + self.searchRange) * global_consts.sampleRate / self.expectedPitch)
        lowerSearchLimit = math.floor((1. - self.searchRange) * global_consts.sampleRate / self.expectedPitch)
        batchStart = 0
        while batchStart + batchSize <= self.waveform.size()[0] - batchSize:
            sample = torch.index_select(self.waveform, 0, torch.linspace(batchStart, batchStart + batchSize, batchSize, dtype = int))
            zeroTransitions = torch.tensor([], dtype = int)
            for i in range(lowerSearchLimit, batchSize):
                if (sample[i-1] < 0) and (sample[i] > 0):
                    zeroTransitions = torch.cat([zeroTransitions, torch.tensor([i])], 0)
            error = math.inf
            delta = math.floor(global_consts.sampleRate / self.expectedPitch)
            for i in zeroTransitions:
                shiftedSample = torch.index_select(self.waveform, 0, torch.linspace(batchStart + i.item(), batchStart + batchSize + i.item(), batchSize, dtype = int))
                newError = torch.sum(torch.pow(sample - shiftedSample, 2))
                if error > newError:
                    delta = i.item()
                    error = newError
            self.pitchDeltas = torch.cat([self.pitchDeltas, torch.tensor([delta])])
            batchStart += delta
        nBatches = self.pitchDeltas.size()[0]
        self.pitchBorders = torch.zeros(nBatches + 1, dtype = int)
        for i in range(nBatches):
            self.pitchBorders[i+1] = self.pitchBorders[i] + self.pitchDeltas[i]
        self.pitch = torch.mean(self.pitchDeltas.float()).int()
        del batchSize
        del lowerSearchLimit
        del batchStart
        del sample
        del zeroTransitions
        del error
        del delta
        del shiftedSample
        del newError
        del nBatches
        
    def calculateSpectra(self):
        """Method for calculating spectral data based on previously set attributes filterWidth, voicedIterations and unvoicedIterations.
        
        Arguments:
            None
            
        Returns:
            None
            
        The spectral calculation uses an adaptation of the True Envelope Estimator. It works with a fixed smoothing range determined by filterWidth (inta fourier space data points).
        The algorithm first runs an amount of filtering iterations determined by voicedIterations, selectively saves the peaking frequencies of the signal into _voicedExcitations, 
        then runs the filtering algorithm again a number of iterations determined by unvoicedIterations.
        The function fills the spectrum, spectra and _voicedExcitations properties."""
        
        
        window = torch.hann_window(global_consts.tripleBatchSize * global_consts.filterBSMult)
        signals = torch.stft(self.waveform, global_consts.tripleBatchSize * global_consts.filterBSMult, hop_length = global_consts.batchSize * global_consts.filterBSMult, win_length = global_consts.tripleBatchSize * global_consts.filterBSMult, window = window, return_complex = True, onesided = True)
        signals = torch.transpose(signals, 0, 1)
        signalsAbs = signals.abs()

        spectralFilterWidth = torch.max(torch.floor(global_consts.tripleBatchSize * global_consts.filterBSMult / self.pitch), torch.Tensor([1])).int()
        
        workingSpectra = torch.sqrt(signalsAbs)
        self.spectra = workingSpectra.clone()
        for j in range(self.voicedIterations):
            for i in range(spectralFilterWidth):
                self.spectra = torch.roll(workingSpectra, -i, dims = 1) + self.spectra + torch.roll(workingSpectra, i, dims = 1)
            self.spectra = self.spectra / (2 * spectralFilterWidth + 1)
            workingSpectra = torch.min(workingSpectra, self.spectra)
            self.spectra = workingSpectra
        

        self._voicedExcitations = signals.clone()
        self._voicedExcitations *= torch.gt(torch.sqrt(signalsAbs), self.spectra)

        excitationAbs = signalsAbs
        voicedExcitationAbs = self._voicedExcitations.abs()
        self.excitation = torch.transpose(torch.sqrt(signals) * (excitationAbs - voicedExcitationAbs), 0, 1)
        self.voicedExcitation = torch.transpose(self._voicedExcitations, 0, 1)

        self.excitation = torch.istft(self.excitation, global_consts.tripleBatchSize * global_consts.filterBSMult, hop_length = global_consts.batchSize * global_consts.filterBSMult, win_length = global_consts.tripleBatchSize * global_consts.filterBSMult, window = window, onesided = True)
        self.voicedExcitation = torch.istft(self.voicedExcitation, global_consts.tripleBatchSize * global_consts.filterBSMult, hop_length = global_consts.batchSize * global_consts.filterBSMult, win_length = global_consts.tripleBatchSize * global_consts.filterBSMult, window = window, onesided = True)

        window = torch.hann_window(global_consts.tripleBatchSize)

        self.excitation = torch.stft(self.excitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)
        self.voicedExcitation = torch.stft(self.voicedExcitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)

        signals = torch.stft(self.waveform, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)
        signals = torch.transpose(signals, 0, 1)
        signalsAbs = signals.abs()

        workingSpectra = torch.sqrt(signalsAbs)
        self.spectra = torch.full_like(workingSpectra, -float("inf"), dtype=torch.float)
                
        for j in range(self.unvoicedIterations):
            workingSpectra = torch.max(workingSpectra, self.spectra)
            self.spectra = workingSpectra
            for i in range(spectralFilterWidth):
                self.spectra = torch.roll(workingSpectra, -i, dims = 1) + self.spectra + torch.roll(workingSpectra, i, dims = 1)
            self.spectra = self.spectra / (2 * spectralFilterWidth + 1)

        self.spectrum = torch.mean(self.spectra, 0)
        for i in range(self.spectra.size()[0]):
            self.spectra[i] = self.spectra[i] - self.spectrum

        self.voicedExcitation = self.voicedExcitation / torch.transpose(torch.square(self.spectrum + self.spectra)[0:self.voicedExcitation.size()[1]], 0, 1)
        self.excitation = torch.transpose(self.excitation, 0, 1) / torch.square(self.spectrum + self.spectra)[0:self.excitation.size()[1]]

        del window
        del signals
        del workingSpectra
        
class loadedAudioSample:
    """A stripped down version of AudioSample only holding the data required for synthesis.
    
    Attributes:
        pitchDeltas: duration of each vocal chord vibration in data points
        
        pitch: average pitch of the sample given as wavelength in data points
        
        spectra: deviations of the audio spectrum from the average for each stft batch
        
        spectrum: audio spectrum averaged across the entire sample
        
        excitation: unvoiced excitation signal
        
        voicedExcitation: voiced excitation signal
        
    Methods:
        __init__: Constructor for initialising the class based on an AudioSample object, discarding all extraneous data."""
    
    
    def __init__(self, audioSample):
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
        
class RelLoss(nn.Module):
    """function for calculating relative loss values between target and actual Tensor objects. Designed to be used with AI optimizers.
    
    Attributes:
        None
        
    Methods:
        __init__: basic class constructor
        
        forward: calculates relative loss based on input and target tensors after successful initialisation."""
    
    
    def __init__(self, weight=None, size_average=True):
        """basic class constructor.
        
        Arguments:
            weight: required by PyTorch in some situations. Unused.
            
            size_average: required by PyTorch in some situations. Unused.
            
        Returns:
            None"""
        
        
        super(RelLoss, self).__init__()
 
    def forward(self, inputs, targets):  
        """calculates relative loss based on input and target tensors after successful initialisation.
        
        Arguments:
            inputs: AI-generated input Tensor
            
            targets: target Tensor
            
        Returns:
            Relative error value calculated from the difference between input and target Tensor as Float"""
        
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        differences = torch.abs(inputs - targets)
        refs = torch.abs(targets)
        out = (differences / refs).sum() / inputs.size()[0]