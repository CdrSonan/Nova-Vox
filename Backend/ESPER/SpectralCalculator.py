from math import inf
import torch
import global_consts

def calculateSpectra(audioSample):
        """Method for calculating spectral data based on previously set attributes filterWidth, voicedIterations and unvoicedIterations.
        
        Arguments:
            None
            
        Returns:
            None
            
        The spectral calculation uses an adaptation of the True Envelope Estimator. It works with a fixed smoothing range determined by filterWidth (inta fourier space data points).
        The algorithm first runs an amount of filtering iterations determined by voicedIterations, selectively saves the peaking frequencies of the signal into _voicedExcitations, 
        then runs the filtering algorithm again a number of iterations determined by unvoicedIterations.
        The function fills the spectrum, spectra and _voicedExcitations properties."""
        SiLU = torch.nn.SiLU()# softplus?
        threshold = torch.nn.Threshold(0.001, 0.001)
        #perhaps lower FilterTEEMult for voiced/unvoiced separation and increase UI filter value instead
        window = torch.hann_window(global_consts.tripleBatchSize * global_consts.filterBSMult)
        spectralFilterWidth = torch.max(torch.floor(global_consts.tripleBatchSize * global_consts.filterBSMult * global_consts.filterTEEMult / audioSample.pitch), torch.Tensor([1])).int().item()
        signals = torch.stft(audioSample.waveform, global_consts.tripleBatchSize * global_consts.filterBSMult, hop_length = global_consts.batchSize * global_consts.filterBSMult, win_length = global_consts.tripleBatchSize * global_consts.filterBSMult, window = window, return_complex = True, onesided = True)
        signals = torch.transpose(signals, 0, 1)
        signalsAbs = signals.abs()
        """
        spectralFilterWidth = torch.max(torch.floor(global_consts.tripleBatchSize * global_consts.filterBSMult / audioSample.pitch), torch.Tensor([1])).int().item()
        workingSpectra = torch.sqrt(signalsAbs)
        audioSample.spectra = workingSpectra.clone()
        for j in range(audioSample.voicedIterations):
            for i in range(spectralFilterWidth):
                audioSample.spectra = torch.roll(workingSpectra, -i, dims = 1) + audioSample.spectra + torch.roll(workingSpectra, i, dims = 1)
            audioSample.spectra = audioSample.spectra / (2 * spectralFilterWidth + 1)
            workingSpectra = torch.min(workingSpectra, audioSample.spectra)
            audioSample.spectra = workingSpectra
        """
        signalsAbs = torch.sqrt(signalsAbs)
        audioSample.spectra = signalsAbs.clone()

        audioSample.spectra = torch.fft.rfft(audioSample.spectra, dim = 1)
        cutoffWindow = torch.zeros(audioSample.spectra.size()[1])
        cutoffWindow[0:spectralFilterWidth] = 1.
        cutoffWindow[spectralFilterWidth] = 0.5
        audioSample.spectra = threshold(torch.fft.irfft(cutoffWindow * audioSample.spectra, dim = 1, n = global_consts.halfTripleBatchSize * global_consts.filterBSMult + 1))
        

        audioSample._voicedExcitations = signals.clone()
        audioSample._voicedExcitations *= torch.gt(signalsAbs, audioSample.spectra * audioSample.voicedFilter)

        excitationAbs = signalsAbs
        voicedExcitationAbs = torch.sqrt(audioSample._voicedExcitations.abs())
        audioSample.excitation = torch.transpose(torch.sqrt(signals) * (excitationAbs - voicedExcitationAbs), 0, 1)
        audioSample.voicedExcitation = torch.transpose(audioSample._voicedExcitations, 0, 1)

        audioSample.excitation = torch.istft(audioSample.excitation, global_consts.tripleBatchSize * global_consts.filterBSMult, hop_length = global_consts.batchSize * global_consts.filterBSMult, win_length = global_consts.tripleBatchSize * global_consts.filterBSMult, window = window, onesided = True)
        audioSample.voicedExcitation = torch.istft(audioSample.voicedExcitation, global_consts.tripleBatchSize * global_consts.filterBSMult, hop_length = global_consts.batchSize * global_consts.filterBSMult, win_length = global_consts.tripleBatchSize * global_consts.filterBSMult, window = window, onesided = True)

        window = torch.hann_window(global_consts.tripleBatchSize)

        audioSample.excitation = torch.stft(audioSample.excitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)
        audioSample.voicedExcitation = torch.stft(audioSample.voicedExcitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)

        spectralFilterWidth = torch.max(torch.floor(global_consts.tripleBatchSize * global_consts.filterTEEMult / audioSample.pitch), torch.Tensor([1])).int().item()

        signals = torch.stft(audioSample.waveform, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, return_complex = True, onesided = True)
        signals = torch.transpose(signals, 0, 1)
        signalsAbs = signals.abs()
        signalsAbs = torch.sqrt(signalsAbs)
        audioSample.spectra = signalsAbs.clone()
        for j in range(audioSample.unvoicedIterations):
            audioSample.spectra = torch.maximum(audioSample.spectra, signalsAbs)
            audioSample.spectra = torch.fft.rfft(audioSample.spectra, dim = 1)
            cutoffWindow = torch.zeros(audioSample.spectra.size()[1])
            cutoffWindow[0:int(spectralFilterWidth / 2)] = 1.
            cutoffWindow[int(spectralFilterWidth / 2):spectralFilterWidth] = torch.linspace(1, 0, spectralFilterWidth - int(spectralFilterWidth / 2))
            audioSample.spectra = torch.fft.irfft(cutoffWindow * audioSample.spectra, dim = 1, n = global_consts.halfTripleBatchSize + 1)
        audioSample.spectra = threshold(audioSample.spectra)


        spectralFilterWidth = torch.max(torch.floor(global_consts.tripleBatchSize / audioSample.pitch), torch.Tensor([1])).int()
        workingSpectra = signalsAbs.clone()
        workingSpectra = torch.cat((workingSpectra, torch.tile(torch.unsqueeze(workingSpectra[:, -1], 1), (1, audioSample.unvoicedIterations))), 1)
        spectra = workingSpectra.clone()
        for j in range(audioSample.unvoicedIterations):
            for i in range(spectralFilterWidth):
                spectra = torch.roll(workingSpectra, -i, dims = 1) + spectra + torch.roll(workingSpectra, i, dims = 1)
            spectra = spectra / (2 * spectralFilterWidth + 1)
            workingSpectra = torch.max(workingSpectra, spectra)
            spectra = workingSpectra
        spectra = spectra[:, 0:global_consts.halfTripleBatchSize + 1]
        slope = torch.ones_like(spectra)
        slope[:, global_consts.spectralRolloff2:] = 0.
        slope[:, global_consts.spectralRolloff1:global_consts.spectralRolloff2] = torch.linspace(1, 0, global_consts.spectralRolloff2 - global_consts.spectralRolloff1)
        audioSample.spectra = slope * audioSample.spectra + ((1. - slope) * spectra)


        audioSample.spectrum = torch.mean(audioSample.spectra, 0)
        for i in range(audioSample.spectra.size()[0]):
            audioSample.spectra[i] = audioSample.spectra[i] - audioSample.spectrum

        audioSample.voicedExcitation = audioSample.voicedExcitation / torch.transpose(torch.square(audioSample.spectrum + audioSample.spectra)[0:audioSample.voicedExcitation.size()[1]], 0, 1)
        audioSample.excitation = torch.transpose(audioSample.excitation, 0, 1) / torch.square(audioSample.spectrum + audioSample.spectra)[0:audioSample.excitation.size()[1]]

        audioSample.voicedExcitation = torch.istft(audioSample.voicedExcitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = window, onesided = True)