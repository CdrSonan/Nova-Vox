import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")
import global_consts

class Synthesizer():
    """Class providing a minimal implementation of the ESPER synthesizer. Provided for completeness of the ESPER Python module and not used in Nova-Vox

    Methods:
        synthesize (static method): synthesizes audio based on the ESPER speech model
    """
    
    def __init__(self) -> None:
        pass

    @staticmethod
    def synthesize(breathiness:torch.Tensor, spectrum:torch.Tensor, excitation:torch.Tensor, voicedExcitation:torch.Tensor, filepath:str = None) -> torch.Tensor:
        """synthesizes audio based on the ESPER speech model. The required parameters can be easily extracted from a VocalSequence object.
        
        Arguments:
            breathiness: Tensor containing breathiness values from -1 to 1. Must be broadcastable to the shape of the signal.

            spectrum: Tensor containing a spectrum or sequence thereof as returned by the ESPER analysis functions. Must be broadcastable to the shape of the signal.

            excitation: Tensor containing the unvoiced excitation signal as returned by the ESPER analysis functions.

            voicedExcitation: Tensor containing the voiced excitation signal as returned by the ESPER analysis functions.

            filepath:
        
        Returns:
            Tensor containing the synthesized audio"""
            
        Window = torch.hann_window(global_consts.tripleBatchSize)
        
        returnSignal = torch.stft(voicedExcitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, return_complex = True, onesided = True)
        unvoicedSignal = torch.stft(excitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, return_complex = True, onesided = True)
        
        breathinessCompensation = torch.sum(torch.abs(returnSignal), 0) / torch.sum(torch.abs(unvoicedSignal), 0) * global_consts.breCompPremul
        breathinessUnvoiced = 1. + breathiness * breathinessCompensation[0:-1] * torch.gt(breathiness, 0) + breathiness * torch.logical_not(torch.gt(breathiness, 0))
        breathinessVoiced = 1. - (breathiness * torch.gt(breathiness, 0))
        returnSignal = returnSignal[:, 0:-1] * torch.transpose(spectrum, 0, 1) * breathinessVoiced
        unvoicedSignal = unvoicedSignal[:, 0:-1] * torch.transpose(spectrum, 0, 1) * breathinessUnvoiced

        returnSignal = torch.istft(returnSignal, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, onesided=True)
        unvoicedSignal = torch.istft(unvoicedSignal, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, onesided=True)
        returnSignal += unvoicedSignal

        del Window

        if filepath != None:
            torchaudio.save(filepath, torch.unsqueeze(returnSignal.detach(), 0), global_consts.sampleRate, format="wav", encoding="PCM_S", bits_per_sample=32)

        return returnSignal
        