class Synthesizer:
    def __init__(self, sampleRate):
        self.sampleRate = sampleRate
        self.returnSignal = torch.tensor([], dtype = float)
        
    def synthesize(self, breathiness, spectrum, excitation, voicedExcitation):
        Window = torch.hann_window(global_consts.tripleBatchSize)
        
        self.returnSignal = torch.stft(voicedExcitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, return_complex = True, onesided = True)
        unvoicedSignal = torch.stft(excitation, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, return_complex = True, onesided = True)
        
        breathinessCompensation = torch.sum(torch.abs(self.returnSignal), 0) / torch.sum(torch.abs(unvoicedSignal), 0) * global_consts.breCompPremul
        breathinessUnvoiced = 1. + breathiness * breathinessCompensation[0:-1] * torch.gt(breathiness, 0) + breathiness * torch.logical_not(torch.gt(breathiness, 0))
        breathinessVoiced = 1. - (breathiness * torch.gt(breathiness, 0))
        self.returnSignal = self.returnSignal[:, 0:-1] * torch.transpose(spectrum, 0, 1) * breathinessVoiced
        unvoicedSignal = unvoicedSignal[:, 0:-1] * torch.transpose(spectrum, 0, 1) * breathinessUnvoiced

        self.returnSignal = torch.istft(self.returnSignal, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, onesided=True)
        unvoicedSignal = torch.istft(unvoicedSignal, global_consts.tripleBatchSize, hop_length = global_consts.batchSize, win_length = global_consts.tripleBatchSize, window = Window, onesided=True)
        self.returnSignal += unvoicedSignal

        del Window
        
    def save(self, filepath):
        torchaudio.save(filepath, torch.unsqueeze(self.returnSignal.detach(), 0), global_consts.sampleRate, format="wav", encoding="PCM_S", bits_per_sample=32)