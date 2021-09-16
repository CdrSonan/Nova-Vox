from multiprocessing import Manager
import torch.multiprocessing as mp
import NV_Multiprocessing.ResamplerProcess
import NV_Multiprocessing.AiParamProcess
import NV_Multiprocessing.SynthProcess
if __name__ == '__main__':
    mp.freeze_support()
    with mp.Manager() as manager:
        statusControl = manager.Namespace()
        statusControl.test = "hello"

        resamplerProcess = mp.Process(target=NV_Multiprocessing.ResamplerProcess.resamplerProcess, args=(statusControl), daemon = True)
        resamplerProcess.start()
        aiParamProcess = mp.Process(target=NV_Multiprocessing.AiParamProcess.aiParamProcess, args=(statusControl), daemon = True)
        aiParamProcess.start()
        synthProcess = mp.Process(target=NV_Multiprocessing.SynthProcess.synthProcess, args=(statusControl), daemon = True)
        synthProcess.start()