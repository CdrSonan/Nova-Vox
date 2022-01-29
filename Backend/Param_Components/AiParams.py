class AiParam:
    def __init__(self, filepath = None):
        pass

class AiParamStack:
    def __init__(self, params = []):
        self.params = params
        self.enabled = []
        for i in self.params:
            self.enabled.append(True)
    def addParam(self, filepath):
        self.params.append(AiParam(filepath))
        self.enabled.append(True)
    def removeParam(self, index):
        del self.params[index]
        del self.enabled[index]
    def enableParam(self, index):
        self.enabled[index] = True
    def disableParam(self, index):
        self.enabled[index] = False