class AiParam:
    """currently unused container class for an AI-driven parameter"""
    
    def __init__(self, filepath = None):
        pass

class AiParamStack:
    """currently unused class for holding and managing an execution stack of AI-driven parameters. Will be replaced by appropriate container for node tree."""

    def __init__(self, params):
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