class AiParamStack:
    def __init__(self, params = []):
        self.params = params
        self.enabled = []
        for i in self.params:
            self.enabled.append(True)