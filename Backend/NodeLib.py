from API.Node import *

class addFloatNode(Node):
    def __init__(self, **kwargs) -> None:
        inputs = {"A": Float, "B":Float}
        outputs = {"Result": Float}
        def func(A, B):
            return {"Result": A + B}
        super().__init__(inputs, outputs, func, False, **kwargs)

    @staticmethod
    def name() -> str:
        return ["Math", "Float", "Add"]