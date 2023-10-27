#Copyright 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from kivy.base import ExceptionHandler, ExceptionManager
from traceback import print_exc

class ErrorHandler(ExceptionHandler):
    def handle_exception(self, inst):
        print("Attempting error recovery for UI main process error:")
        print_exc()
        global middleLayer
        from UI.code.editor.Main import middleLayer
        try:
            middleLayer.validate()
        except:
            "Recovery failed due to:"
            print_exc()
            return ExceptionManager.RAISE
        return ExceptionManager.PASS

def handleRendererException(exception):
    print("Attempting error recovery for renderer error:")
    print(exception)
    global middleLayer
    from UI.code.editor.Main import middleLayer
    try:
        middleLayer.validate()
    except:
        "Recovery failed due to:"
        print_exc()

def handleMainException(exception):
    print("Attempting error recovery for non-UI main process error:")
    global middleLayer
    from UI.code.editor.Main import middleLayer
    try:
        print_exc()
    except:
        print(exception)
    try:
        middleLayer.validate()
    except:
        "Recovery failed due to:"
        print_exc()